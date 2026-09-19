"""Command-line entry point: the only supported way to run anything.

Three working modes, each self-contained:

``mvy ingest``
    Dataset analysis on its own. Produces the yield-point artefact and the
    dataset figures. No yield probability is computed.

``mvy run``
    Yield probability on an existing model. The dataset is never opened, so
    this path does not even import ``h5py``.

``mvy all``
    Ingest, build and run in one go, skipping steps whose inputs are
    unchanged.

Every command accepts ``--set key.path=value`` and ``--dry-run``, so a
configuration can be inspected and adjusted without editing files.

Each command is a thin wrapper: it resolves the config, prepares a run
directory, calls one pipeline function and reports. Anything longer than
that belongs in :mod:`mvyield.pipeline`, where it can be tested without a
command line.
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import contextmanager
from pathlib import Path

from mvyield import __version__
from mvyield.model.bundle import ModelBundle
from mvyield.paths import RunPaths, find_repo_root, resolve_input
from mvyield.settings import CaseConfig, ConfigError, dump_resolved, load_case

log = logging.getLogger("mvy")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a command.

    Returns
    -------
    int
        Process exit code. Configuration and input errors return 2 with a
        one-line message; unexpected failures propagate with a traceback,
        because those are bugs and hiding them helps nobody.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    _setup_logging(args.verbose)

    try:
        return args.handler(args)
    except ConfigError as exc:
        log.error("configuration error: %s", exc)
        return 2
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2


# --- Commands ----------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check the environment, paths and artefacts before a long run."""
    root = find_repo_root()
    print(f"mvyield {__version__}")
    print(f"repository root : {root}")

    print("\noptional dependencies:")
    for module, purpose in (
        ("h5py", "dataset ingest"),
        ("pandas", "dataset ingest"),
        ("meshio", "VTK export for 3D fields"),
        ("torch", "neural yield surface"),
    ):
        print(f"  {module:<8s} {_probe_import(module):<12s} ({purpose})")

    if args.case:
        cfg = load_case(Path(args.case), overrides=args.set)
        print(f"\ncase '{cfg.name}':")
        print(_indent(cfg.describe()))
        _report_inputs(cfg, root)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Mode A: analyse the dataset and export yield points."""
    cfg = _load(args)
    if cfg.dataset.source is None:
        raise ConfigError(
            "'mvy ingest' needs dataset.source, but the case config does not "
            "set it. Add the dataset path, or use 'mvy run' if you only want "
            "to evaluate an existing model."
        )
    if args.dry_run:
        return _dry_run(cfg, "ingest")

    from mvyield.pipeline.run import run_mode_a

    with _run_directory(cfg, "ingest") as paths:
        outcome = run_mode_a(
            cfg,
            paths,
            force=getattr(args, "force", False),
            export=not args.no_export,
            plots=args.plots,
            geometries=_geometries(args),
        )
    return _report_done(outcome)


def cmd_build_model(args: argparse.Namespace) -> int:
    """Fit the requested estimators over the yield-point cloud."""
    cfg = _load(args)
    if args.dry_run:
        return _dry_run(cfg, "build-model")

    from mvyield.pipeline.ingest import dataset_key, resolve_dataset, run_ingest
    from mvyield.pipeline.model import run_build_model

    with _run_directory(cfg, "run") as paths:
        ingest = run_ingest(cfg, paths)
        key = dataset_key(resolve_dataset(cfg, paths), cfg.dataset)
        bundle = run_build_model(cfg, ingest.points, paths, key, force=args.force)

    print(bundle.describe())
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Select hyper-parameters and record them in the model bundle."""
    cfg = _load(args)
    if args.dry_run:
        return _dry_run(cfg, "calibrate")

    from mvyield.pipeline.ingest import dataset_key, resolve_dataset, run_ingest
    from mvyield.pipeline.model import run_calibrate

    with _run_directory(cfg, "run") as paths:
        ingest = run_ingest(cfg, paths)
        key = dataset_key(resolve_dataset(cfg, paths), cfg.dataset)
        bundle = run_calibrate(
            cfg, ingest.points, paths, key, refine=args.refine, verify=args.verify
        )

    print(bundle.describe())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Mode B: evaluate yield probability on a stress field."""
    cfg = _load(args)
    if args.dry_run:
        return _dry_run(cfg, "run")

    from mvyield.pipeline.run import load_model_for, run_evaluate

    with _run_directory(cfg, "run") as paths:
        bundle = load_model_for(cfg, paths)
        outcome = run_evaluate(cfg, bundle, paths, plots=args.plots)
    return _report_done(outcome)


def cmd_all(args: argparse.Namespace) -> int:
    """Modes A, B and the fit between them, skipping unchanged steps."""
    cfg = _load(args)
    if args.dry_run:
        return _dry_run(cfg, "all")

    from mvyield.pipeline.run import run_mode_c

    with _run_directory(cfg, "run") as paths:
        outcome = run_mode_c(
            cfg,
            paths,
            force_ingest=args.force or args.force_ingest,
            force_model=args.force or args.force_model,
            calibrate=not args.no_calibrate,
            plots=args.plots,
        )
    return _report_done(outcome)


def cmd_plot(args: argparse.Namespace) -> int:
    """Redraw figures from a finished run, without recomputing anything."""
    # Importing the viz package registers every plotting function and pulls
    # in matplotlib; that cost belongs here, not in 'mvy run'.
    from mvyield.viz import describe_registry

    if args.list:
        print(describe_registry())
        return 0

    if not args.run_dir:
        log.error("'mvy plot' needs a run directory, or --list")
        return 2

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    from mvyield.pipeline.replot import replot

    written = replot(run_dir, preset=args.plots, formats=args.format)
    print(f"redrew {len(written)} figure file(s) in {run_dir / 'plots'}")
    return 0


def cmd_init_case(args: argparse.Namespace) -> int:
    """Write a starter case config."""
    from mvyield.pipeline.scaffold import write_case

    target = write_case(args.name, args.template, find_repo_root() / "configs" / "cases")
    print(f"wrote {target}\n\nNext: set dataset.source, then\n  mvy doctor --case {target}")
    return 0


def cmd_show_model(args: argparse.Namespace) -> int:
    """Print what a stored model bundle contains, including provenance."""
    bundle = ModelBundle.load(Path(args.path))
    print(bundle.describe())
    return 0


# --- Helpers -----------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser."""
    parser = argparse.ArgumentParser(
        prog="mvy",
        description="Data-driven yield probability from microstructure datasets.",
    )
    parser.add_argument("--version", action="version", version=f"mvyield {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    def add(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        sub.add_argument("-v", "--verbose", action="store_true", help="debug logging")
        sub.set_defaults(handler=handler)
        return sub

    def add_case_options(sub: argparse.ArgumentParser, required: bool = True) -> None:
        sub.add_argument("--case", required=required, help="path to a case YAML file")
        sub.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override a config key, e.g. --set model.methods=[kde_slice]",
        )
        sub.add_argument(
            "--dry-run",
            action="store_true",
            help="show the resolved configuration and the planned steps, then stop",
        )
        sub.add_argument("--seed", type=int, help="override the random seed")

    doctor = add("doctor", cmd_doctor, "Check the environment, paths and artefacts.")
    doctor.add_argument("--case", help="also validate this case config")
    doctor.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    ingest = add("ingest", cmd_ingest, "Mode A: analyse the dataset only.")
    add_case_options(ingest)
    ingest.add_argument(
        "--no-export",
        action="store_true",
        help="draw the dataset figures without writing the yield-point artefact",
    )
    ingest.add_argument("--plots", help="plot preset override")
    ingest.add_argument("--force", action="store_true", help="re-analyse even if cached")
    ingest.add_argument(
        "--geometries",
        metavar="IDS",
        help="analyse only these RVE, comma separated, e.g. 0,1,2",
    )

    build = add("build-model", cmd_build_model, "Fit a yield model from yield points.")
    add_case_options(build)
    build.add_argument("--force", action="store_true", help="rebuild even if up to date")

    calibrate = add("calibrate", cmd_calibrate, "Select and record hyper-parameters.")
    add_case_options(calibrate)
    calibrate.add_argument(
        "--refine", action="store_true", help="refine around the best point"
    )
    calibrate.add_argument("--verify", action="store_true", help="re-score the selected value")

    run = add("run", cmd_run, "Mode B: yield probability on a stress field.")
    add_case_options(run)
    run.add_argument("--plots", help="plot preset override")

    run_all = add("all", cmd_all, "Mode C: ingest, build and run in one pass.")
    add_case_options(run_all)
    run_all.add_argument("--force", action="store_true", help="redo every step")
    run_all.add_argument(
        "--force-ingest", action="store_true", help="redo the dataset analysis"
    )
    run_all.add_argument("--force-model", action="store_true", help="refit the model")
    run_all.add_argument("--plots", help="plot preset override")
    run_all.add_argument(
        "--no-calibrate",
        action="store_true",
        help="use grid midpoints instead of searching, for a fast smoke run",
    )

    plot = add("plot", cmd_plot, "Redraw figures from a finished run.")
    plot.add_argument("run_dir", nargs="?", help="run directory to redraw")
    plot.add_argument("--plots", help="plot preset")
    plot.add_argument("--format", action="append", default=[], help="output format")
    plot.add_argument("--list", action="store_true", help="list registered plots and exit")

    init = add("init-case", cmd_init_case, "Write a starter case config.")
    init.add_argument("name", help="name of the new case")
    init.add_argument("--from", dest="template", default="kirsch", choices=("kirsch", "ansys"))

    show = add("show-model", cmd_show_model, "Print the contents of a model bundle.")
    show.add_argument("path", help="path to a .model.npz file")

    return parser


@contextmanager
def _run_directory(cfg: CaseConfig, kind: str):
    """Prepare a run directory, record the config, and log into it.

    The resolved configuration is written before any work starts, so a run
    that fails half way still says exactly what it was asked to do. The log
    handler is detached afterwards; leaving it attached would make a second
    run in the same process write into the first one's directory.
    """
    from mvyield.pipeline.report import attach_log_file
    from mvyield.pipeline.run import record_config

    paths = RunPaths.for_run(cfg.name, kind=kind).prepare()
    record_config(cfg, paths)
    handler = attach_log_file(paths)

    log.info("run directory: %s", paths.run_dir)
    try:
        yield paths
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def _report_done(outcome) -> int:
    """Print where the results landed."""
    print(f"\nDone. Results in {outcome.paths.run_dir}")
    print(f"  {outcome.paths.summary_file.name}")
    if outcome.figures:
        print(f"  {len(outcome.figures)} figure file(s) in {outcome.paths.plots_dir.name}/")
    return 0


def _geometries(args: argparse.Namespace) -> list[str] | None:
    """Parse a comma-separated geometry list, if one was given."""
    raw = getattr(args, "geometries", None)
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _load(args: argparse.Namespace) -> CaseConfig:
    """Load and validate the case config named on the command line."""
    cfg = load_case(Path(args.case), overrides=args.set)
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
    if getattr(args, "plots", None):
        cfg.plots.preset = args.plots
    return cfg


def _dry_run(cfg: CaseConfig, command: str) -> int:
    """Print the resolved configuration and the steps that would run."""
    print(f"[dry-run] {command}")
    print(_indent(cfg.describe()))

    steps = {
        "ingest": ["read dataset", "analyse steps", "export yield points"],
        "build-model": ["load yield points", "fit estimators", "write model bundle"],
        "calibrate": ["load yield points", "search hyper-parameters", "update bundle"],
        "run": ["load model", "build stress field", "evaluate methods", "draw plots"],
        "all": ["ingest (if stale)", "build model (if stale)", "run"],
    }
    print("\n  planned steps:")
    for i, step in enumerate(steps.get(command, []), start=1):
        print(f"    {i}. {step}")

    root = find_repo_root()
    _report_inputs(cfg, root)
    return 0


def _report_inputs(cfg: CaseConfig, root: Path) -> None:
    """Print each declared input path and whether it exists."""
    print("\n  inputs:")
    entries: list[tuple[str, Path | None]] = [
        ("dataset", cfg.dataset.source),
        ("field", cfg.field.path),
    ]
    for label, path in entries:
        if path is None:
            print(f"    {label:<8s} (not set)")
            continue
        resolved = resolve_input(path, root)
        status = "ok" if resolved.exists() else "MISSING"
        print(f"    {label:<8s} [{status}] {resolved}")


def _probe_import(module: str) -> str:
    """Report whether an optional dependency is importable."""
    try:
        __import__(module)
    except ImportError:
        return "not installed"
    return "available"


def _indent(text: str, prefix: str = "  ") -> str:
    """Indent every line of a block."""
    return "\n".join(prefix + line for line in text.splitlines())


def _setup_logging(verbose: bool) -> None:
    """Configure console logging.

    Verbosity is set on the console handler rather than on the root logger,
    so that the per-run file handler can still record everything: a handler
    only ever sees what the logger did not already discard.

    Only a console handler this function installed earlier is removed.
    Clearing the root handlers wholesale would work for a command line and
    break every other caller -- a test harness capturing records, a
    notebook, an application embedding this package.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_mvy_console", False):
            root.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    console._mvy_console = True  # type: ignore[attr-defined]

    root.addHandler(console)
    root.setLevel(logging.DEBUG)

    # The root logger is open to DEBUG so the run log can be complete;
    # without this, a run log is mostly Matplotlib font-cache chatter.
    for noisy in ("matplotlib", "PIL", "h5py", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


if __name__ == "__main__":
    raise SystemExit(main())
