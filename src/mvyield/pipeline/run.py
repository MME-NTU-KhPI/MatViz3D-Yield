"""The three modes, assembled from steps.

Each mode is a short sequence of calls, which is the point of having steps
that take and return artefacts: ``all`` is not a fourth code path but the
other two run back to back, sharing one run directory and one log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mvyield.mechanics.field import StressField
from mvyield.model.bundle import ModelBundle
from mvyield.paths import RunPaths
from mvyield.pipeline.evaluate import ResultBundle, evaluate_field
from mvyield.pipeline.figures import FigureContext, draw_figures
from mvyield.pipeline.ingest import dataset_key, resolve_dataset, run_ingest
from mvyield.pipeline.model import run_build_model
from mvyield.pipeline.report import write_summary
from mvyield.settings import CaseConfig, ConfigError

log = logging.getLogger(__name__)


@dataclass
class RunOutcome:
    """What one invocation produced, for the report and for tests."""

    paths: RunPaths
    ingest: object | None = None
    bundle: ModelBundle | None = None
    result: ResultBundle | None = None
    figures: list[Path] | None = None
    comparison: dict | None = None


def record_config(cfg: CaseConfig, paths: RunPaths) -> None:
    """Make sure the run directory says what produced it.

    A run directory is only useful later if it is self-describing, and
    ``mvy plot`` reads its configuration back rather than the case file,
    which may have been edited since. Writing this from the pipeline rather
    than from the command line means a run driven from Python leaves the
    same complete record as one driven from a terminal.
    """
    from mvyield.settings import dump_resolved

    if not paths.resolved_config.exists():
        dump_resolved(cfg, paths.resolved_config)


def build_case_field(cfg: CaseConfig, paths: RunPaths) -> StressField:
    """Build this case's stress field, opening the dataset only if asked.

    The analytical case can take the size of its plotting window from the
    dataset's coordinate box. That is the one thing in mode B that would
    otherwise reach for the dataset, so it is resolved here and only when
    ``field.domain_from`` actually calls for it.
    """
    from mvyield.cases import build_field

    extent = None
    if cfg.field.source == "analytic" and cfg.field.domain_from == "dataset":
        from mvyield.ingest import DatasetReader

        source = resolve_dataset(cfg, paths)
        with DatasetReader(source) as reader:
            extent = reader.coordinate_extent()
        log.info("plotting window from the dataset: extent %.4g", extent)

    return build_field(cfg.case, cfg.field, extent=extent)


def run_evaluate(
    cfg: CaseConfig,
    bundle: ModelBundle,
    paths: RunPaths,
    ingest=None,
    plots: str | None = None,
) -> RunOutcome:
    """Mode B: evaluate a field against an existing model, then report."""
    record_config(cfg, paths)
    field = build_case_field(cfg, paths)
    log.info("%s", field.summary())

    result = evaluate_field(field, bundle)
    for name, stats in result.summary().items():
        # eff_n goes beside coverage on purpose: the two answer different
        # questions, and 'covered 0 %' next to 'eff_n 173' is the pair that
        # says the data are plentiful and the query is simply outside the
        # radius range they cover.
        effective = result.outputs[name].diagnostics.get("eff_n")
        support = ""
        if effective is not None and np.isfinite(effective).any():
            positive = np.asarray(effective)[np.asarray(effective) > 0]
            if positive.size:
                support = f", median eff_n {np.median(positive):.0f}"

        log.info(
            "%-10s mean P(G) %.4f, max %.4f, covered %.0f %%%s",
            name,
            stats["p_mean"],
            stats["p_max"],
            100.0 * stats["n_covered"] / max(stats["n_points"], 1),
            support,
        )

    _export(cfg, result, paths)
    comparison = _compare(cfg, result, bundle, paths)

    figures = draw_figures(
        FigureContext(
            result=result,
            config=cfg,
            ingest=ingest,
            bundle=bundle,
            extra={"locus": comparison.get("locus"), "points": bundle.points},
        ),
        paths,
        preset=plots,
    )
    write_summary(cfg, paths, result=result, ingest=ingest, bundle=bundle,
                  figures=figures, comparison=comparison)

    return RunOutcome(paths=paths, ingest=ingest, bundle=bundle, result=result,
                      figures=figures, comparison=comparison)


def run_mode_a(
    cfg: CaseConfig,
    paths: RunPaths,
    force: bool = False,
    export: bool = True,
    plots: str | None = None,
    geometries: list[str] | None = None,
) -> RunOutcome:
    """Mode A: analyse the dataset, draw its figures, write the report."""
    record_config(cfg, paths)
    ingest = run_ingest(cfg, paths, force=force, geometries=geometries)
    log.info("%s", ingest.describe())

    if not export:
        log.info("--no-export: the yield-point artefact was not kept")

    figures = draw_figures(
        FigureContext(config=cfg, ingest=ingest),
        paths,
        preset=plots or cfg.dataset.plots_preset,
    )
    write_summary(cfg, paths, ingest=ingest, figures=figures)
    return RunOutcome(paths=paths, ingest=ingest, figures=figures)


def run_mode_c(
    cfg: CaseConfig,
    paths: RunPaths,
    force_ingest: bool = False,
    force_model: bool = False,
    calibrate: bool = True,
    plots: str | None = None,
) -> RunOutcome:
    """Mode C: dataset to figures in one pass, skipping unchanged steps."""
    record_config(cfg, paths)
    ingest = run_ingest(cfg, paths, force=force_ingest)
    log.info("%s", ingest.describe())

    key = dataset_key(resolve_dataset(cfg, paths), cfg.dataset)
    bundle = run_build_model(
        cfg, ingest.points, paths, key, force=force_model, calibrate=calibrate
    )

    return run_evaluate(cfg, bundle, paths, ingest=ingest, plots=plots)


def load_model_for(cfg: CaseConfig, paths: RunPaths) -> ModelBundle:
    """Load the model a mode-B run needs, or say how to build it.

    Refusing here is deliberate. The legacy case class carried on with only
    the analytic baseline when the model file was missing, so a run that
    silently answered a different question still produced figures that
    looked like the intended ones.
    """
    if cfg.model.path != "auto":
        return ModelBundle.load(Path(cfg.model.path))

    if cfg.dataset.source is None:
        raise ConfigError(
            "model.path is 'auto', which names the bundle after the dataset, "
            "but the case sets no dataset.source. Point model.path at an "
            "existing bundle, or add the dataset."
        )

    key = dataset_key(resolve_dataset(cfg, paths), cfg.dataset)
    return ModelBundle.load(cfg.model_path(paths.models_dir, key))


def _compare(cfg: CaseConfig, result: ResultBundle, bundle: ModelBundle,
             paths: RunPaths) -> dict:
    """Probe the yield locus and, with two or more methods, compare them.

    The locus is computed even for a single method: it is the shape of the
    surface the model fitted, and reading it against the reference values is
    how one finds out that a model is wrong in a way a probability field
    would not show.
    """
    from mvyield.pipeline.compare import compare_methods, save_report

    try:
        comparison = compare_methods(result, bundle, cfg)
    except Exception:
        log.exception("method comparison failed; the run's other results stand")
        return {}

    written = save_report(comparison, paths.data_dir / "method_comparison.json")
    log.info("wrote %s", written.name)

    ratios = comparison.get("locus", {}).get("anisotropy", {})
    for method in comparison.get("methods", []):
        if method in ratios and ratios[method] == ratios[method]:
            log.info("%-10s tension/shear ratio %.3f", method, ratios[method])
    return comparison


def _export(cfg: CaseConfig, result: ResultBundle, paths: RunPaths) -> None:
    """Write the numerical outputs the config asks for."""
    if cfg.export.results_npz:
        written = result.save(paths.data_dir / "results.npz")
        log.info("wrote %s", written.name)

    if cfg.export.vtk:
        from mvyield.viz.io import export_vtu

        try:
            target = export_vtu(
                result.field,
                {"P": result.probability()},
                paths.data_dir / "field.vtu",
            )
            log.info("wrote %s", Path(target).name)
        except ImportError:
            log.warning(
                "export.vtk is set but meshio is not installed; "
                "install it with: pip install -e \".[vtk]\""
            )
