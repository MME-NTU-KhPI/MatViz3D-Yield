"""Redrawing a finished run without recomputing it.

The expensive part of a run is the evaluation, and it already wrote its
answers to ``data/results.npz``. Changing a colour map, a preset or an
output format should therefore cost seconds, not the minutes it took to
produce the numbers -- and it should be impossible for the redrawn figure to
disagree with the original, because it is the same array.

The configuration comes from the run's own ``config.resolved.yaml`` rather
than from the case file, which may have been edited since. A run directory
is a record of what happened, and redrawing it must not quietly adopt
today's settings.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from mvyield.mechanics.field import (
    FieldMeta,
    GeometryHints,
    Outline,
    StressField,
    Topology,
)
from mvyield.mechanics.voigt import get_convention
from mvyield.model.base import EstimatorOutput
from mvyield.paths import RunPaths, find_repo_root
from mvyield.pipeline.evaluate import ResultBundle
from mvyield.pipeline.figures import FigureContext, draw_figures
from mvyield.settings import ConfigError, build_case, load_yaml

log = logging.getLogger(__name__)


def replot(
    run_dir: Path,
    preset: str | None = None,
    formats: list[str] | None = None,
) -> list[Path]:
    """Redraw the figures of a finished run.

    Parameters
    ----------
    run_dir : Path
        A directory written by ``mvy run`` or ``mvy all``.
    preset : str, optional
        Overrides the preset the run used.
    formats : list of str, optional
        Overrides the output formats.

    Returns
    -------
    list of Path
        Files written.
    """
    run_dir = Path(run_dir)
    cfg = _load_resolved(run_dir)

    if formats:
        cfg.plots.formats = tuple(f.lower().lstrip(".") for f in formats)
        cfg.plots.__post_init__()

    paths = RunPaths(root=find_repo_root(), run_dir=run_dir)
    paths.plots_dir.mkdir(parents=True, exist_ok=True)

    result = load_results(run_dir / "data" / "results.npz", cfg)
    bundle = _load_bundle(cfg, paths)

    return draw_figures(
        FigureContext(
            result=result,
            config=cfg,
            bundle=bundle,
            extra=_stored_extras(run_dir, bundle),
        ),
        paths,
        preset=preset,
    )


def _load_bundle(cfg, paths):
    """Load the run's model, if it is still there.

    The diagnostics figures describe the estimator rather than the field --
    its angular kernel, its calibration curve, how the locus moves with the
    bandwidth -- so redrawing them needs the model. Loading it recomputes
    nothing, which is what ``mvy plot`` promises.

    A missing model is not an error. A run directory outlives the artefact
    store, and the figures that need one say so on the page.
    """
    from mvyield.pipeline.run import load_model_for

    try:
        return load_model_for(cfg, paths)
    except (FileNotFoundError, ConfigError, OSError) as exc:
        log.info("redrawing without a model: %s", exc)
        return None


def _stored_extras(run_dir: Path, bundle) -> dict:
    """Recover the tables the pipeline computed, rather than recomputing."""
    extras: dict = {"points": bundle.points if bundle is not None else None}

    comparison = run_dir / "data" / "method_comparison.json"
    if comparison.exists():
        import json

        extras["locus"] = json.loads(comparison.read_text(encoding="utf-8")).get("locus")
    return extras


def load_results(path: Path, cfg) -> ResultBundle:
    """Rebuild a :class:`ResultBundle` from what a run wrote.

    Only what the figures need is restored: the field, and each method's
    probability, spread, coverage and diagnostics. The model is loaded
    separately by :func:`_load_bundle`, and its absence is survivable.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no stored results in {path}\n"
            f"Redrawing needs the arrays a run wrote. Set export.results_npz "
            f"and run the case again."
        )

    with np.load(path) as data:
        keys = list(data.files)
        points = data["points"]
        grid_shape = _grid_shape(cfg, len(points))
        field = StressField(
            points=points,
            sigma6=data["sigma6"],
            topology=_topology(points, grid_shape),
            meta=FieldMeta(
                source=cfg.field.source,
                convention=get_convention(cfg.field.convention),
                grid_shape=grid_shape,
            ),
            valid=data["valid"] if "valid" in keys else None,
            hints=_hints(data, keys),
        )

        # Named by what a method must have written, not by the first path
        # segment: the archive also holds a 'hints/' namespace, and a
        # prefix scan would offer it as a method.
        outputs: dict[str, EstimatorOutput] = {}
        for method in sorted(
            key[: -len("/probability")] for key in keys if key.endswith("/probability")
        ):
            outputs[method] = EstimatorOutput(
                probability=data[f"{method}/probability"],
                spread=data[f"{method}/spread"],
                covered=data[f"{method}/covered"],
                diagnostics={
                    key.split("/", 2)[2]: data[key]
                    for key in keys
                    if key.startswith(f"{method}/diag/")
                },
            )

    if not outputs:
        raise ConfigError(f"{path.name} holds no method output to draw")

    primary = cfg.model.primary if cfg.model.primary in outputs else sorted(outputs)[0]
    return ResultBundle(field=field, outputs=outputs, primary=primary)


# --- Helpers -----------------------------------------------------------------


def _hints(data, keys: list[str]) -> GeometryHints:
    """Rebuild the presentation hints a run stored beside its arrays.

    A run written before these were stored simply has none, and the figures
    fall back to physical axes with no overlay -- which is what they do for
    a case that never supplied hints either.
    """
    hints = GeometryHints()
    if "hints/length_scale" in keys:
        hints.length_scale = float(data["hints/length_scale"][0])

    index = 0
    while f"hints/outline/{index}/points" in keys:
        radius_key = f"hints/outline/{index}/radius"
        hints.outlines.append(
            Outline(
                kind="circle" if radius_key in keys else "polyline",
                points=data[f"hints/outline/{index}/points"],
                radius=float(data[radius_key][0]) if radius_key in keys else None,
            )
        )
        index += 1
    return hints


def _load_resolved(run_dir: Path):
    """Read the configuration the run itself recorded."""
    resolved = run_dir / "config.resolved.yaml"
    if not resolved.exists():
        raise FileNotFoundError(
            f"{run_dir} has no config.resolved.yaml, so it is not a run "
            f"directory this version wrote."
        )
    return build_case(load_yaml(resolved), source_path=resolved)


def _grid_shape(cfg, n_points: int) -> tuple[int, int] | None:
    """Recover the grid shape of an analytical field."""
    if cfg.field.source != "analytic":
        return None
    side = int(round(np.sqrt(n_points)))
    return (side, side) if side * side == n_points else None


def _topology(points: np.ndarray, grid_shape: tuple[int, int] | None) -> Topology:
    """Recover the topology from the stored coordinates.

    A regular analytical grid is recognised by its square point count; for
    everything else the z spread decides, exactly as the ANSYS import does,
    so a 3D field is not silently redrawn as a plane.
    """
    if grid_shape is not None:
        return Topology.GRID2D
    if len(points) and float(np.ptp(points[:, 2])) > 1e-9:
        return Topology.CLOUD3D
    return Topology.CLOUD2D
