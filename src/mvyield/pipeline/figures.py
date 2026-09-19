"""Drawing whatever the registry offers for a finished result.

This module contains no figures. It answers "which of the registered ones
apply here?", reduces the field when they need it, gives each a set of axes
and writes the result. A plotting function therefore never decides whether
it should run, never opens a file and never learns the topology it is being
handed -- which is what makes adding one a matter of writing a drawing
function and declaring its conditions.

Everything a figure could want arrives in a :class:`FigureContext`, so the
signature stays ``(context, ax, theme)`` whether the figure draws the
material, an aggregate, a field or a probe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any

from mvyield.paths import RunPaths
from mvyield.pipeline.evaluate import ResultBundle
from mvyield.settings import CaseConfig

log = logging.getLogger(__name__)


@dataclass
class FigureContext:
    """Everything a figure may draw from.

    Parameters
    ----------
    result : ResultBundle
        The evaluation, or ``None`` for a dataset-only run where no field
        was assessed.
    config : CaseConfig
        The whole case, so a figure can read probe paths or the comparison
        threshold without them being threaded through the call.
    view : ReducedField, optional
        The 2D view being drawn, for figures declared ``per_reduction``.
        ``None`` for the rest.
    ingest : IngestResult, optional
        Dataset analysis, when one was run. Carries the yield cloud, the
        per-step table and the material.
    bundle : ModelBundle, optional
        The fitted model, for figures that describe the surface rather than
        a field.
    extra : dict
        Anything a later package needs to pass along without changing this
        signature again.
    """

    result: ResultBundle | None = None
    config: CaseConfig | None = None
    view: Any = None
    ingest: Any = None
    bundle: Any = None
    extra: dict = dataclass_field(default_factory=dict)

    @property
    def methods(self) -> tuple[str, ...]:
        """Methods that produced output, primary first."""
        return self.result.methods if self.result is not None else ()

    @property
    def field(self):
        """The field being drawn: the reduced view when there is one."""
        if self.view is not None:
            return self.view.field
        return self.result.field if self.result is not None else None


def draw_figures(
    context: FigureContext,
    paths: RunPaths,
    preset: str | None = None,
) -> list[Path]:
    """Draw every applicable registered figure and write it out.

    Parameters
    ----------
    context : FigureContext
        What the figures draw from.
    paths : RunPaths
        Prepared run layout.
    preset : str, optional
        Overrides ``plots.preset`` from the config.

    Returns
    -------
    list of Path
        Files written.
    """
    # Importing the viz package registers the figures and pulls in
    # Matplotlib. A run that draws nothing should not pay for either.
    from mvyield.viz import REGISTRY, resolve_preset, select
    from mvyield.viz.io import save_figure
    from mvyield.viz.theme import THEME, apply_rcparams

    config = context.config
    plots = config.plots
    presets = config.raw.get("plot_presets") or load_presets()

    if not REGISTRY:
        log.warning(
            "no figures are registered, so none were drawn. The plotting "
            "packages have not landed yet; everything else in this run is "
            "complete."
        )
        return []

    names = resolve_preset(
        preset or plots.preset, presets, include=plots.include, exclude=plots.exclude
    )
    views = _views(context)
    topology = context.field.topology if context.field is not None else None

    specs, skipped = select(names, context.methods, topology, _diagnostics(context))
    if not specs:
        log.warning("preset %r selected no drawable figure (%s)", plots.preset, skipped)
        return []

    apply_rcparams(THEME)
    written: list[Path] = []

    for spec in specs:
        targets = views if spec.per_reduction else [None]
        for view in targets:
            written.extend(
                _draw_one(spec, context, view, paths, plots, save_figure, THEME)
            )

    log.info("drew %d figure file(s)", len(written))
    return written


# --- Internals ---------------------------------------------------------------


def _draw_one(spec, context, view, paths, plots, save_figure, theme) -> list[Path]:
    """Draw one figure, letting a failure cost only that figure.

    A run that produced numbers should not be lost because one plotting
    function raised on an edge case; the exception is logged with its figure
    named, and the rest of the run continues.
    """
    import matplotlib.pyplot as plt

    local = FigureContext(
        result=context.result,
        config=context.config,
        view=view,
        ingest=context.ingest,
        bundle=context.bundle,
        extra=context.extra,
    )
    suffix = getattr(view, "name", "") or ""

    figure, axes = _canvas(spec, plt)
    try:
        spec.function(local, axes, theme)
    except Exception:
        plt.close(figure)
        log.exception("[skip] figure %r failed", spec.name)
        return []

    return save_figure(figure, spec.name, paths, plots, suffix=suffix)


def load_presets(root: Path | None = None) -> dict[str, list[str]]:
    """Read the figure sets from ``configs/defaults.yaml``.

    ``load_case`` deliberately keeps this table out of the case object: a
    preset is a name resolved by the registry, not a setting the rest of the
    program reads. That leaves the table to be loaded here, where it is
    used, and a case may still override it by declaring ``plot_presets`` of
    its own.
    """
    from mvyield.paths import find_repo_root
    from mvyield.settings import load_yaml

    defaults = (root or find_repo_root()) / "configs" / "defaults.yaml"
    if not defaults.exists():
        log.warning("no %s, so only the 'all' preset is available", defaults)
        return {}
    return load_yaml(defaults).get("plot_presets") or {}


def _canvas(spec, plt):
    """Create the axes a figure declared it needs.

    A figure that wants a panel per stress component says so with
    ``layout=(2, 3)`` and is handed the grid. The alternative -- handing
    every figure one axes and letting the multi-panel ones remove it and
    build their own -- puts layout arithmetic inside drawing code, which is
    where it stops being reviewable.
    """
    if spec.layout is None:
        return plt.subplots(figsize=spec.figsize or (7.0, 5.5))

    rows, columns = spec.layout
    default = (3.1 * columns + 1.0, 2.7 * rows + 0.8)
    return plt.subplots(rows, columns, figsize=spec.figsize or default)


def _views(context: FigureContext) -> list:
    """Reduce a 3D field to the 2D views the config asks for."""
    if context.result is None:
        return [None]

    from mvyield.viz.reduce import apply_reductions

    field = context.result.field
    if not field.topology.is_3d:
        return apply_reductions(field, _scalars(context.result), ())

    return apply_reductions(
        field, _scalars(context.result), context.config.view.reductions
    )


def _scalars(result: ResultBundle) -> dict:
    """Per-point quantities carried through a reduction, keyed by method."""
    scalars: dict = {}
    for name in result.methods:
        output = result.outputs[name]
        scalars[f"{name}/probability"] = output.probability
        scalars[f"{name}/spread"] = output.spread
        scalars[f"{name}/covered"] = output.covered.astype(float)
        for key, values in output.diagnostics.items():
            scalars[f"{name}/{key}"] = values
    return scalars


def _diagnostics(context: FigureContext) -> tuple[str, ...]:
    """Diagnostic field names available, for the registry's conditions."""
    if context.result is None:
        return ()
    names: set[str] = set()
    for output in context.result.outputs.values():
        names.update(output.diagnostics)
    return tuple(sorted(names))
