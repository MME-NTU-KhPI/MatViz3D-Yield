"""Figures over the body: where it yields, and how sure the model is.

These are the only figures that depend on the problem's geometry, and they
depend on it only through :class:`~mvyield.mechanics.field.StressField` --
coordinates, a mask and a set of geometry hints. Nothing here knows what a
hole is; the outline it draws is whatever the case attached.

Shared colour limits
--------------------
Every probability map is drawn on a fixed 0 to 1 scale rather than on its
own range. An automatic range makes a field spanning 0.001 to 0.004 look
exactly like one spanning 0 to 1, which is the single easiest way to read a
safe component as a failing one.
"""

from __future__ import annotations

import numpy as np

from mvyield.viz.labels import label, method_label
from mvyield.viz.plots._common import empty_panel, suptitle
from mvyield.viz.registry import Space, plot
from mvyield.viz.render import render_scalar
from mvyield.viz.theme import CMAP_PROBABILITY, Theme

YIELD_ZONE_COLOURS = ("#DBEAFE", "#FFE4E6")
"""Elastic and yielding fills: the two states a decision threshold produces."""


def _field_and_values(context, method: str, quantity: str = "probability"):
    """The drawn field and one of its scalars, reduced view aware."""
    if context.view is not None:
        return context.view.field, context.view.scalars[f"{method}/{quantity}"]
    return context.result.field, getattr(context.result.outputs[method], quantity)


@plot(
    "kde_probability",
    space=Space.FIELD,
    groups=["minimal", "paper"],
    figsize=(7.2, 5.8),
    description="Yield probability over the body",
)
def kde_probability(context, ax, theme: Theme) -> None:
    """Yield probability over the body."""
    method = context.result.primary
    field, probability = _field_and_values(context, method)

    render_scalar(
        ax, field, probability,
        label=label("yield_prob"), cmap=CMAP_PROBABILITY,
        vmin=0.0, vmax=1.0, theme=theme,
    )
    theme.title(ax, f"{method_label(method)}")

    finite = np.isfinite(probability)
    if finite.any():
        ax.annotate(
            f"max P(G) {np.nanmax(probability):.3f}",
            xy=(0.02, 0.97), xycoords="axes fraction", va="top",
            color=theme.text_body, fontsize=theme.tick_size,
            bbox={"facecolor": theme.card, "edgecolor": theme.border, "pad": 3.0},
        )


@plot(
    "yield_zone",
    space=Space.FIELD,
    groups=["minimal", "paper"],
    figsize=(7.2, 5.8),
    description="The region above the decision threshold",
)
def yield_zone(context, ax, theme: Theme) -> None:
    """The region above the decision threshold.

    The engineering answer the probability field is built to give: not "how
    likely", but "does this component yield". Drawn as two states rather
    than a ramp, because a threshold has already been applied and shading it
    smoothly would suggest it had not.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    method = context.result.primary
    field, probability = _field_and_values(context, method)
    p_crit = context.config.compare.p_crit

    render_scalar(
        ax, field, np.where(np.isfinite(probability), probability, np.nan),
        cmap=ListedColormap(YIELD_ZONE_COLOURS),
        norm=BoundaryNorm([0.0, p_crit, 1.0], 2),
        theme=theme, colorbar=False,
    )

    finite = np.isfinite(probability)
    share = 100.0 * float(np.mean(probability[finite] > p_crit)) if finite.any() else 0.0

    theme.title(ax, f"Yield zone, P(G) > {p_crit:g}")
    ax.annotate(
        f"{share:.2f} % of the section",
        xy=(0.02, 0.97), xycoords="axes fraction", va="top",
        color=theme.text_body, fontsize=theme.tick_size,
        bbox={"facecolor": theme.card, "edgecolor": theme.border, "pad": 3.0},
    )

    handles = [
        _patch(YIELD_ZONE_COLOURS[0], theme, f"elastic, P ≤ {p_crit:g}"),
        _patch(YIELD_ZONE_COLOURS[1], theme, f"yielding, P > {p_crit:g}"),
    ]
    theme.legend(ax, handles=handles, loc="lower right")


@plot(
    "three_methods",
    space=Space.FIELD,
    groups=["paper"],
    needs=["kde_slice"],
    layout=(1, 3),
    figsize=(15.0, 4.8),
    description="Every method's probability field, on one colour scale",
)
def three_methods(context, axes, theme: Theme) -> None:
    """Every method's probability field, on one colour scale.

    Panels share limits so the comparison is between the fields rather than
    between three colourbars, which is the whole reason to put them side by
    side.
    """
    methods = list(context.methods)[:3]
    panels = np.atleast_1d(axes).ravel()

    for ax, method in zip(panels, methods):
        field, probability = _field_and_values(context, method)
        render_scalar(
            ax, field, probability,
            label=label("yield_prob") if method == methods[-1] else "",
            cmap=CMAP_PROBABILITY, vmin=0.0, vmax=1.0, theme=theme,
            colorbar=(method == methods[-1]),
        )
        theme.title(ax, method_label(method, short=True))
        ax.annotate(
            f"max {np.nanmax(probability):.3f}",
            xy=(0.02, 0.97), xycoords="axes fraction", va="top",
            color=theme.text_body, fontsize=theme.tick_size,
            bbox={"facecolor": theme.card, "edgecolor": theme.border, "pad": 2.0},
        )

    for ax in panels[len(methods):]:
        ax.set_axis_off()

    suptitle(axes, theme, "Yield probability by method", "one colour scale, 0 to 1")


@plot(
    "probability_profile",
    space=Space.PROBE,
    groups=["paper"],
    figsize=(7.6, 5.0),
    description="Yield probability along the configured probe paths",
)
def probability_profile(context, ax, theme: Theme) -> None:
    """Yield probability along the configured probe paths.

    A field map shows where the peak is; a profile shows how quickly it
    falls away, which is what decides whether a yielding region is a surface
    blemish or a through-section problem.
    """
    from mvyield.viz.sample import build_path, path_abscissa, sample_along

    field = context.result.field
    paths = context.config.probes.paths
    theme.style_axes(ax)

    if not paths:
        empty_panel(ax, theme, "no probes.paths are configured")
        return

    scale = field.hints.length_scale
    drawn = 0

    for index, spec in enumerate(paths):
        try:
            coordinates = build_path(spec, scale)
        except (KeyError, ValueError) as exc:
            ax.annotate(f"path {spec.get('name')!r}: {exc}", xy=(0.02, 0.02 + 0.05 * index),
                        xycoords="axes fraction", color=theme.rose,
                        fontsize=theme.tick_size - 1)
            continue

        abscissa = path_abscissa(coordinates, scale)
        for method in context.methods:
            values = sample_along(field, context.result.probability(method), coordinates)
            if not np.isfinite(values).any():
                continue
            ax.plot(
                abscissa, values, lw=2.0,
                color=theme.method_colour(method),
                ls=_LINESTYLES[index % len(_LINESTYLES)],
                label=f"{method_label(method, short=True)} · {spec.get('name', index)}",
            )
            drawn += 1

    if not drawn:
        empty_panel(ax, theme, "every probe path falls outside the field")
        return

    p_crit = context.config.compare.p_crit
    ax.axhline(p_crit, color=theme.text_muted, lw=1.0, ls=":")
    ax.annotate(f"P = {p_crit:g}", xy=(0.99, p_crit), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", color=theme.text_muted, fontsize=theme.tick_size)

    ax.set_ylim(-0.02, 1.02)
    theme.title(ax, "Probability along the probe paths")
    theme.axis_labels(
        ax,
        x=label("path_position") + (f", {field.hints.length_symbol}" if scale else ""),
        y=label("yield_prob"),
    )
    theme.legend(ax, loc="best")


_LINESTYLES = ("-", "--", "-.", ":")


def _patch(colour: str, theme: Theme, text: str):
    """A legend swatch."""
    from matplotlib.patches import Patch

    return Patch(facecolor=colour, edgecolor=theme.border, label=text)

