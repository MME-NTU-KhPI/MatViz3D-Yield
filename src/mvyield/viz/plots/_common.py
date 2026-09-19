"""Helpers shared by the figure modules.

Small, but each exists because getting it wrong is invisible: a title that
overlaps a panel label on one figure size and not another, an axis set by
three outliers, a note placed where a series label already is.
"""

from __future__ import annotations

import numpy as np

from mvyield.viz.theme import Theme


def robust_limits(values: np.ndarray, coverage: float = 99.0) -> tuple[float, float]:
    """Axis limits that show the distribution rather than its extremes.

    Scaling a load step to yield multiplies its hydrostatic part too, and a
    nearly hydrostatic direction is scaled enormously: the cloud reaches
    ``|sigma|`` of several thousand MPa while its deviatoric radius stays
    near 270. Plotted raw, every component panel becomes a spike at zero
    with an empty decade beside it. The criterion ignores pressure, so these
    are not outliers to remove -- only ones to keep off the axis.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, 1.0)

    margin = 0.5 * (100.0 - coverage)
    low, high = np.percentile(finite, [margin, 100.0 - margin])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return (low - 1.0, high + 1.0)

    pad = 0.05 * (high - low)
    return (float(low - pad), float(high + pad))


def note_clipped(ax, values: np.ndarray, limits: tuple[float, float], theme: Theme) -> None:
    """Say how many points the axis limits leave outside.

    Placed bottom left: direct series labels sit at the right edge, where
    the last data point is.
    """
    values = np.asarray(values, dtype=np.float64)
    outside = int(np.count_nonzero((values < limits[0]) | (values > limits[1])))
    if outside:
        ax.annotate(
            f"{outside} off-scale",
            xy=(0.03, 0.06), xycoords="axes fraction", ha="left",
            color=theme.text_muted, fontsize=theme.tick_size - 1,
        )


def suptitle(axes, theme: Theme, text: str, subtitle: str = "") -> None:
    """Title a multi-panel figure, with an optional line of context.

    Positions are derived from the figure's height in inches rather than
    fixed fractions: a fraction that separates two lines on a tall figure
    overlaps them on a short one, and these figures range from 4 to 10
    inches tall. The reserved band also has to clear the panels' own titles,
    which is why it is wider than the two text lines need.
    """
    figure = np.atleast_1d(axes).ravel()[0].figure
    height = figure.get_size_inches()[1]

    def at(inches_from_top: float) -> float:
        return 1.0 - inches_from_top / height

    figure.tight_layout(rect=(0.0, 0.0, 1.0, at(0.78 if subtitle else 0.50)))
    figure.suptitle(
        text,
        y=at(0.24),
        va="center",
        color=theme.text_heading,
        fontweight="bold",
        fontsize=theme.title_size + 1,
    )
    if subtitle:
        figure.text(
            0.5, at(0.52), subtitle, ha="center", va="center",
            color=theme.text_muted, fontsize=theme.tick_size,
        )


def empty_panel(ax, theme: Theme, reason: str) -> None:
    """Leave a panel that explains why it is empty.

    A figure that silently comes out blank is indistinguishable from one
    whose data happened to be zero.
    """
    ax.annotate(reason, xy=(0.5, 0.5), xycoords="axes fraction", ha="center",
                color=theme.text_muted, fontsize=theme.label_size)
    ax.set_xticks([])
    ax.set_yticks([])
