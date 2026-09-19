"""Figures of the yield surface itself: its shape, and the cloud behind it.

Both are drawn from the model and the point cloud, never from the field, so
they describe the material and hold for any case built on the same dataset.

The locus is not recomputed here. Probing a yield radius means a search
along a ray for every direction and every method, which the pipeline has
already paid for once to write the comparison report; the figure reads that
table. A plotting function that recomputes is a plotting function that can
disagree with the report beside it.
"""

from __future__ import annotations

import numpy as np

from mvyield.viz.labels import method_label
from mvyield.viz.plots._common import empty_panel, robust_limits, suptitle
from mvyield.viz.registry import Space, plot
from mvyield.viz.theme import Theme

PI_PLANE_BINS = 160
"""Resolution of the density histogram in the deviatoric plane."""


def _octahedral(sigma6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project onto the pi-plane from the normal components.

    Uses the frame components rather than sorted principal values: sorting a
    triple confines its projection to one sextant, filling the plane with a
    band that is an artefact of the sort rather than a property of the
    surface.
    """
    xx, yy, zz = sigma6[:, 0], sigma6[:, 1], sigma6[:, 2]
    return (xx - yy) / np.sqrt(2.0), (2.0 * zz - xx - yy) / np.sqrt(6.0)


def _cloud(context) -> np.ndarray | None:
    """The yield-point cloud, from whichever artefact this run carries."""
    if context.ingest is not None:
        return context.ingest.points.sigma6
    if context.extra.get("points") is not None:
        return context.extra["points"].sigma6
    return None


@plot(
    "pi_plane_density",
    space=Space.MATERIAL,
    groups=["paper"],
    figsize=(6.8, 6.0),
    description="Density of yield points in the deviatoric plane",
)
def pi_plane_density(context, ax, theme: Theme) -> None:
    """Density of yield points in the deviatoric plane.

    Where the surface is well sampled and where it is not. A ray drawn
    through a sparse sector is a ray whose yield radius the model is
    guessing, and this is the figure that says which sectors those are.
    """
    from matplotlib.colors import LogNorm

    cloud = _cloud(context)
    theme.style_axes(ax)

    if cloud is None or len(cloud) < 2:
        empty_panel(ax, theme, "no yield-point cloud in this run")
        return

    x, y = _octahedral(cloud)
    inside = _within(x, y)

    counts, x_edges, y_edges = np.histogram2d(
        x[inside], y[inside], bins=PI_PLANE_BINS
    )
    mesh = ax.pcolormesh(
        x_edges, y_edges, counts.T,
        cmap="viridis", norm=LogNorm(vmin=1.0, vmax=max(counts.max(), 2.0)),
        shading="auto", zorder=2,
    )

    ax.axhline(0.0, color=theme.border, lw=0.8, zorder=3)
    ax.axvline(0.0, color=theme.border, lw=0.8, zorder=3)
    ax.set_aspect("equal", adjustable="box")

    theme.colorbar(ax.get_figure(), mesh, ax, label="Yield points per cell")
    theme.title(ax, "Yield-point density, deviatoric plane")
    theme.axis_labels(
        ax,
        x=r"$(\sigma_{xx}-\sigma_{yy})/\sqrt{2}$, MPa",
        y=r"$(2\sigma_{zz}-\sigma_{xx}-\sigma_{yy})/\sqrt{6}$, MPa",
    )

    outside = int((~inside).sum())
    if outside:
        ax.annotate(
            f"{outside} off-scale",
            xy=(0.03, 0.03), xycoords="axes fraction",
            color=theme.text_muted, fontsize=theme.tick_size - 1,
        )


@plot(
    "yield_locus_polar",
    space=Space.MATERIAL,
    groups=["paper"],
    figsize=(8.6, 5.6),
    description="Yield radius by loading direction, per method",
)
def yield_locus_polar(context, ax, theme: Theme) -> None:
    """Yield radius by loading direction, per method.

    Drawn as dots on a suppressed axis rather than as bars. The radii span a
    few per cent -- roughly 215 to 230 MPa for this material -- so bars from
    zero would put every difference worth reading into the top five per cent
    of the panel. Bars promise a zero baseline and truncating one lies about
    the ratios; dots make no such promise, so the axis is free to show the
    range the data occupies.

    The tension-to-shear ratio above the panel is the number most sensitive
    to a surface smeared by angular mixing. In this deviatoric metric an
    isotropic von Mises surface sits at exactly one, so a value above it is
    genuine anisotropy and a method that averages over too wide a cone is
    pulled back towards one.
    """
    locus = (context.extra or {}).get("locus")
    theme.style_axes(ax)

    if not locus or not locus.get("rows"):
        empty_panel(ax, theme, "no locus was computed for this run")
        return

    rows = locus["rows"]
    methods = locus.get("methods", [])
    positions = np.arange(len(rows))
    values: list[np.ndarray] = []

    for position in positions:
        ax.axvline(position, color=theme.grid, lw=8.0, zorder=1)

    for method in methods:
        radii = np.array([row.get(method, np.nan) for row in rows], dtype=np.float64)
        values.append(radii)
        ax.plot(
            positions, radii,
            marker="o", markersize=8.0, lw=1.2, ls=":",
            color=theme.method_colour(method),
            markeredgecolor=theme.card, markeredgewidth=1.5,
            label=method_label(method, short=True), zorder=3,
        )

    references = np.array([row.get("reference", np.nan) for row in rows], dtype=np.float64)
    if np.isfinite(references).any():
        ax.scatter(
            positions[np.isfinite(references)], references[np.isfinite(references)],
            marker="D", s=60.0, facecolor="none",
            edgecolor=theme.text_heading, linewidths=1.8,
            label="reference", zorder=4,
        )
        values.append(references)

    everything = np.concatenate(values) if values else np.array([0.0, 1.0])
    ax.set_ylim(*robust_limits(everything, coverage=100.0))
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels([row["label"] for row in rows], rotation=30, ha="right")

    theme.title(ax, "Yield locus by direction")
    theme.axis_labels(ax, y="Yield radius (deviatoric norm), MPa")
    theme.legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    ratios = locus.get("anisotropy", {})
    expected = ratios.get("von_mises_expected", 1.0)
    shown = "   ".join(
        f"{method_label(name, short=True)} {ratios[name]:.3f}"
        for name in methods
        if np.isfinite(ratios.get(name, np.nan))
    )
    if shown:
        ax.annotate(
            f"tension / shear:  {shown}   (isotropic {expected:.3f})",
            xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 30),
            textcoords="offset points", ha="left",
            color=theme.text_muted, fontsize=theme.tick_size,
        )


def _within(x: np.ndarray, y: np.ndarray, coverage: float = 99.0) -> np.ndarray:
    """Mask keeping the central share of a two-dimensional cloud."""
    margin = 0.5 * (100.0 - coverage)
    limits = [np.percentile(values, [margin, 100.0 - margin]) for values in (x, y)]
    return (
        (x >= limits[0][0]) & (x <= limits[0][1])
        & (y >= limits[1][0]) & (y <= limits[1][1])
    )

