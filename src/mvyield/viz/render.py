"""Drawing a scalar field, whatever shape its points happen to be in.

Every difference between a regular grid, a scattered plane and a reduced 3D
cloud is handled here. A plotting function calls :func:`render_scalar` and
never learns which case it got, which is what lets the figures written for
the plate serve the wire without modification.

The backend follows the field's topology, not the name of the problem:

``grid2d``
    ``imshow`` on the reshaped array. Fastest, and exact.
``cloud2d``
    Filled triangulation, with triangles inside an excluded region masked
    out so a hole does not get spanned.
``cloud3d`` / ``mesh3d``
    Not drawn directly. These must be reduced first
    (:mod:`mvyield.viz.reduce`); reaching here with one is a routing error
    and says so.

Triangulation can fail on degenerate point sets -- collinear points, or a
region so thin the triangles are slivers. The legacy code fell back to a
scatter plot with a printed warning, and that behaviour is kept, because a
readable scatter is better than a missing figure. The fallback is logged, so
it does not pass unnoticed.
"""

from __future__ import annotations

import logging

import numpy as np
from matplotlib.tri import Triangulation

from mvyield.mechanics.field import GeometryHints, Outline, StressField, Topology

from .labels import axis_label
from .theme import THEME, Theme

log = logging.getLogger(__name__)


def render_scalar(
    ax,
    field: StressField,
    values: np.ndarray,
    label: str = "",
    cmap="viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    levels: int = 24,
    theme: Theme = THEME,
    fig=None,
    colorbar: bool = True,
    **kwargs,
):
    """Draw a scalar field over the body.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    field : StressField
        Field supplying coordinates, mask and geometry hints.
    values : ndarray of shape (M,)
        Scalar values, one per field point. ``NaN`` is left blank.
    label : str
        Colourbar label, from :mod:`mvyield.viz.labels`.
    cmap : str or Colormap
        Colour mapping.
    vmin, vmax : float, optional
        Colour limits. Fixing these matters when comparing methods: an
        automatic range per figure makes two panels look alike when they are
        not.
    levels : int
        Contour levels for the triangulated path.
    theme : Theme
        Visual theme.
    fig : matplotlib.figure.Figure, optional
        Needed to attach a colourbar; taken from the axes when omitted.
    colorbar : bool
        Whether to draw a colourbar.

    Returns
    -------
    mappable
        The artist, for callers that want to add contour lines or a marker.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) != field.n_points:
        raise ValueError(
            f"values has {len(values)} entries for a field of {field.n_points} points"
        )

    if field.topology.is_3d:
        raise ValueError(
            f"cannot render a {field.topology.value} field directly. Reduce it "
            f"to 2D first via view.reductions, or export it with export.vtk and "
            f"inspect it in ParaView."
        )

    theme.style_axes(ax, fig)
    fig = fig or ax.get_figure()

    if field.topology is Topology.GRID2D:
        mappable = _render_grid(ax, field, values, cmap, vmin, vmax, **kwargs)
    else:
        mappable = _render_cloud(ax, field, values, cmap, vmin, vmax, levels, **kwargs)

    _draw_outlines(ax, field.hints, theme)
    _label_axes(ax, field.hints, theme)

    if colorbar and mappable is not None:
        theme.colorbar(fig, mappable, ax, label=label)
    return mappable


def _render_grid(ax, field, values, cmap, vmin, vmax, **kwargs):
    """Draw on a regular grid with ``imshow``."""
    rows, columns = field.meta.grid_shape
    masked = np.where(field.valid_mask(), values, np.nan).reshape(rows, columns)

    x, y = field.x, field.y
    extent = (x.min(), x.max(), y.min(), y.max())

    image = ax.imshow(
        np.ma.masked_invalid(masked),
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
        zorder=2,
        **kwargs,
    )
    return image


def _render_cloud(ax, field, values, cmap, vmin, vmax, levels, **kwargs):
    """Draw scattered planar points, with a scatter fallback."""
    usable = field.valid_mask() & np.isfinite(values)
    if not usable.any():
        log.warning("nothing to render: every value is masked or non-finite")
        ax.set_aspect("equal", adjustable="box")
        return None

    x, y, v = field.x[usable], field.y[usable], values[usable]

    try:
        triangulation = _triangulate(x, y, field.hints)
        contour = ax.tricontourf(
            triangulation,
            v,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            zorder=2,
            **kwargs,
        )
        ax.set_aspect("equal", adjustable="box")
        return contour
    except (ValueError, RuntimeError) as exc:
        log.warning("triangulation failed (%s); falling back to a scatter plot", exc)
        scatter = ax.scatter(
            x, y, c=v, s=6, cmap=cmap, vmin=vmin, vmax=vmax, zorder=2, **kwargs
        )
        ax.set_aspect("equal", adjustable="box")
        return scatter


def _triangulate(x: np.ndarray, y: np.ndarray, hints: GeometryHints) -> Triangulation:
    """Triangulate planar points, masking triangles inside excluded regions.

    Without masking, the triangulation spans a hole and the plot shows a
    coloured disc where there is no material.
    """
    triangulation = Triangulation(x, y)

    for outline in hints.outlines:
        if outline.kind != "circle" or not outline.radius:
            continue
        centre = np.asarray(outline.points, dtype=np.float64).reshape(-1, 2)[0]
        centres_x = x[triangulation.triangles].mean(axis=1)
        centres_y = y[triangulation.triangles].mean(axis=1)
        inside = np.hypot(centres_x - centre[0], centres_y - centre[1]) < outline.radius

        existing = triangulation.mask
        triangulation.set_mask(inside if existing is None else (existing | inside))

    return triangulation


def _draw_outlines(ax, hints: GeometryHints, theme: Theme) -> None:
    """Draw the geometry outlines a case supplied."""
    import matplotlib.pyplot as plt

    for outline in hints.outlines:
        if outline.kind == "circle" and outline.radius:
            centre = np.asarray(outline.points, dtype=np.float64).reshape(-1, 2)[0]
            ax.add_patch(
                plt.Circle(
                    tuple(centre),
                    outline.radius,
                    color=theme.text_muted,
                    fill=False,
                    linestyle="--",
                    linewidth=1.5,
                    zorder=5,
                )
            )
        elif outline.kind == "polyline":
            points = np.asarray(outline.points, dtype=np.float64).reshape(-1, 2)
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=theme.text_muted,
                linestyle="--",
                linewidth=1.5,
                zorder=5,
            )


def _label_axes(ax, hints: GeometryHints, theme: Theme) -> None:
    """Label the coordinate axes, normalised when a reference length exists."""
    theme.axis_labels(
        ax,
        x=axis_label("x_coord", hints.length_scale, hints.length_symbol, hints.length_unit),
        y=axis_label("y_coord", hints.length_scale, hints.length_symbol, hints.length_unit),
    )
    ax.set_aspect("equal", adjustable="box")


def normalise_coordinates(field: StressField) -> np.ndarray:
    """Return coordinates divided by the reference length, when there is one.

    Figures should call this rather than dividing by a hole radius
    themselves, so a case without a reference length simply plots in
    physical units instead of raising.
    """
    if field.hints.length_scale:
        return field.points / field.hints.length_scale
    return field.points


def circle_outline(radius: float, centre=(0.0, 0.0)) -> Outline:
    """Build a circular outline, the common case for a hole or a shaft."""
    return Outline(
        kind="circle",
        points=np.asarray([centre], dtype=np.float64),
        radius=float(radius),
    )
