"""Figures that describe the material, not the problem.

Everything here is drawn from the yield-point cloud and the per-step table.
None of it touches a geometry, a stress field or a probability, which is why
these are :attr:`~mvyield.viz.registry.Space.MATERIAL` and work unchanged for
any case built on the same dataset.

Colour
------
The cloud comes from forty RVE, and forty is far past the point where a
categorical palette carries identity: the legacy figures cycled a colormap
over geometries, producing forty hues no reader can tell apart or look up.
Geometry is not the variable of interest here anyway -- the shape and spread
of the cloud is -- so the cloud is drawn in a single hue with transparency,
where overlap reads as density. Categorical colour is reserved for the one
place it means something: the three principal axes, which are three fixed
series with a legend.
"""

from __future__ import annotations

import numpy as np

from mvyield.viz.labels import label
from mvyield.viz.plots._common import (
    empty_panel,
    note_clipped,
    robust_limits,
    suptitle,
)
from mvyield.viz.registry import Space, plot
from mvyield.viz.theme import Theme

COMPONENT_LABELS = (
    r"$\sigma_{xx}$",
    r"$\sigma_{yy}$",
    r"$\sigma_{zz}$",
    r"$\sqrt{2}\,\sigma_{yz}$",
    r"$\sqrt{2}\,\sigma_{xz}$",
    r"$\sqrt{2}\,\sigma_{xy}$",
)
"""Slot labels of the model convention, MANDEL_XY_LAST."""

STRAIN_LABELS = (
    r"$\varepsilon_{xx}$",
    r"$\varepsilon_{yy}$",
    r"$\varepsilon_{zz}$",
    r"$\gamma_{xy}$",
    r"$\gamma_{yz}$",
    r"$\gamma_{xz}$",
)
"""Slot labels of the dataset's own strain order, engineering shear."""

PRINCIPAL_LABELS = (r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$")

CLOUD_ALPHA = 0.25
"""Transparency at which overlap in a scatter reads as density."""

MAX_SCATTER = 6000
"""Points drawn in a scatter before thinning; beyond this it is a solid blob."""


# --- Small helpers -----------------------------------------------------------


def _cloud(context) -> np.ndarray:
    """Yield-point stresses in the model convention, MPa."""
    return context.ingest.points.sigma6


def _thin(values: np.ndarray, seed: int = 0) -> np.ndarray:
    """Subsample a large cloud so a scatter still shows structure."""
    if len(values) <= MAX_SCATTER:
        return values
    rng = np.random.default_rng(seed)
    return values[rng.choice(len(values), MAX_SCATTER, replace=False)]


def _density(
    ax,
    values: np.ndarray,
    theme: Theme,
    colour: str | None = None,
    limits: tuple[float, float] | None = None,
) -> None:
    """Draw a filled kernel density estimate of one quantity.

    When ``limits`` are given the estimate is evaluated over them, so the
    curve fills the panel instead of collapsing into a spike beside a tail
    that carries no shape.
    """
    from scipy.stats import gaussian_kde

    colour = colour or theme.blue
    finite = values[np.isfinite(values)]

    if finite.size < 2 or np.ptp(finite) <= 0.0:
        ax.axvline(float(finite[0]) if finite.size else 0.0, color=colour, lw=2.0)
        return

    low, high = limits if limits is not None else (finite.min(), finite.max())
    grid = np.linspace(low, high, 256)
    estimate = gaussian_kde(finite)(grid)

    ax.fill_between(grid, estimate, color=colour, alpha=0.30, lw=0)
    ax.plot(grid, estimate, color=colour, lw=2.0)
    ax.set_xlim(low, high)
    ax.set_ylim(bottom=0.0)

    if limits is not None:
        note_clipped(ax, finite, limits, theme)


def _panel_grid(axes, theme: Theme):
    """Style every panel of a grid and return it flattened."""
    flat = np.atleast_1d(axes).ravel()
    for ax in flat:
        theme.style_axes(ax)
    return flat


def _pair_matrix(axes, values: np.ndarray, names, theme: Theme, colour: str) -> None:
    """Lower-triangle scatter matrix with densities on the diagonal.

    Only the lower triangle is drawn: the upper one shows the same pairs
    with the axes swapped, so filling it doubles the ink and the reader's
    work without adding a fact.
    """
    n = len(names)
    thinned = _thin(values)
    limits = [robust_limits(values[:, slot]) for slot in range(n)]

    for row in range(n):
        for column in range(n):
            ax = axes[row, column]
            theme.style_axes(ax)

            if column > row:
                ax.set_axis_off()
                continue

            if row == column:
                _density(ax, values[:, row], theme, colour, limits=limits[row])
                ax.set_yticks([])
            else:
                ax.scatter(
                    thinned[:, column],
                    thinned[:, row],
                    s=3.0,
                    c=colour,
                    alpha=CLOUD_ALPHA,
                    linewidths=0.0,
                )
                ax.set_xlim(*limits[column])
                ax.set_ylim(*limits[row])

            if row == n - 1:
                theme.axis_labels(ax, x=names[column])
            else:
                ax.set_xticklabels([])
            if column == 0 and row > 0:
                theme.axis_labels(ax, y=names[row])
            elif column != 0:
                ax.set_yticklabels([])


def _octahedral(principal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project principal stresses onto the pi-plane.

    The deviatoric plane is where a pressure-insensitive yield surface lives:
    the hydrostatic axis projects to a point, so what is left is the shape of
    the surface rather than the load level.
    """
    s1, s2, s3 = principal[:, 0], principal[:, 1], principal[:, 2]
    return (s1 - s2) / np.sqrt(2.0), (2.0 * s3 - s1 - s2) / np.sqrt(6.0)


# --- The cloud ---------------------------------------------------------------


@plot(
    "yield_components",
    space=Space.MATERIAL,
    groups=["ingest_minimal", "ingest_diagnostics"],
    layout=(2, 3),
    description="Distribution of each yield-stress component",
)
def yield_components(context, axes, theme: Theme) -> None:
    """Distribution of each yield-stress component."""
    cloud = _cloud(context)
    panels = _panel_grid(axes, theme)

    for slot, ax in enumerate(panels):
        _density(ax, cloud[:, slot], theme, limits=robust_limits(cloud[:, slot]))
        theme.axis_labels(ax, x=f"{COMPONENT_LABELS[slot]}, MPa")
        ax.set_yticks([])
        median = float(np.median(cloud[:, slot]))
        ax.axvline(median, color=theme.rose, lw=1.2, ls="--")
        ax.annotate(
            f"median {median:.0f}",
            xy=(0.97, 0.90),
            xycoords="axes fraction",
            ha="right",
            color=theme.text_muted,
            fontsize=theme.tick_size,
        )

    suptitle(axes, theme, "Yield-point components",
              f"{len(cloud)} points from {context.ingest.points.n_groups} RVE; "
              f"central 99 % shown")


@plot(
    "stress_pairs_2d",
    space=Space.MATERIAL,
    groups=["ingest_diagnostics"],
    layout=(6, 6),
    figsize=(11.0, 10.0),
    description="Pairwise structure of the yield-stress components",
)
def stress_pairs_2d(context, axes, theme: Theme) -> None:
    """Pairwise structure of the yield-stress components."""
    _pair_matrix(axes, _cloud(context), COMPONENT_LABELS, theme, theme.blue)
    suptitle(axes, theme, "Yield stress, pairwise",
              "MPa; lower triangle only; central 99 % of each component")


@plot(
    "strain_pairs_2d",
    space=Space.MATERIAL,
    groups=["ingest_diagnostics"],
    layout=(6, 6),
    figsize=(11.0, 10.0),
    description="Pairwise structure of the applied strain at yield",
)
def strain_pairs_2d(context, axes, theme: Theme) -> None:
    """Pairwise structure of the applied strain at yield.

    Shows which loading directions the dataset actually sampled, which is
    what a gap in the yield surface has to be read against.
    """
    _pair_matrix(axes, context.ingest.steps.strain6, STRAIN_LABELS, theme, theme.green)
    suptitle(axes, theme, "Applied strain at yield, pairwise",
              "lower triangle only; central 99 % of each component")


@plot(
    "projected_kde",
    space=Space.MATERIAL,
    groups=["ingest_diagnostics"],
    layout=(2, 3),
    description="Cloud density projected onto each component axis",
)
def projected_kde(context, axes, theme: Theme) -> None:
    """Cloud density projected onto each component axis.

    The one-dimensional view the ray-marginal estimator works from: what the
    density looks like along a single axis once every other component has
    been integrated out.
    """
    cloud = _cloud(context)
    panels = _panel_grid(axes, theme)

    for slot, ax in enumerate(panels):
        limits = robust_limits(cloud[:, slot])
        _density(ax, cloud[:, slot], theme, limits=limits)
        theme.axis_labels(ax, x=f"{COMPONENT_LABELS[slot]}, MPa")
        ax.set_yticks([])
        spread = float(np.subtract(*np.percentile(cloud[:, slot], [75, 25])))
        ax.annotate(
            f"IQR {abs(spread):.0f} MPa",
            xy=(0.97, 0.90),
            xycoords="axes fraction",
            ha="right",
            color=theme.text_muted,
            fontsize=theme.tick_size,
        )

    suptitle(axes, theme, "Projected density",
              "marginal along each axis; central 99 % shown")


# --- Principal spaces --------------------------------------------------------


@plot(
    "stress_space",
    space=Space.MATERIAL,
    groups=["ingest_minimal", "ingest_diagnostics"],
    layout=(1, 2),
    figsize=(11.0, 4.8),
    description="Yield points in principal stress space",
)
def stress_space(context, axes, theme: Theme) -> None:
    """Yield points in principal stress space."""
    principal = context.ingest.steps.principal_stress
    left, right = _panel_grid(axes, theme)

    x, y = _octahedral(principal)
    thinned_x, thinned_y = _thin(np.column_stack([x, y])).T
    left.scatter(thinned_x, thinned_y, s=4.0, c=theme.blue, alpha=CLOUD_ALPHA, linewidths=0.0)
    left.axhline(0.0, color=theme.border, lw=0.8)
    left.axvline(0.0, color=theme.border, lw=0.8)
    left.set_aspect("equal", adjustable="datalim")
    theme.title(left, "Deviatoric (pi) plane")
    theme.axis_labels(left, x=r"$(\sigma_1-\sigma_2)/\sqrt{2}$, MPa",
                      y=r"$(2\sigma_3-\sigma_1-\sigma_2)/\sqrt{6}$, MPa")

    hydrostatic = principal.mean(axis=1)
    equivalent = context.ingest.steps.von_mises
    sample = _thin(np.column_stack([hydrostatic, equivalent]))
    right.scatter(sample[:, 0], sample[:, 1], s=4.0, c=theme.blue,
                  alpha=CLOUD_ALPHA, linewidths=0.0)
    right.axhline(float(np.mean(equivalent)), color=theme.rose, lw=1.4, ls="--")
    right.annotate(
        f"mean {np.mean(equivalent):.1f} MPa",
        xy=(0.02, 0.93), xycoords="axes fraction",
        color=theme.rose, fontsize=theme.tick_size,
    )
    limits = robust_limits(hydrostatic)
    right.set_xlim(*limits)
    note_clipped(right, hydrostatic, limits, theme)
    theme.title(right, "Pressure against equivalent stress")
    theme.axis_labels(right, x=label("hydrostatic"), y=label("vm_stress"))

    suptitle(axes, theme, "Principal stress space",
              "a pressure-insensitive criterion leaves the right panel flat")


@plot(
    "strain_space",
    space=Space.MATERIAL,
    groups=["ingest_diagnostics"],
    layout=(1, 2),
    figsize=(11.0, 4.8),
    description="Applied strain in principal strain space",
)
def strain_space(context, axes, theme: Theme) -> None:
    """Applied strain in deviatoric strain space.

    Drawn from the normal components in the frame the dataset uses, not from
    sorted principal values. Sorting a triple so that
    ``e_1 >= e_2 >= e_3`` confines its octahedral projection to a single
    sextant, which fills a pi-plane with a narrow band that looks like
    anisotropy and is an artefact of the sort. The stress figure escapes it
    because those principal values are tracked along the load path rather
    than sorted; the applied strain has no such history, so it is shown in
    frame components instead.
    """
    strain = context.ingest.steps.strain6
    left, right = _panel_grid(axes, theme)

    x, y = _octahedral(strain[:, :3])
    sample = _thin(np.column_stack([x, y]))
    left.scatter(sample[:, 0], sample[:, 1], s=4.0, c=theme.green,
                 alpha=CLOUD_ALPHA, linewidths=0.0)
    left.axhline(0.0, color=theme.border, lw=0.8)
    left.axvline(0.0, color=theme.border, lw=0.8)
    left.set_xlim(*robust_limits(x))
    left.set_ylim(*robust_limits(y))
    theme.title(left, "Deviatoric (pi) plane")
    theme.axis_labels(left, x=r"$(\varepsilon_{xx}-\varepsilon_{yy})/\sqrt{2}$",
                      y=r"$(2\varepsilon_{zz}-\varepsilon_{xx}-\varepsilon_{yy})/\sqrt{6}$")

    volumetric = strain[:, :3].sum(axis=1)
    magnitude = np.linalg.norm(strain, axis=1)
    sample = _thin(np.column_stack([volumetric, magnitude]))
    right.scatter(sample[:, 0], sample[:, 1], s=4.0, c=theme.green,
                  alpha=CLOUD_ALPHA, linewidths=0.0)
    limits = robust_limits(volumetric)
    right.set_xlim(*limits)
    right.set_ylim(*robust_limits(magnitude))
    note_clipped(right, volumetric, limits, theme)
    theme.title(right, "Volumetric against total strain")
    theme.axis_labels(right, x=r"$\varepsilon_{xx}+\varepsilon_{yy}+\varepsilon_{zz}$",
                      y=r"$\|\varepsilon\|$")

    suptitle(axes, theme, "Applied strain at yield",
              "frame components, not sorted principal values")


# --- Along the load path -----------------------------------------------------


@plot(
    "eigen_tracking",
    space=Space.MATERIAL,
    groups=["ingest_diagnostics"],
    layout=(1, 2),
    figsize=(11.0, 4.8),
    description="Principal stresses along the load path of one RVE",
)
def eigen_tracking(context, axes, theme: Theme) -> None:
    """Principal stresses along the load path of one RVE.

    Drawn for a single geometry because the load paths of different RVE are
    unrelated sequences; overlaying forty of them would suggest a trend
    across a variable that has no order.

    The three axes are matched to their predecessors step by step rather
    than sorted by magnitude, so a crossing shows as a crossing instead of
    the discontinuity that sorting produces.
    """
    steps = context.ingest.steps
    left, right = _panel_grid(axes, theme)

    first = steps.geometry_index == steps.geometry_index.min()
    order = np.argsort(steps.step[first])
    path = steps.step[first][order]
    principal = steps.principal_stress[first][order]
    colours = (theme.blue, theme.rose, theme.green)

    for axis in range(3):
        left.plot(path, principal[:, axis], lw=2.0, color=colours[axis],
                  label=PRINCIPAL_LABELS[axis])
        left.annotate(
            PRINCIPAL_LABELS[axis],
            xy=(path[-1], principal[-1, axis]),
            xytext=(4, 0), textcoords="offset points",
            color=colours[axis], fontsize=theme.tick_size, va="center",
        )

    limits = robust_limits(principal.ravel(), coverage=98.0)
    left.set_ylim(*limits)
    note_clipped(left, principal.ravel(), limits, theme)
    theme.title(left, f"RVE {steps.geometry_id[first][0]}")
    theme.axis_labels(left, x="Load step", y=label("stress_mpa"))
    theme.legend(left, loc="best")

    ordered = np.argsort(steps.step[first])
    right.plot(steps.step[first][ordered], steps.von_mises[first][ordered],
               lw=2.0, color=theme.blue)
    right.axhline(float(np.mean(steps.von_mises)), color=theme.rose, lw=1.2, ls="--")
    right.annotate(
        "cloud mean", xy=(0.02, 0.92), xycoords="axes fraction",
        color=theme.rose, fontsize=theme.tick_size,
    )
    theme.title(right, "Equivalent stress at yield")
    theme.axis_labels(right, x="Load step", y=label("vm_stress"))

    suptitle(axes, theme, "Principal stresses step by step",
              "independent load directions, not a continuous path; "
              "axes tracked between steps rather than sorted")


# --- Distributions -----------------------------------------------------------


@plot(
    "histograms",
    space=Space.MATERIAL,
    groups=["ingest_diagnostics"],
    layout=(1, 3),
    figsize=(13.0, 4.2),
    description="Yield strength, scaling factor and cloud radius",
)
def histograms(context, axes, theme: Theme) -> None:
    """Yield strength, scaling factor and cloud radius."""
    steps = context.ingest.steps
    equivalent, scaling, ax = steps.von_mises, steps.k_factor, _panel_grid(axes, theme)

    _density(ax[0], equivalent, theme)
    mean, std = float(np.mean(equivalent)), float(np.std(equivalent))
    ax[0].axvline(mean, color=theme.rose, lw=1.4, ls="--")
    theme.title(ax[0], "Yield strength")
    theme.axis_labels(ax[0], x=label("vm_stress"))
    ax[0].annotate(
        f"{mean:.1f} ± {std:.1f} MPa\nCV {100.0 * std / mean:.2f} %",
        xy=(0.97, 0.92), xycoords="axes fraction", ha="right", va="top",
        color=theme.text_body, fontsize=theme.tick_size,
    )

    _density(ax[1], scaling, theme, theme.amber, limits=robust_limits(scaling))
    theme.title(ax[1], "Scaling to yield")
    theme.axis_labels(ax[1], x=r"$k = \tau_{\mathrm{crss}} / \tau_{\max}$")

    radius = np.linalg.norm(_cloud(context), axis=1)
    _density(ax[2], radius, theme, theme.green, limits=robust_limits(radius))
    theme.title(ax[2], "Cloud radius")
    theme.axis_labels(ax[2], x=label("yield_radius"))
    ax[2].annotate(
        f"deviatoric {np.median(equivalent):.0f} MPa,\nfull norm reaches {radius.max():.0f}",
        xy=(0.97, 0.92), xycoords="axes fraction", ha="right", va="top",
        color=theme.text_muted, fontsize=theme.tick_size - 1,
    )

    for panel in ax:
        panel.set_yticks([])

    suptitle(axes, theme, "Distributions over the cloud",
              "central 99 % shown where a hydrostatic tail would flatten the panel")


@plot(
    "schmid_factors",
    space=Space.MATERIAL,
    groups=["ingest_diagnostics"],
    layout=(1, 2),
    figsize=(11.0, 4.4),
    description="Resolved shear and how many slip systems activate",
)
def schmid_factors(context, axes, theme: Theme) -> None:
    """Resolved shear and how many slip systems activate.

    The Schmid factor is the resolved shear a unit load produces, so it is
    recovered here as ``tau_max / |sigma|`` before scaling: a high factor
    means a grain oriented favourably for slip, and the spread across the
    cloud is what makes yielding orientation-dependent at all.
    """
    steps = context.ingest.steps
    left, right = _panel_grid(axes, theme)

    radius = np.linalg.norm(_cloud(context), axis=1)
    factor = np.divide(steps.tau_max * steps.k_factor, radius,
                       out=np.zeros_like(radius), where=radius > 0.0)

    _density(left, factor, theme, theme.amber)
    left.set_yticks([])
    theme.title(left, "Schmid factor at yield")
    theme.axis_labels(left, x=r"$\tau_{\max} / \|\sigma\|$")

    counts = steps.active_systems
    values, occurrences = np.unique(counts, return_counts=True)
    right.bar(values, 100.0 * occurrences / counts.size, width=0.55,
              color=theme.blue, zorder=3)
    for value, share in zip(values, 100.0 * occurrences / counts.size):
        right.annotate(f"{share:.1f} %", xy=(value, share), xytext=(0, 3),
                       textcoords="offset points", ha="center",
                       color=theme.text_body, fontsize=theme.tick_size)
    right.set_xticks(values)
    right.set_ylim(0.0, 108.0)
    theme.title(right, "Slip systems active at yield")
    theme.axis_labels(right, x="Systems within 0.5 % of the maximum", y="Share of points, %")

    suptitle(axes, theme, "Schmid criterion",
              "systems counted once per pair: reversing the slip direction is not a second system")
