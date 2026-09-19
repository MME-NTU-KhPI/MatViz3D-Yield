"""Figures that describe the estimate rather than the answer.

Every figure here is about whether to believe the probability field: which
points the dataset actually supports, how many yield points formed each
estimate, how the hyper-parameter was chosen and how much the answer moves
when it changes.

They are not in the ``paper`` preset. A reader of a paper wants the result;
a reader deciding whether the result is sound wants these, and putting both
in one set means neither audience gets a figure list that fits on a page.

Recomputation
-------------
:func:`bandwidth_sensitivity` and :func:`uncertainty_band` do call the model
again, unlike every other figure in this package. There is no stored table
for them to disagree with -- the pipeline computes no sweep and no
per-path yield radius -- and both are bounded: a few dozen radius searches
each, in a preset that is not run for every result.
"""

from __future__ import annotations

import numpy as np

from mvyield.viz.labels import label, method_label
from mvyield.viz.plots._common import empty_panel, robust_limits, suptitle
from mvyield.viz.registry import Space, plot
from mvyield.viz.render import render_scalar
from mvyield.viz.theme import CMAP_COVERAGE, CMAP_SPREAD, Theme

SENSITIVITY_DIRECTIONS = ("x_uniaxial", "xy_shear", "biaxial_xy")
"""Directions swept by :func:`bandwidth_sensitivity`: two poles and a mix."""

SENSITIVITY_STEPS = 9
"""Candidate values per sweep. Each costs one yield-radius search."""


def _scalar(context, method: str, quantity: str):
    """A per-point quantity, from the reduced view when there is one.

    Returns ``None`` for the values when the method does not carry that
    quantity, so a figure can explain itself instead of raising.
    """
    if context.view is not None:
        values = context.view.scalars.get(f"{method}/{quantity}")
        field = context.view.field
    else:
        output = context.result.outputs[method]
        values = output.diagnostics.get(quantity, getattr(output, quantity, None))
        field = context.result.field

    if values is None:
        return field, None
    return field, np.asarray(values, dtype=np.float64)


def _modelled_methods(context) -> list[str]:
    """Methods this run both evaluated and kept parameters for.

    A figure that rebuilds an estimator must not assume the two lists
    agree: a result can be read back from ``results.npz`` beside a model
    that was rebuilt with fewer methods since.
    """
    if context.bundle is None:
        return []
    return [name for name in context.methods if name in context.bundle.methods]


def _estimator(context, method: str):
    """Rebuild one estimator from the bundle this run carries."""
    from mvyield.model.registry import load_estimator

    bundle = context.bundle
    cloud = bundle.transform.forward(bundle.points.sigma6)
    return load_estimator(method, cloud, bundle.params.get(method, {}))


# --- Over the body -----------------------------------------------------------


@plot(
    "coverage_map",
    space=Space.FIELD,
    groups=["diagnostics"],
    figsize=(7.2, 5.8),
    description="Where the dataset supports the query, and where it does not",
)
def coverage_map(context, ax, theme: Theme) -> None:
    """Where the dataset supports the query, and where it does not.

    An uncovered point still carries a probability, and that probability is
    an extrapolation. This is the figure that says which parts of a
    probability map are answers and which are guesses.
    """
    method = context.result.primary
    field, covered = _scalar(context, method, "covered")

    render_scalar(
        ax, field, covered.astype(float),
        cmap=CMAP_COVERAGE, vmin=0.0, vmax=1.0, theme=theme, colorbar=False,
    )

    valid = field.valid_mask()
    share = 100.0 * float(np.mean(covered[valid] > 0.5)) if valid.any() else 0.0

    theme.title(ax, f"Dataset coverage, {method_label(method, short=True)}")
    ax.annotate(
        f"{share:.1f} % covered",
        xy=(0.02, 0.97), xycoords="axes fraction", va="top",
        color=theme.text_body, fontsize=theme.tick_size,
        bbox={"facecolor": theme.card, "edgecolor": theme.border, "pad": 3.0},
    )

    from matplotlib.patches import Patch

    theme.legend(
        ax,
        handles=[
            Patch(facecolor=CMAP_COVERAGE(0.0), edgecolor=theme.border,
                  label="outside the data"),
            Patch(facecolor=CMAP_COVERAGE(1.0), edgecolor=theme.border,
                  label="supported"),
        ],
        loc="lower right",
    )


@plot(
    "eff_n_map",
    space=Space.FIELD,
    groups=["diagnostics"],
    needs_diagnostics=["eff_n"],
    figsize=(7.2, 5.8),
    description="Yield points forming the estimate at each query",
)
def eff_n_map(context, ax, theme: Theme) -> None:
    """Yield points forming the estimate at each query.

    ``n_eff = (sum w)^2 / sum w^2`` counts how many cloud points actually
    contribute along a direction, whatever the weights. It is the honest
    sample size behind a local estimate, and where it falls to single
    figures the estimate is noise however confident the probability looks.
    """
    method = next(
        (name for name in context.methods
         if "eff_n" in context.result.outputs[name].diagnostics),
        None,
    )
    if method is None:
        theme.style_axes(ax)
        empty_panel(ax, theme, "no method in this run reports an effective sample size")
        return

    field, effective = _scalar(context, method, "eff_n")
    if effective is None:
        theme.style_axes(ax)
        empty_panel(ax, theme, f"{method} carries no eff_n in this view")
        return

    render_scalar(
        ax, field, effective,
        label=label("eff_n"), cmap=CMAP_SPREAD, theme=theme,
        vmin=0.0, vmax=float(np.nanpercentile(effective[np.isfinite(effective)], 99))
        if np.isfinite(effective).any() else None,
    )

    threshold = float(context.config.model.kde_slice.get("min_eff_n", 15.0))
    finite = np.isfinite(effective)
    starved = 100.0 * float(np.mean(effective[finite] < threshold)) if finite.any() else 0.0

    theme.title(ax, f"Effective sample size, {method_label(method, short=True)}")
    ax.annotate(
        f"median {np.nanmedian(effective):.0f}\n"
        f"{starved:.1f} % below {threshold:g}",
        xy=(0.02, 0.97), xycoords="axes fraction", va="top",
        color=theme.text_body, fontsize=theme.tick_size,
        bbox={"facecolor": theme.card, "edgecolor": theme.border, "pad": 3.0},
    )


# --- Along a path ------------------------------------------------------------


@plot(
    "uncertainty_band",
    space=Space.PROBE,
    groups=["diagnostics"],
    figsize=(7.8, 5.2),
    description="Applied load against the yield surface, with its spread",
)
def uncertainty_band(context, ax, theme: Theme) -> None:
    """Applied load against the yield surface, with its spread.

    The probability profile answers "how likely"; this one shows the two
    quantities behind it in the same units. Where the applied radius enters
    the band, the margin is inside the model's own uncertainty -- which is a
    different and more useful statement than a probability near one half.
    """
    from mvyield.viz.sample import build_path, path_abscissa, sample_stress_along

    field = context.result.field
    paths = context.config.probes.paths
    theme.style_axes(ax)

    if not paths:
        empty_panel(ax, theme, "no probes.paths are configured")
        return
    if context.bundle is None:
        empty_panel(ax, theme, "no model bundle in this context")
        return

    spec = paths[0]
    scale = field.hints.length_scale
    coordinates = build_path(spec, scale)
    abscissa = path_abscissa(coordinates, scale)

    sigma = sample_stress_along(field, coordinates)
    applied = context.bundle.transform.forward(sigma, source=field.meta.convention)
    radius = np.linalg.norm(applied, axis=1)

    ax.plot(abscissa, radius, lw=2.4, color=theme.text_heading,
            label="applied deviatoric radius")

    for method in _modelled_methods(context):
        estimator = _estimator(context, method)
        centres, widths = _surface_along(estimator, applied)
        if not np.isfinite(centres).any():
            continue

        colour = theme.method_colour(method)
        ax.plot(abscissa, centres, lw=2.0, color=colour,
                label=f"{method_label(method, short=True)} yield radius")
        ax.fill_between(abscissa, centres - widths, centres + widths,
                        color=colour, alpha=0.18, lw=0)

    theme.title(ax, f"Margin along '{spec.get('name', 'path')}'")
    theme.axis_labels(
        ax,
        x=label("path_position") + (f", {field.hints.length_symbol}" if scale else ""),
        y="Deviatoric radius, MPa",
    )
    theme.legend(ax, loc="best")


def _surface_along(estimator, applied: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Yield radius and spread along a sequence of query states."""
    centres = np.full(len(applied), np.nan)
    widths = np.zeros(len(applied))

    output = estimator.evaluate(applied)
    for index, state in enumerate(applied):
        magnitude = float(np.linalg.norm(state))
        if magnitude <= 1e-9:
            continue
        try:
            centres[index] = estimator.yield_radius(state / magnitude)
        except Exception:  # noqa: BLE001 - a direction the method declines
            continue
    widths = np.nan_to_num(output.spread, nan=0.0)
    return centres, widths


# --- The estimator itself ----------------------------------------------------


@plot(
    "weight_profile",
    space=Space.AGGREGATE,
    groups=["diagnostics"],
    figsize=(7.4, 5.0),
    description="How each method weights a neighbour by angle",
)
def weight_profile(context, ax, theme: Theme) -> None:
    """How each method weights a neighbour by angle.

    Both data-driven methods answer along a direction by borrowing from
    yield points near it, and they differ mainly in how quickly that
    borrowing falls off. A wide kernel buys effective sample size and pays
    for it by mixing in directions where the surface is a different
    distance away -- which is what flattens an anisotropic surface.
    """
    theme.style_axes(ax)
    bundle = context.bundle
    if bundle is None:
        empty_panel(ax, theme, "no model bundle in this context")
        return

    angle = np.linspace(0.0, 90.0, 361)
    cosine = np.cos(np.deg2rad(angle))
    drawn = False

    for method in bundle.methods:
        estimator = _estimator(context, method)
        if not hasattr(estimator, "angular_weight"):
            continue

        # The kernel comes from the estimator, not from a copy of its
        # formula here: the console states the same half-weight angle, and
        # two statements of one kernel eventually disagree.
        weight = estimator.angular_weight(angle)
        params = bundle.params.get(method, {})
        text = (
            f"von Mises-Fisher, $\\kappa$ = {float(params.get('kappa', 0)):g}"
            if method == "kde_slice"
            else f"$\\cos^p$, $p$ = {float(params.get('power', 0)):g}"
        )

        ax.plot(angle, weight, lw=2.2, color=theme.method_colour(method),
                label=f"{method_label(method, short=True)}: {text}")
        half = estimator.half_weight_angle()
        # Staggered: two kernels can reach half weight a few degrees apart,
        # and one offset would stack the two labels on each other.
        ax.annotate(
            f"half weight at {half:.1f}°",
            xy=(half, 0.5), xytext=(16, 22 if drawn else -30),
            textcoords="offset points",
            color=theme.method_colour(method), fontsize=theme.tick_size,
            arrowprops={"arrowstyle": "-", "color": theme.border},
        )
        drawn = True

    if not drawn:
        empty_panel(ax, theme, "no kernel method in this model")
        return

    ax.axhline(0.5, color=theme.text_muted, lw=0.9, ls=":")
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(0.0, 1.05)
    theme.title(ax, "Angular weighting")
    theme.axis_labels(ax, x="Angle from the query direction, degrees",
                      y=label("weight"))
    theme.legend(ax, loc="upper right")


@plot(
    "calibration_curve",
    space=Space.AGGREGATE,
    groups=["diagnostics"],
    figsize=(7.4, 5.0),
    description="The objective that selected each hyper-parameter",
)
def calibration_curve(context, ax, theme: Theme) -> None:
    """The objective that selected each hyper-parameter.

    A flat optimum and a sharp one justify very different confidence in the
    chosen value, and the single number in the report cannot tell them
    apart. Where the curve is flat, the cheaper end of the plateau is the
    better choice: it keeps effective sample size.
    """
    theme.style_axes(ax)
    bundle = context.bundle
    if bundle is None:
        empty_panel(ax, theme, "no model bundle in this context")
        return

    curves = {
        name: record for name, record in bundle.hyperparams.items()
        if record.get("curve")
    }
    if not curves:
        manual = ", ".join(sorted(bundle.hyperparams)) or "none"
        empty_panel(
            ax, theme,
            f"no hyper-parameter was calibrated in this run\n({manual} set manually)",
        )
        return

    for index, (name, record) in enumerate(sorted(curves.items())):
        grid = np.asarray(record["curve"]["grid"], dtype=np.float64)
        objective = np.asarray(record["curve"]["objective"], dtype=np.float64)
        colour = (theme.blue, theme.rose, theme.green, theme.amber)[index % 4]

        ax.plot(grid, objective, marker="o", markersize=7.0, lw=2.0,
                color=colour, label=f"${name}$")
        ax.axvline(record["value"], color=colour, lw=1.2, ls="--")
        ax.annotate(
            f"{name} = {record['value']:.4g}",
            xy=(record["value"], 0.04 + 0.06 * index), xycoords=("data", "axes fraction"),
            xytext=(8, 0), textcoords="offset points",
            color=colour, fontsize=theme.tick_size, va="bottom",
        )

    if all(np.ptp(np.asarray(r["curve"]["grid"])) > 0 for r in curves.values()):
        ax.set_xscale("log")

    theme.title(ax, "Calibration objective")
    theme.axis_labels(ax, x="Candidate value", y=label("loglik"))
    theme.legend(ax, loc="best")


@plot(
    "bandwidth_sensitivity",
    space=Space.AGGREGATE,
    groups=["diagnostics"],
    needs=["kde_slice"],
    figsize=(7.6, 5.2),
    description="How the yield radius moves with the kernel width",
)
def bandwidth_sensitivity(context, ax, theme: Theme) -> None:
    """How the yield radius moves with the kernel width.

    The calibration curve says which value scored best; this says how much
    it mattered. A locus that barely moves across a decade of the parameter
    is a result that does not depend on the choice, and one that swings is a
    result that has to be reported with it.
    """
    from mvyield.pipeline.compare import STANDARD_DIRECTIONS

    theme.style_axes(ax)
    bundle = context.bundle
    if bundle is None or "kde_slice" not in bundle.methods:
        empty_panel(ax, theme, "the slice method did not run")
        return

    from mvyield.model.kde_slice import SliceEstimator

    params = dict(bundle.params.get("kde_slice", {}))
    chosen = float(params.get("kappa", 20.0))
    grid = np.geomspace(max(chosen / 8.0, 1.0), chosen * 8.0, SENSITIVITY_STEPS)
    cloud = bundle.transform.forward(bundle.points.sigma6)

    colours = (theme.blue, theme.rose, theme.green)
    for index, key in enumerate(SENSITIVITY_DIRECTIONS):
        direction = bundle.transform.forward(STANDARD_DIRECTIONS[key][None, :])[0]
        radii = []
        for kappa in grid:
            estimator = SliceEstimator.from_params(cloud, {**params, "kappa": kappa})
            try:
                radii.append(float(estimator.yield_radius(direction)))
            except Exception:  # noqa: BLE001
                radii.append(np.nan)

        ax.plot(grid, radii, marker="o", markersize=6.0, lw=2.0,
                color=colours[index % len(colours)],
                label=STANDARD_DIRECTIONS and key.replace("_", " "))

    ax.axvline(chosen, color=theme.text_muted, lw=1.4, ls="--")
    ax.annotate(f"chosen $\\kappa$ = {chosen:g}", xy=(chosen, 0.02),
                xycoords=("data", "axes fraction"), rotation=90,
                color=theme.text_muted, fontsize=theme.tick_size,
                ha="right", va="bottom")

    ax.set_xscale("log")
    theme.title(ax, "Sensitivity to the kernel concentration")
    theme.axis_labels(ax, x=label("kappa"), y="Yield radius, MPa")
    theme.legend(ax, loc="best")


# --- The cloud ---------------------------------------------------------------


@plot(
    "dataset_3d_scatter",
    space=Space.MATERIAL,
    groups=["diagnostics"],
    layout=(1, 3),
    figsize=(13.5, 4.6),
    description="The yield cloud in three orthogonal deviatoric planes",
)
def dataset_3d_scatter(context, axes, theme: Theme) -> None:
    """The yield cloud in three orthogonal deviatoric planes.

    Three flat projections rather than one perspective view. A perspective
    scatter of several thousand points hides density behind whichever face
    is nearest, and the depth cue it buys cannot be measured off the page;
    orthogonal panels can, and they tile without overlapping.
    """
    from mvyield.viz.plots.surface import _cloud

    cloud = _cloud(context)
    panels = np.atleast_1d(axes).ravel()
    for ax in panels:
        theme.style_axes(ax)

    if cloud is None or len(cloud) < 2:
        for ax in panels:
            empty_panel(ax, theme, "no yield-point cloud in this run")
        return

    names = (
        r"$\sqrt{2}\,\sigma_{yz}$",
        r"$\sqrt{2}\,\sigma_{xz}$",
        r"$\sqrt{2}\,\sigma_{xy}$",
    )
    pairs = ((3, 4), (3, 5), (4, 5))
    limits = [robust_limits(cloud[:, slot]) for slot in range(6)]

    for ax, (first, second) in zip(panels, pairs):
        ax.scatter(cloud[:, first], cloud[:, second], s=4.0, c=theme.blue,
                   alpha=0.22, linewidths=0.0)
        ax.axhline(0.0, color=theme.border, lw=0.8)
        ax.axvline(0.0, color=theme.border, lw=0.8)
        ax.set_xlim(*limits[first])
        ax.set_ylim(*limits[second])
        ax.set_aspect("equal", adjustable="box")
        theme.axis_labels(ax, x=f"{names[first - 3]}, MPa", y=f"{names[second - 3]}, MPa")

    suptitle(axes, theme, "Yield cloud, shear subspace",
             f"{len(cloud)} points; central 99 % of each component")
