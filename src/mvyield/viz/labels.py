"""Every string that appears on a figure, in one place.

Figures are English because they go into papers. That is easy to state and
hard to keep: in the pre-refactor code the method name for the ray estimator
appeared as a literal in eight plotting functions, half of them in
Ukrainian, so renaming it meant finding all eight.

Direction names matter more than they look. ``SLICE_REFERENCE`` and
``STANDARD_DIRECTIONS`` used Cyrillic strings as **dictionary keys**, which
put the language inside the program logic rather than at its surface: a
translation would have silently broken every lookup. Here the keys are
ASCII identifiers and only the displayed text is a label.

Anything a reader sees on a figure belongs here. Console messages and
``summary.md`` follow ``report.lang`` and stay wherever they are written.
"""

from __future__ import annotations

# --- Quantities --------------------------------------------------------------

LABELS: dict[str, str] = {
    # Probability and its companions
    "yield_prob": r"Yield probability $P(G)$",
    "yield_prob_pct": r"Yield probability $P(G)$, %",
    "yield_prob_diff": r"$\Delta P(G)$",
    "coverage": "Dataset coverage",
    "eff_n": r"Effective sample size $n_{\mathrm{eff}}$",
    "spread": r"Estimator spread, MPa",
    "sigma_d": r"Marginal width $\sigma_d$, MPa",
    "tau": r"Slice width $\tau$, MPa",
    "weight": "Angular weight",
    "density": r"Density $p(r \mid \hat{u})$",
    # Stress
    "vm_stress": r"von Mises stress $\sigma_{\mathrm{vm}}$, MPa",
    "vm_ratio": r"$f = \sigma_{\mathrm{vm}} / \sigma_y$",
    "stress_mpa": r"Stress, MPa",
    "yield_radius": r"Yield radius $r$, MPa",
    "applied_radius": r"Applied deviatoric magnitude $r^{*}$, MPa",
    "hydrostatic": r"Hydrostatic stress $\sigma_m$, MPa",
    "principal_1": r"$\sigma_1$, MPa",
    "principal_2": r"$\sigma_2$, MPa",
    "principal_3": r"$\sigma_3$, MPa",
    # Geometry
    "x_coord": r"$x$",
    "y_coord": r"$y$",
    "z_coord": r"$z$",
    "radial": r"Radial position $r$",
    "angle_deg": r"Angle $\theta$, degrees",
    "path_position": "Position along path",
    # Model diagnostics
    "power": r"Angular weight exponent $p$",
    "kappa": r"Concentration $\kappa$",
    "bandwidth": r"Bandwidth scale",
    "loglik": "Held-out log-likelihood",
    "objective": "Calibration objective",
}


def label(key: str, unit: str | None = None) -> str:
    """Look up a label, optionally appending a unit.

    Parameters
    ----------
    key : str
        Entry in :data:`LABELS`. Unknown keys are returned unchanged, so a
        one-off label does not require an entry, though a repeated one
        should get one.
    unit : str, optional
        Appended as ``", unit"`` when the label does not already carry one.
    """
    text = LABELS.get(key, key)
    if unit and "," not in text:
        text = f"{text}, {unit}"
    return text


def axis_label(
    key: str, length_scale: float | None = None, symbol: str = "a", unit: str = "mm"
) -> str:
    """Build a coordinate axis label, normalised or physical.

    Normalising by a reference length is a convention of the Kirsch problem,
    not a property of all problems. When a case supplies no reference the
    axis falls back to physical units, and nothing else has to change.

    Parameters
    ----------
    key : str
        ``"x_coord"``, ``"y_coord"`` or ``"z_coord"``.
    length_scale : float, optional
        Reference length. ``None`` selects physical units.
    symbol : str
        Symbol for the reference length, giving ``x/a``.
    unit : str
        Physical unit used when there is no reference length.
    """
    base = LABELS.get(key, key).strip("$")
    if length_scale is None:
        return f"${base}$, {unit}"
    return f"${base}/{symbol}$"


# --- Methods -----------------------------------------------------------------

METHOD_LABELS: dict[str, str] = {
    "kde_ray": "Ray marginal (6D KDE)",
    "kde_ray_global": "Ray marginal, global projection",
    "kde_ray_local": "Ray marginal, angle-weighted",
    "kde_slice": "Conditional slice (6D KDE)",
    "analytic": r"Isotropic baseline $\mathcal{N}(\mu,\sigma)$",
    "pinn": "Neural yield surface",
}

METHOD_SHORT: dict[str, str] = {
    "kde_ray": "Ray",
    "kde_slice": "Slice",
    "analytic": "Isotropic",
    "pinn": "PINN",
}


def method_label(method: str, localized: bool | None = None, short: bool = False) -> str:
    """Display name for a method.

    Parameters
    ----------
    method : str
        Method identifier.
    localized : bool, optional
        Distinguishes the two ray variants. They are one method in two
        modes, so they share an identifier but should be distinguishable on
        a figure that shows both.
    short : bool
        Return the compact form, for legends and table headers.
    """
    if short:
        return METHOD_SHORT.get(method, method)
    if method == "kde_ray" and localized is not None:
        return METHOD_LABELS["kde_ray_local" if localized else "kde_ray_global"]
    return METHOD_LABELS.get(method, method)


# --- Loading directions ------------------------------------------------------

DIRECTION_LABELS: dict[str, str] = {
    "x_uniaxial": "X uniaxial",
    "y_uniaxial": "Y uniaxial",
    "z_uniaxial": "Z uniaxial",
    "xy_shear": "XY shear",
    "yz_shear": "YZ shear",
    "xz_shear": "XZ shear",
    "biaxial_xy": "XY equibiaxial",
    "pure_shear": "Pure shear",
}


def direction_label(key: str) -> str:
    """Display name for a loading direction identifier."""
    return DIRECTION_LABELS.get(key, key)


# --- Provenance --------------------------------------------------------------

SOURCE_LABELS: dict[str, str] = {
    "manual": "chosen manually",
    "cv_loglik": "cross-validated",
    "locus_mae": "fitted to reference locus",
    "frozen": "reused from bundle",
    "from_dataset": "estimated from dataset",
}


def provenance_note(name: str, value: float, source: str) -> str:
    """Caption fragment stating where a parameter value came from.

    A reader cannot tell a cross-validated value from one picked by eye off
    a sweep table, and the distinction changes how much weight the figure
    carries. Stating it costs one line.
    """
    return f"{name} = {value:g} ({SOURCE_LABELS.get(source, source)})"
