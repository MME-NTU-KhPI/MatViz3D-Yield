"""Estimator behaviour: independence, options, and refusal to guess."""

from __future__ import annotations

import numpy as np
import pytest

from mvyield.mechanics.voigt import MANDEL_XY_LAST
from mvyield.model.base import EstimatorContext, EstimatorOutput
from mvyield.model.basis import (
    DEVIATORIC_BASIS,
    subspace_basis,
    subspace_direction,
    to_subspace,
)
from mvyield.model.gaussian import IsotropicBaseline
from mvyield.model.kde_ray import RayMarginalEstimator
from mvyield.model.kde_slice import SliceEstimator
from mvyield.model.transform import SpaceTransform


@pytest.fixture(scope="module")
def shell():
    """A yield shell of radius ~275 MPa with mild anisotropy."""
    rng = np.random.default_rng(5)
    n = 800
    direction = rng.normal(size=(n, 5))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = 275.0 + 25.0 * direction[:, 0] + rng.normal(0.0, 8.0, n)
    return (direction * radius[:, None]) @ DEVIATORIC_BASIS.T


@pytest.fixture(scope="module")
def context(shell):
    transform = SpaceTransform(convention=MANDEL_XY_LAST, deviatoric=True)
    return EstimatorContext(sigma_model=transform.forward(shell), transform=transform)


def uniaxial(magnitude: float) -> np.ndarray:
    """A uniaxial tension state of the given magnitude, in model space."""
    raw = np.array([[magnitude, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return SpaceTransform(convention=MANDEL_XY_LAST, deviatoric=True).forward(raw)


# --- Basis -------------------------------------------------------------------


def test_deviatoric_basis_is_orthonormal():
    np.testing.assert_allclose(DEVIATORIC_BASIS.T @ DEVIATORIC_BASIS, np.eye(5), atol=1e-15)


def test_basis_columns_are_trace_free():
    """Projection removes pressure by itself, so no prior step is required."""
    np.testing.assert_allclose(DEVIATORIC_BASIS[:3, :].sum(axis=0), 0.0, atol=1e-15)


def test_projection_discards_pressure(shell):
    """Adding hydrostatic pressure must not move subspace coordinates."""
    pressurised = shell.copy()
    pressurised[:, :3] += 137.0
    np.testing.assert_allclose(
        to_subspace(pressurised, DEVIATORIC_BASIS),
        to_subspace(shell, DEVIATORIC_BASIS),
        atol=1e-9,
    )


def test_full_space_basis_is_the_identity():
    """A pressure-sensitive model keeps all six dimensions."""
    basis = subspace_basis(deviatoric=False, convention=MANDEL_XY_LAST)
    np.testing.assert_allclose(basis, np.eye(6))


def test_purely_hydrostatic_direction_is_rejected():
    """A direction the deviatoric model cannot represent must not pass silently."""
    with pytest.raises(ValueError, match="hydrostatic"):
        subspace_direction(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), DEVIATORIC_BASIS)


# --- Method independence -----------------------------------------------------


def test_each_estimator_fits_alone(context):
    """Neither method may require the other to be constructed."""
    assert SliceEstimator.fit(context).name == "kde_slice"
    assert RayMarginalEstimator.fit(context).name == "kde_ray"


def test_slice_works_in_a_pressure_sensitive_space(shell):
    """With the hydrostatic axis kept, the slice uses a full 6D basis.

    A pressure-sensitive material has genuine hydrostatic scatter in its
    yield points, which is what makes the 6D covariance invertible. The
    fixture adds it explicitly.
    """
    rng = np.random.default_rng(9)
    pressurised = shell + rng.normal(0.0, 40.0, (len(shell), 1)) * np.array(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    )

    transform = SpaceTransform(convention=MANDEL_XY_LAST, deviatoric=False)
    context = EstimatorContext(sigma_model=transform.forward(pressurised), transform=transform)

    estimator = SliceEstimator.fit(context, mode="polar")
    assert estimator.basis.shape == (6, 6)
    assert np.isfinite(estimator.evaluate(pressurised).probability).any()


def test_degenerate_cloud_is_reported_not_pseudo_inverted(shell):
    """A rank-deficient cloud must fail loudly in the slice method.

    Points with no hydrostatic scatter span only five dimensions. The ray
    method tolerates that -- it never inverts -- but the slice does invert,
    and a pseudo-inverse would silently ignore the degenerate direction
    instead of penalising it, corrupting the exponents. Better to stop.
    """
    transform = SpaceTransform(convention=MANDEL_XY_LAST, deviatoric=False)
    context = EstimatorContext(sigma_model=transform.forward(shell), transform=transform)

    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        SliceEstimator.fit(context, mode="polar")


# --- Monotonicity and physical sanity ---------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda c: RayMarginalEstimator.fit(c, localized=False),
        lambda c: RayMarginalEstimator.fit(c, localized=True, power=8.0),
        lambda c: SliceEstimator.fit(c, mode="polar", kappa=20.0, tau_floor=2.0),
    ],
)
def test_probability_increases_with_load(context, make):
    """More load must never mean less yield probability."""
    estimator = make(context)
    loads = np.array([150.0, 250.0, 300.0, 400.0, 600.0])
    probabilities = np.array(
        [float(estimator.evaluate(uniaxial(load)).probability[0]) for load in loads]
    )

    assert np.all(np.diff(probabilities) >= -1e-9)
    assert probabilities[-1] > 0.95


@pytest.mark.parametrize(
    "make",
    [
        lambda c: RayMarginalEstimator.fit(c, localized=True, power=8.0),
        lambda c: SliceEstimator.fit(c, mode="polar", kappa=20.0, tau_floor=2.0),
    ],
)
def test_load_well_below_the_shell_is_safe(context, make):
    """At 150 MPa against a 275 MPa shell, yielding must be improbable."""
    assert float(make(context).evaluate(uniaxial(150.0)).probability[0]) < 0.05


def test_global_projection_overestimates_at_low_load(context):
    """Quantify the pathology that motivates weighting.

    Marginalising over the hyperplane orthogonal to the ray projects a thin
    shell onto values concentrated near zero, so a query at 150 MPa is
    compared against a distribution centred far below the true radius and
    comes out alarmingly high. Both weighted methods put the same state
    below 5%. This is why the plain projection is kept only as a textbook
    reference and is not the default.
    """
    plain = RayMarginalEstimator.fit(context, localized=False)
    weighted = RayMarginalEstimator.fit(context, localized=True, power=8.0)

    safe_load = uniaxial(150.0)
    assert float(plain.evaluate(safe_load).probability[0]) > 0.5
    assert float(weighted.evaluate(safe_load).probability[0]) < 0.05


def test_probability_stays_in_the_unit_interval(context, shell):
    for estimator in (
        RayMarginalEstimator.fit(context),
        SliceEstimator.fit(context, mode="polar"),
        IsotropicBaseline.fit(context),
    ):
        probability = estimator.evaluate(shell).probability
        finite = probability[np.isfinite(probability)]
        assert finite.min() >= 0.0 and finite.max() <= 1.0


def test_null_stress_yields_zero_probability(context):
    """A stress-free point cannot be yielding."""
    zero = np.zeros((1, 6))
    assert (
        RayMarginalEstimator.fit(context, localized=False).evaluate(zero).probability[0] == 0.0
    )
    assert np.isnan(SliceEstimator.fit(context).evaluate(zero).probability[0])


def test_slice_width_never_exceeds_the_marginal(context):
    """Cauchy-Schwarz: marginalisation inflates uncertainty, never reduces it.

    This is the quantitative reason the slice exists, so it is worth
    checking rather than asserting in a comment.
    """
    ray = RayMarginalEstimator.fit(context, localized=False)
    slice_global = SliceEstimator.fit(context, mode="global")

    rng = np.random.default_rng(2)
    for _ in range(20):
        direction = rng.normal(size=6)
        direction[:3] -= direction[:3].mean()
        direction /= np.linalg.norm(direction)

        assert slice_global.marginal_std(direction) <= ray.marginal_std(direction) + 1e-9


def test_localized_ray_recovers_the_shell_radius(context):
    """The angle-weighted estimate must land on the shell, not inside it.

    The plain projection collapses toward the centre of the cloud in five
    dimensions; this is the failure the weighting exists to fix.
    """
    direction = np.array([2.0, -1.0, -1.0, 0.0, 0.0, 0.0])
    direction /= np.linalg.norm(direction)

    localized = RayMarginalEstimator.fit(context, localized=True, power=8.0)
    plain = RayMarginalEstimator.fit(context, localized=False)

    assert 240.0 < localized.yield_radius(direction) < 320.0
    assert plain.yield_radius(direction) < localized.yield_radius(direction)


def test_slice_truncation_removes_the_antipodal_branch(context):
    """Weight must not leak onto the opposite branch of the ray.

    Distance is measured to the line, not the ray, so without truncation a
    point at ``mu ~ -275`` would count as a neighbour.
    """
    direction = np.array([2.0, -1.0, -1.0, 0.0, 0.0, 0.0])
    direction /= np.linalg.norm(direction)

    estimator = SliceEstimator.fit(context, mode="polar", kappa=20.0, tau_floor=2.0)
    centres, _, log_weights = estimator.ray_geometry(direction)
    weights = np.exp(log_weights - log_weights.max())

    assert float(weights[centres < 0].sum()) == pytest.approx(0.0, abs=1e-12)


def test_effective_sample_size_falls_as_the_cone_narrows(context):
    """Higher concentration means a narrower cone and fewer contributing points."""
    direction = np.array([2.0, -1.0, -1.0, 0.0, 0.0, 0.0])
    direction /= np.linalg.norm(direction)

    wide = SliceEstimator.fit(context, mode="polar", kappa=5.0).effective_n(direction)
    narrow = SliceEstimator.fit(context, mode="polar", kappa=100.0).effective_n(direction)
    assert narrow < wide


def test_localized_ray_withholds_an_estimate_when_starved(context):
    """A confident number from three points is worse than an admitted gap."""
    estimator = RayMarginalEstimator.fit(context, localized=True, power=200.0, min_eff_n=50.0)
    result = estimator.evaluate(uniaxial(300.0))

    assert np.isnan(result.probability[0])
    assert not result.covered[0]


def test_yield_radius_refuses_rather_than_extrapolates(context):
    estimator = RayMarginalEstimator.fit(context, localized=True, power=400.0, min_eff_n=100.0)
    with pytest.raises(ValueError, match="too few dataset points"):
        estimator.yield_radius(np.array([1.0, -0.5, -0.5, 0.0, 0.0, 0.0]))


# --- Isotropic baseline ------------------------------------------------------


def test_baseline_estimates_from_the_cloud(context):
    """The default must read the material from the data, not from a config."""
    baseline = IsotropicBaseline.fit(context)

    assert baseline.source == "from_dataset"
    assert 250.0 < baseline.mean < 300.0
    assert baseline.n_points == context.n_points


def test_baseline_caption_states_its_provenance(context):
    """A reader must not mistake the baseline for a data-driven result."""
    text = IsotropicBaseline.fit(context).describe()
    assert "Isotropic baseline" in text
    assert "estimated from" in text

    manual = IsotropicBaseline.manual(275.0, 10.0).describe()
    assert "manual" in manual


def test_baseline_is_direction_blind(context):
    """Having no shape is the property under examination, so pin it down."""
    baseline = IsotropicBaseline.fit(context)
    rng = np.random.default_rng(0)

    values = []
    for _ in range(10):
        direction = rng.normal(size=6)
        direction[:3] -= direction[:3].mean()
        direction /= np.linalg.norm(direction)
        values.append(baseline.probability_at(direction, 280.0))

    assert np.ptp(values) == pytest.approx(0.0, abs=1e-12)


def test_baseline_rejects_a_zero_scatter_cloud():
    transform = SpaceTransform(convention=MANDEL_XY_LAST, deviatoric=True)
    sphere = np.tile([200.0, -100.0, -100.0, 0.0, 0.0, 0.0], (50, 1))
    context = EstimatorContext(sigma_model=transform.forward(sphere), transform=transform)

    with pytest.raises(ValueError, match="no scatter"):
        IsotropicBaseline.fit(context)


# --- Round trips and contracts ----------------------------------------------


@pytest.mark.parametrize(
    "make,loader",
    [
        (lambda c: RayMarginalEstimator.fit(c, power=12.0), RayMarginalEstimator.from_params),
        (
            lambda c: SliceEstimator.fit(c, mode="polar", kappa=33.0),
            SliceEstimator.from_params,
        ),
    ],
)
def test_parameters_survive_a_round_trip(context, shell, make, loader):
    original = make(context)
    restored = loader(context.sigma_model, original.to_params())

    np.testing.assert_allclose(
        restored.evaluate(shell).probability,
        original.evaluate(shell).probability,
        rtol=1e-12,
    )


def test_kernel_estimators_report_no_gradient(context):
    """A method without a usable normal must say so, not fake one."""
    assert RayMarginalEstimator.fit(context).gradient(context.sigma_model) is None
    assert SliceEstimator.fit(context).gradient(context.sigma_model) is None


def test_output_length_mismatch_is_caught():
    with pytest.raises(ValueError, match="disagree on length"):
        EstimatorOutput(
            probability=np.zeros(5), spread=np.zeros(5), covered=np.zeros(3, dtype=bool)
        )


def test_unknown_bandwidth_mode_is_rejected(context):
    with pytest.raises(ValueError, match="bandwidth mode"):
        SliceEstimator.fit(context, mode="magic")


def test_tiny_cloud_is_rejected_with_a_clear_message():
    transform = SpaceTransform(convention=MANDEL_XY_LAST, deviatoric=True)
    rng = np.random.default_rng(0)
    context = EstimatorContext(
        sigma_model=transform.forward(rng.normal(size=(4, 6)) * 100), transform=transform
    )
    with pytest.raises(ValueError, match="too small for the slice"):
        SliceEstimator.fit(context)


# --- Kernel geometry reported to the user ------------------------------------


def test_ray_half_weight_angle_matches_its_own_kernel(context):
    """The reported angle must be where the kernel actually halves.

    A config sets a dimensionless ``power``; the console and the report
    quote an angle derived from it. Deriving that angle anywhere other than
    from the kernel itself is how the two come to disagree.
    """
    estimator = RayMarginalEstimator.fit(context, power=8.0)
    half = estimator.half_weight_angle()

    assert estimator.angular_weight(half) == pytest.approx(0.5, abs=1e-12)
    assert half == pytest.approx(23.5, abs=0.1)


def test_slice_half_weight_angle_matches_its_own_kernel(context):
    estimator = SliceEstimator.fit(context, kappa=20.0)
    half = estimator.half_weight_angle()

    assert estimator.angular_weight(half) == pytest.approx(0.5, abs=1e-12)


def test_a_wider_kernel_reports_a_wider_angle(context):
    """Monotone in the parameter, or the number would mislead."""
    narrow = SliceEstimator.fit(context, kappa=100.0).half_weight_angle()
    wide = SliceEstimator.fit(context, kappa=5.0).half_weight_angle()

    assert narrow < wide

    sharp = RayMarginalEstimator.fit(context, power=32.0).half_weight_angle()
    blunt = RayMarginalEstimator.fit(context, power=2.0).half_weight_angle()

    assert sharp < blunt


def test_ray_borrows_from_a_wider_cone_than_the_slice(context):
    """The mechanism behind their different anisotropy, stated in degrees."""
    ray = RayMarginalEstimator.fit(context, power=8.0).half_weight_angle()
    slice_ = SliceEstimator.fit(context, kappa=44.72).half_weight_angle()

    assert ray > 2.0 * slice_


def test_angular_weight_is_one_at_zero_and_falls_off(context):
    for estimator in (
        RayMarginalEstimator.fit(context, power=8.0),
        SliceEstimator.fit(context, kappa=20.0),
    ):
        angles = np.array([0.0, 5.0, 30.0, 89.0])
        weights = estimator.angular_weight(angles)

        assert weights[0] == pytest.approx(1.0)
        assert np.all(np.diff(weights) < 0.0)


def test_descriptions_state_the_angle_not_just_the_parameter(context):
    """The parameter alone says nothing about how much sphere is averaged."""
    ray = RayMarginalEstimator.fit(context, power=8.0).describe()
    slice_ = SliceEstimator.fit(context, kappa=20.0).describe()

    assert "23.5 deg" in ray and "p = 8" in ray
    assert "deg" in slice_ and "kappa = 20" in slice_


def test_baseline_description_reports_the_coefficient_of_variation(context):
    """A sigma in MPa cannot be compared between materials; a CV can."""
    text = IsotropicBaseline.fit(context).describe()

    assert "CV" in text and "%" in text
