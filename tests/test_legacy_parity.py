"""Parity with the pre-refactor estimators.

Every method is run against :mod:`legacy_reference`, a verbatim copy of the
original code, on synthetic clouds and a real Kirsch field. Agreement to
machine precision is what makes "the maths survived the move" a checked
statement rather than a claim.

If a test here fails, the port is wrong. The reference is not to be adjusted.
"""

from __future__ import annotations

import numpy as np
import pytest

import legacy_reference as legacy
from mvyield.mechanics.voigt import MANDEL_XY_LAST, from_components
from mvyield.model.base import EstimatorContext
from mvyield.model.kde_ray import RayMarginalEstimator
from mvyield.model.kde_slice import SliceEstimator
from mvyield.model.transform import SpaceTransform

TOL = dict(rtol=1e-10, atol=1e-12)


@pytest.fixture(scope="module")
def cloud():
    """A yield shell with anisotropy, hydrostatic scatter and shear."""
    rng = np.random.default_rng(20260811)
    n = 1200

    direction = rng.normal(size=(n, 5))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = 275.0 + 30.0 * direction[:, 0] + rng.normal(0.0, 9.0, n)
    y = direction * radius[:, None]

    sigma = y @ legacy.DEV_BASIS.T
    sigma[:, :3] += rng.normal(0.0, 60.0, (n, 1))
    return sigma


@pytest.fixture(scope="module")
def field():
    """A Kirsch stress field, 150 MPa remote tension, hole radius 1."""
    applied, a = 150.0, 1.0
    r = np.linspace(1.0, 4.0, 60)
    theta = np.linspace(0.0, 2.0 * np.pi, 48)
    grid_r, grid_t = np.meshgrid(r, theta, indexing="ij")

    ar2, ar4 = (a / grid_r) ** 2, (a / grid_r) ** 4
    s_rr = applied / 2 * (1 - ar2) + applied / 2 * (1 - 4 * ar2 + 3 * ar4) * np.cos(2 * grid_t)
    s_tt = applied / 2 * (1 + ar2) - applied / 2 * (1 + 3 * ar4) * np.cos(2 * grid_t)
    s_rt = -applied / 2 * (1 + 2 * ar2 - 3 * ar4) * np.sin(2 * grid_t)

    c, s = np.cos(grid_t), np.sin(grid_t)
    sxx = s_rr * c**2 - 2 * s_rt * s * c + s_tt * s**2
    syy = s_rr * s**2 + 2 * s_rt * s * c + s_tt * c**2
    sxy = (s_rr - s_tt) * s * c + s_rt * (c**2 - s**2)

    zeros = np.zeros(sxx.size)
    return from_components(
        sxx.ravel(),
        syy.ravel(),
        zeros,
        sxy.ravel(),
        zeros,
        zeros,
        convention=MANDEL_XY_LAST,
    )


@pytest.fixture(scope="module")
def legacy_bundle(cloud):
    """The legacy bundle, including the 5D upgrade used by the slice."""
    return legacy.upgrade_bundle(legacy.compute_kde_bundle(cloud))


@pytest.fixture(scope="module")
def context(cloud):
    """Refactored fitting context over the same cloud."""
    transform = SpaceTransform(convention=MANDEL_XY_LAST, deviatoric=True)
    return EstimatorContext(sigma_model=transform.forward(cloud), transform=transform)


def prepared(field):
    """Field mapped into model space, as the pipeline would."""
    return SpaceTransform(convention=MANDEL_XY_LAST, deviatoric=True).forward(field)


# --- Bandwidth fitting -------------------------------------------------------


def test_ray_bandwidth_matches_legacy(context, legacy_bundle):
    """Scott's rule and the covariance must reproduce the legacy bandwidth."""
    estimator = RayMarginalEstimator.fit(context)

    assert estimator.bandwidth_h == pytest.approx(legacy_bundle["bandwidth_h"], rel=1e-14)
    np.testing.assert_allclose(estimator.bandwidth_matrix, legacy_bundle["bandwidth_H"], **TOL)


def test_slice_bandwidth_matches_legacy(context, legacy_bundle):
    """The 5D subspace bandwidth must match the legacy v2 upgrade."""
    estimator = SliceEstimator.fit(context, mode="global")
    np.testing.assert_allclose(
        estimator.inverse_bandwidth, legacy_bundle["bandwidth_H5_inv"], rtol=1e-9, atol=1e-12
    )


def test_subspace_projection_matches_legacy(context, legacy_bundle):
    """Subspace coordinates must be identical, or every slice number shifts."""
    estimator = SliceEstimator.fit(context, mode="polar")
    np.testing.assert_allclose(estimator.y_data, legacy_bundle["sigma_data_5d"], **TOL)


def test_subspace_projection_is_an_isometry(context):
    """Norms must be preserved, so radii stay the same numbers in MPa."""
    estimator = SliceEstimator.fit(context, mode="polar")
    np.testing.assert_allclose(
        np.linalg.norm(estimator.y_data, axis=1),
        np.linalg.norm(context.sigma_model, axis=1),
        rtol=1e-12,
    )


# --- Field evaluation --------------------------------------------------------


def test_ray_global_matches_legacy(context, legacy_bundle, field):
    """Plain projection: probability, width and coverage must all agree."""
    result = RayMarginalEstimator.fit(context, localized=False).evaluate(prepared(field))
    expected_p, expected_sd, expected_cov = legacy.evaluate_field(field, legacy_bundle)

    np.testing.assert_allclose(result.probability, expected_p, **TOL)
    np.testing.assert_allclose(result.spread, expected_sd, **TOL)
    np.testing.assert_array_equal(result.covered, expected_cov)


@pytest.mark.parametrize("power", [4.0, 8.0, 16.0])
def test_ray_localized_matches_legacy(context, legacy_bundle, field, power):
    """Angle-weighted variant, across the exponents the sweep table covers."""
    result = RayMarginalEstimator.fit(context, localized=True, power=power).evaluate(
        prepared(field)
    )
    expected_p, expected_sd, expected_cov = legacy.evaluate_field_local(
        field, legacy_bundle, power=power
    )

    np.testing.assert_allclose(result.probability, expected_p, **TOL)
    np.testing.assert_allclose(result.spread, expected_sd, **TOL)
    np.testing.assert_array_equal(result.covered, expected_cov)


@pytest.mark.parametrize("kappa", [10.0, 20.0, 50.0])
def test_slice_polar_matches_legacy(context, legacy_bundle, field, kappa):
    """Polar mode, the default, across the calibration grid."""
    result = SliceEstimator.fit(context, mode="polar", kappa=kappa, tau_floor=2.0).evaluate(
        prepared(field)
    )
    expected_p, expected_spread, expected_cov, expected_eff = legacy.evaluate_field_slice(
        field, legacy_bundle, mode="polar", kappa=kappa, tau_floor=2.0
    )

    np.testing.assert_allclose(result.probability, expected_p, **TOL)
    np.testing.assert_allclose(result.spread, expected_spread, **TOL)
    np.testing.assert_array_equal(result.covered, expected_cov)
    np.testing.assert_allclose(result.diagnostics["eff_n"], expected_eff, **TOL)


def test_slice_global_matches_legacy(context, legacy_bundle, field):
    """Textbook slice through the full kernel."""
    result = SliceEstimator.fit(context, mode="global").evaluate(prepared(field))
    expected_p, expected_spread, expected_cov, _ = legacy.evaluate_field_slice(
        field, legacy_bundle, mode="global"
    )

    np.testing.assert_allclose(result.probability, expected_p, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(result.spread, expected_spread, rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(result.covered, expected_cov)


def test_slice_adapted_matches_legacy(context, legacy_bundle, field):
    """Separate radial and angular widths."""
    result = SliceEstimator.fit(
        context, mode="adapted", sigma_r=10.0, sigma_t=0.3, sigma_t_mode="angular"
    ).evaluate(prepared(field))
    expected_p, expected_spread, expected_cov, _ = legacy.evaluate_field_slice(
        field,
        legacy_bundle,
        mode="adapted",
        sigma_r=10.0,
        sigma_t=0.3,
        sigma_t_mode="angular",
    )

    np.testing.assert_allclose(result.probability, expected_p, **TOL)
    np.testing.assert_allclose(result.spread, expected_spread, **TOL)
    np.testing.assert_array_equal(result.covered, expected_cov)


def test_slice_censoring_matches_legacy(context, legacy_bundle, field):
    """With censoring on, a low effective sample size clears coverage."""
    result = SliceEstimator.fit(
        context, mode="polar", kappa=50.0, tau_floor=2.0, censor=True, min_eff_n=15.0
    ).evaluate(prepared(field))
    _, _, expected_cov, _ = legacy.evaluate_field_slice(
        field,
        legacy_bundle,
        mode="polar",
        kappa=50.0,
        tau_floor=2.0,
        censor=True,
        min_eff_n=15.0,
    )
    np.testing.assert_array_equal(result.covered, expected_cov)


def test_top_k_selection_matches_legacy(context, legacy_bundle, field):
    """Keeping only the heaviest points must reproduce the legacy shortcut."""
    result = SliceEstimator.fit(
        context, mode="polar", kappa=20.0, tau_floor=2.0, top_k=256
    ).evaluate(prepared(field))
    expected_p, _, _, _ = legacy.evaluate_field_slice(
        field, legacy_bundle, mode="polar", kappa=20.0, tau_floor=2.0, top_k=256
    )
    np.testing.assert_allclose(result.probability, expected_p, **TOL)


def test_batch_size_does_not_change_results(context, field):
    """Batching is an implementation detail and must not move any number."""
    prepared_field = prepared(field)
    small = SliceEstimator.fit(context, mode="polar", batch_size=37).evaluate(prepared_field)
    large = SliceEstimator.fit(context, mode="polar", batch_size=4096).evaluate(prepared_field)

    np.testing.assert_allclose(small.probability, large.probability, **TOL)
    np.testing.assert_allclose(small.spread, large.spread, **TOL)


# --- Independence of the two methods ----------------------------------------


def test_slice_needs_no_ray_estimator(context, legacy_bundle, field):
    """The slice must be computable on its own.

    In the legacy code ``SliceChannel.from_ray`` built the slice out of an
    already constructed ray model, so this was impossible. Nothing in this
    test touches the ray estimator.
    """
    result = SliceEstimator.fit(context, mode="polar", kappa=20.0, tau_floor=2.0).evaluate(
        prepared(field)
    )
    expected, _, _, _ = legacy.evaluate_field_slice(
        field, legacy_bundle, mode="polar", kappa=20.0, tau_floor=2.0
    )
    np.testing.assert_allclose(result.probability, expected, **TOL)
