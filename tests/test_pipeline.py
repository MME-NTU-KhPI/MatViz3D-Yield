"""Calibration provenance and the build/evaluate pipeline stages."""

from __future__ import annotations

import numpy as np
import pytest

from mvyield.mechanics.field import FieldMeta, StressField, Topology
from mvyield.mechanics.voigt import MANDEL_XY_LAST
from mvyield.model.basis import DEVIATORIC_BASIS
from mvyield.model.bundle import ModelBundle, YieldPointSet
from mvyield.model.calibration import (
    CalibrationError,
    PolarLogLikCalibrator,
    SweepReportCalibrator,
    get_calibrator,
)
from mvyield.model.transform import SpaceTransform
from mvyield.pipeline.build_model import build_model
from mvyield.pipeline.evaluate import evaluate_field
from mvyield.settings import ModelConfig


@pytest.fixture(scope="module")
def points():
    """A yield-point cloud spread over 12 geometries, with a split."""
    rng = np.random.default_rng(17)
    n, n_groups = 900, 12

    direction = rng.normal(size=(n, 5))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = 275.0 + 25.0 * direction[:, 0] + rng.normal(0.0, 8.0, n)
    sigma = (direction * radius[:, None]) @ DEVIATORIC_BASIS.T

    cloud = YieldPointSet(sigma6=sigma, group_id=rng.integers(0, n_groups, n))
    return cloud.with_split(seed=3)


@pytest.fixture(scope="module")
def transform():
    return SpaceTransform(convention=MANDEL_XY_LAST, deviatoric=True)


def make_field(n: int = 200) -> StressField:
    """A small uniaxial stress field ramping through the yield range."""
    magnitude = np.linspace(50.0, 500.0, n)
    sigma6 = np.zeros((n, 6))
    sigma6[:, 0] = magnitude

    points = np.column_stack([magnitude, np.zeros(n), np.zeros(n)])
    return StressField(
        points=points,
        sigma6=sigma6,
        topology=Topology.CLOUD2D,
        meta=FieldMeta(source="analytic", convention=MANDEL_XY_LAST),
    )


# --- Calibration -------------------------------------------------------------


def test_slice_calibrator_selects_from_the_grid(points, transform):
    grid = (5.0, 10.0, 20.0, 50.0, 100.0)
    result = PolarLogLikCalibrator().calibrate(points, transform, grid)

    assert result.value in grid
    assert result.source == "cv_loglik"
    assert result.objective is not None
    assert set(result.curve) == set(grid)


def test_slice_calibration_uses_held_out_geometries(points, transform):
    """Scoring must not be done on the points the estimate was built from."""
    result = PolarLogLikCalibrator().calibrate(points, transform, (10.0, 20.0, 50.0))
    assert result.diagnostics["split"] == "by group_id"
    assert result.diagnostics["n_train"] > 0
    assert result.diagnostics["n_eval"] > 0


def test_calibration_requires_a_split(transform):
    """Without held-out geometries there is nothing honest to score against."""
    rng = np.random.default_rng(0)
    unsplit = YieldPointSet(
        sigma6=rng.normal(size=(200, 6)) * 100, group_id=rng.integers(0, 5, 200)
    )
    with pytest.raises(CalibrationError, match="no split"):
        PolarLogLikCalibrator().calibrate(unsplit, transform, (10.0, 20.0))


def test_refinement_can_only_improve_the_objective(points, transform):
    grid = (5.0, 20.0, 100.0)
    coarse = PolarLogLikCalibrator().calibrate(points, transform, grid)
    fine = PolarLogLikCalibrator().calibrate(points, transform, grid, refine=True)

    assert fine.objective >= coarse.objective - 1e-9
    assert len(fine.curve) > len(coarse.curve)


def test_edge_selection_is_flagged(points, transform):
    """A value pinned at the edge of the grid means the grid is too narrow."""
    result = PolarLogLikCalibrator().calibrate(points, transform, (1.0, 2.0, 3.0))
    assert "edge" in result.diagnostics["note"]


def test_ray_calibrator_refuses_to_invent_a_criterion(points, transform):
    """The ray exponent has no criterion here; the sweep is a report only.

    Silently picking a value would present a hand-chosen threshold as an
    optimisation. The error carries the same table the legacy script
    printed, plus the config key to set.
    """
    with pytest.raises(CalibrationError) as excinfo:
        SweepReportCalibrator().calibrate(points, transform, (2.0, 4.0, 8.0, 16.0, 32.0))

    message = str(excinfo.value)
    assert "no automatic criterion" in message
    assert "model.kde_ray.power" in message
    assert "eff_n" in message


def test_ray_sweep_shows_effective_sample_size_falling(points, transform):
    """A narrower cone must inform the estimate with fewer points."""
    report = SweepReportCalibrator().sweep(points, transform, (2.0, 8.0, 32.0))
    assert report[32.0]["eff_n_median"] < report[2.0]["eff_n_median"]


def test_calibrator_registry_maps_parameters():
    assert isinstance(get_calibrator("kappa"), PolarLogLikCalibrator)
    assert isinstance(get_calibrator("power"), SweepReportCalibrator)
    with pytest.raises(KeyError, match="no calibrator"):
        get_calibrator("nonexistent")


def test_config_line_is_pasteable(points, transform):
    result = PolarLogLikCalibrator().calibrate(points, transform, (10.0, 20.0))
    assert result.as_config_line("kappa").startswith("kappa: {mode: manual")


# --- build_model -------------------------------------------------------------


def test_only_requested_methods_are_fitted(points):
    """Asking for the slice alone must not fit the ray."""
    config = ModelConfig(methods=("kde_slice",), kde_slice={"kappa": 20.0, "tau_floor": 2.0})
    bundle = build_model(points, config)

    assert bundle.methods == ("kde_slice",)
    assert "kde_ray" not in bundle.params


def test_manual_hyperparameter_is_recorded_as_manual(points):
    config = ModelConfig(methods=("kde_ray",), kde_ray={"power": 8.0})
    bundle = build_model(points, config)

    record = bundle.hyperparams["power"]
    assert record["value"] == 8.0
    assert record["source"] == "manual"
    assert record["calibrated_at"] is None


def test_calibrated_hyperparameter_records_its_criterion(points):
    config = ModelConfig(
        methods=("kde_slice",),
        kde_slice={
            "kappa": {"mode": "auto", "objective": "cv_loglik", "grid": [10.0, 20.0, 50.0]},
            "tau_floor": 2.0,
        },
    )
    bundle = build_model(points, config)

    record = bundle.hyperparams["kappa"]
    assert record["source"] == "cv_loglik"
    assert record["objective"] is not None
    assert record["calibrated_at"] is not None
    assert record["grid_searched"] == [10.0, 20.0, 50.0]


def test_frozen_hyperparameter_refuses_a_fresh_build(points):
    config = ModelConfig(methods=("kde_slice",), kde_slice={"kappa": {"mode": "frozen"}})
    with pytest.raises(CalibrationError, match="no earlier value"):
        build_model(points, config)


def test_deviatoric_flag_reaches_the_stored_space(points):
    for deviatoric in (True, False):
        config = ModelConfig(
            methods=("kde_ray",), deviatoric=deviatoric, kde_ray={"power": 8.0}
        )
        bundle = build_model(points, config)
        assert bundle.transform.deviatoric is deviatoric


def test_diagnostics_report_the_rank(points):
    """Rank 5 in a deviatoric space is expected, not a warning sign."""
    config = ModelConfig(methods=("kde_ray",), kde_ray={"power": 8.0})
    bundle = build_model(points, config)

    assert bundle.diagnostics["rank"] == 5
    assert 250.0 < bundle.diagnostics["r_mean"] < 300.0


def test_built_model_survives_storage(tmp_path, points):
    config = ModelConfig(
        methods=("kde_ray", "kde_slice"),
        primary="kde_slice",
        kde_ray={"power": 8.0},
        kde_slice={"kappa": 20.0, "tau_floor": 2.0},
    )
    bundle = build_model(points, config)
    restored = ModelBundle.load(bundle.save(tmp_path / "m.model.npz"))

    field = make_field()
    original = evaluate_field(field, bundle)
    reloaded = evaluate_field(field, restored)

    for method in original.methods:
        np.testing.assert_allclose(
            reloaded.probability(method), original.probability(method), rtol=1e-12
        )


# --- evaluate ----------------------------------------------------------------


def test_evaluation_covers_every_requested_method(points):
    config = ModelConfig(
        methods=("analytic", "kde_ray", "kde_slice"),
        primary="kde_slice",
        kde_ray={"power": 8.0},
        kde_slice={"kappa": 20.0, "tau_floor": 2.0},
        analytic={"mode": "from_dataset"},
    )
    result = evaluate_field(make_field(), build_model(points, config))

    assert set(result.outputs) == {"analytic", "kde_ray", "kde_slice"}
    assert result.methods[0] == "kde_slice"
    assert result.runs_comparison


def test_single_method_disables_comparison(points):
    config = ModelConfig(methods=("kde_ray",), kde_ray={"power": 8.0})
    result = evaluate_field(make_field(), build_model(points, config))

    assert not result.runs_comparison
    assert result.methods == ("kde_ray",)


def test_subset_of_methods_can_be_evaluated(points):
    config = ModelConfig(
        methods=("kde_ray", "kde_slice"),
        kde_ray={"power": 8.0},
        kde_slice={"kappa": 20.0, "tau_floor": 2.0},
    )
    bundle = build_model(points, config)
    result = evaluate_field(make_field(), bundle, methods=("kde_slice",))

    assert set(result.outputs) == {"kde_slice"}


def test_requesting_an_unfitted_method_is_refused(points):
    config = ModelConfig(methods=("kde_ray",), kde_ray={"power": 8.0})
    bundle = build_model(points, config)

    with pytest.raises(ValueError, match="no parameters for"):
        evaluate_field(make_field(), bundle, methods=("kde_slice",))


def test_unit_mismatch_is_warned_about(points, caplog):
    """A Pa/MPa mix-up produces a plausible-looking field, not an error."""
    config = ModelConfig(methods=("kde_ray",), kde_ray={"power": 8.0})
    bundle = build_model(points, config)

    field = make_field()
    field.sigma6 = field.sigma6 * 1.0e6  # MPa values left in Pa

    with caplog.at_level("WARNING"):
        evaluate_field(field, bundle)
    assert "Pa/MPa" in caplog.text


def test_results_round_trip_through_storage(tmp_path, points):
    """Figures are redrawn from this file, so it must carry every field."""
    config = ModelConfig(
        methods=("kde_ray", "kde_slice"),
        kde_ray={"power": 8.0},
        kde_slice={"kappa": 20.0, "tau_floor": 2.0},
    )
    result = evaluate_field(make_field(), build_model(points, config))
    path = result.save(tmp_path / "results.npz")

    with np.load(path) as stored:
        np.testing.assert_allclose(
            stored["kde_slice/probability"], result.probability("kde_slice"), rtol=1e-12
        )
        assert "kde_slice/diag/eff_n" in stored
        assert "kde_ray/spread" in stored


def test_summary_reports_each_method(points):
    config = ModelConfig(
        methods=("kde_ray", "analytic"),
        kde_ray={"power": 8.0},
        analytic={"mode": "from_dataset"},
    )
    summary = evaluate_field(make_field(), build_model(points, config)).summary()

    assert set(summary) == {"kde_ray", "analytic"}
    assert summary["kde_ray"]["n_points"] == 200.0


def test_transform_is_applied_once_for_all_methods(points):
    """Every method must see the same prepared coordinates.

    Adding hydrostatic pressure to the query must leave a deviatoric model
    unmoved. If any estimator transformed the field itself, this is where
    the spaces would drift apart.
    """
    config = ModelConfig(
        methods=("kde_ray", "kde_slice"),
        kde_ray={"power": 8.0},
        kde_slice={"kappa": 20.0, "tau_floor": 2.0},
    )
    bundle = build_model(points, config)

    plain = make_field()
    pressurised = make_field()
    pressurised.sigma6 = pressurised.sigma6 + np.array([200.0, 200.0, 200.0, 0, 0, 0])

    baseline = evaluate_field(plain, bundle)
    shifted = evaluate_field(pressurised, bundle)

    for method in baseline.methods:
        np.testing.assert_allclose(
            shifted.probability(method), baseline.probability(method), rtol=1e-9, atol=1e-12
        )
