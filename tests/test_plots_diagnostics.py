"""The diagnostics preset, the split it needs, and redrawing a finished run."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mvyield.model.bundle import SPLIT_TRAIN, YieldPointSet
from mvyield.viz import REGISTRY, Space

from test_compare import draw, make_result, paper_context, shell_bundle  # noqa: F401

DIAGNOSTIC_FIGURES = (
    "coverage_map",
    "eff_n_map",
    "uncertainty_band",
    "weight_profile",
    "calibration_curve",
    "bandwidth_sensitivity",
    "dataset_3d_scatter",
)


# --- Registration ------------------------------------------------------------


def test_every_diagnostic_figure_is_registered():
    for name in DIAGNOSTIC_FIGURES:
        assert name in REGISTRY


def test_diagnostics_are_not_in_the_paper_preset():
    """Two audiences, two figure lists: the result, and whether to believe it."""
    for name in DIAGNOSTIC_FIGURES:
        assert "paper" not in REGISTRY[name].groups
        assert "diagnostics" in REGISTRY[name].groups


def test_eff_n_map_declares_the_diagnostic_it_needs():
    """Without eff_n it must be skipped, not drawn empty."""
    assert REGISTRY["eff_n_map"].needs_diagnostics == ("eff_n",)


def test_sensitivity_needs_the_method_it_sweeps():
    assert REGISTRY["bandwidth_sensitivity"].needs == ("kde_slice",)


def test_diagnostic_spaces_are_declared():
    assert REGISTRY["coverage_map"].space is Space.FIELD
    assert REGISTRY["uncertainty_band"].space is Space.PROBE
    assert REGISTRY["weight_profile"].space is Space.AGGREGATE
    assert REGISTRY["dataset_3d_scatter"].space is Space.MATERIAL


# --- Drawing -----------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [n for n in DIAGNOSTIC_FIGURES if n != "bandwidth_sensitivity"],
)
def test_diagnostic_figure_draws(name, paper_context):
    draw(name, paper_context)


def test_sensitivity_says_so_when_the_slice_did_not_run(paper_context):
    """The bundle in this context has no slice, so the panel explains itself."""
    assert "kde_slice" not in paper_context.bundle.methods
    draw("bandwidth_sensitivity", paper_context)


def test_calibration_curve_explains_a_manual_choice(paper_context):
    """Nothing was calibrated here, and the figure must say that."""
    from mvyield.viz.theme import THEME

    assert not any("curve" in r for r in paper_context.bundle.hyperparams.values())

    figure, ax = plt.subplots()
    try:
        REGISTRY["calibration_curve"].function(paper_context, ax, THEME)
        text = " ".join(child.get_text() for child in ax.texts)
        assert "calibrated" in text
    finally:
        plt.close(figure)


def test_calibration_curve_draws_a_recorded_sweep(paper_context):
    """A curve recorded by the calibrator must reach the figure."""
    from mvyield.viz.theme import THEME

    paper_context.bundle.record_hyperparam(
        "kappa", 20.0, source="cv_loglik", objective=-1.9,
        grid=[5.0, 10.0, 20.0],
        curve={5.0: -2.1, 10.0: -2.0, 20.0: -1.9},
    )
    figure, ax = plt.subplots()
    try:
        REGISTRY["calibration_curve"].function(paper_context, ax, THEME)
        assert ax.lines
    finally:
        plt.close(figure)


def test_figures_needing_a_model_explain_its_absence(paper_context):
    paper_context.bundle = None

    for name in ("weight_profile", "calibration_curve", "bandwidth_sensitivity"):
        draw(name, paper_context)


# --- The curve the bundle has to carry ---------------------------------------


def test_bundle_stores_the_calibration_curve(shell_bundle):
    """``CalibrationResult.curve`` exists for this plot; it must survive."""
    shell_bundle.record_hyperparam(
        "kappa", 20.0, source="cv_loglik", objective=-1.9,
        grid=[5.0, 20.0], curve={20.0: -1.9, 5.0: -2.1},
    )
    stored = shell_bundle.hyperparams["kappa"]["curve"]

    assert stored["grid"] == [5.0, 20.0]
    assert stored["objective"] == [-2.1, -1.9]


def test_curve_survives_a_bundle_round_trip(tmp_path, shell_bundle):
    """JSON keys are strings, so the curve is stored as two lists."""
    from mvyield.model.bundle import ModelBundle

    shell_bundle.record_hyperparam(
        "kappa", 20.0, source="cv_loglik", curve={5.0: -2.1, 20.0: -1.9},
    )
    path = shell_bundle.save(tmp_path / "m.model.npz")
    restored = ModelBundle.load(path)

    assert restored.hyperparams["kappa"]["curve"]["grid"] == [5.0, 20.0]


# --- The split calibration needs ---------------------------------------------


def test_ingest_assigns_a_split(tmp_path):
    """Calibration needs held-out geometries and cannot invent them later."""
    h5py = pytest.importorskip("h5py")
    from mvyield.ingest import DatasetReader, analyse_dataset
    from mvyield.ingest.material import MaterialProperties

    from test_ingest import write_dataset

    path = write_dataset(tmp_path / "split.hdf5", geometry_ids=("0", "1", "2", "3"),
                         n_steps=3, cube=3, n_grains=5)
    with DatasetReader(path) as reader:
        result = analyse_dataset(reader, MaterialProperties(crss=150.0))

    assert result.points.split is not None
    assert set(np.unique(result.points.split)) == {0, 1, 2}
    assert (result.points.split == SPLIT_TRAIN).any()


def test_split_is_by_geometry_not_by_point(tmp_path):
    """Neighbouring steps of one RVE are near duplicates; splitting points
    would put them on both sides and flatter the validation score."""
    h5py = pytest.importorskip("h5py")
    from mvyield.ingest import DatasetReader, analyse_dataset
    from mvyield.ingest.material import MaterialProperties

    from test_ingest import write_dataset

    path = write_dataset(tmp_path / "split.hdf5", geometry_ids=("0", "1", "2", "3"),
                         n_steps=3, cube=3, n_grains=5)
    with DatasetReader(path) as reader:
        points = analyse_dataset(reader, MaterialProperties(crss=150.0)).points

    for group in np.unique(points.group_id):
        assert len(np.unique(points.split[points.group_id == group])) == 1


def test_too_few_geometries_warns_instead_of_failing(tmp_path, caplog):
    """One RVE is a legitimate dataset; --geometries 0 produces exactly that."""
    import logging

    h5py = pytest.importorskip("h5py")
    from mvyield.ingest import DatasetReader, analyse_dataset
    from mvyield.ingest.material import MaterialProperties

    from test_ingest import write_dataset

    path = write_dataset(tmp_path / "one.hdf5", geometry_ids=("0",), n_steps=3,
                         cube=3, n_grains=5)
    with DatasetReader(path) as reader, caplog.at_level(logging.WARNING):
        result = analyse_dataset(reader, MaterialProperties(crss=150.0))

    assert result.points.split is None
    assert "cannot be split" in caplog.text


def test_split_is_reproducible(tmp_path):
    h5py = pytest.importorskip("h5py")
    from mvyield.ingest import DatasetReader, analyse_dataset
    from mvyield.ingest.material import MaterialProperties

    from test_ingest import write_dataset

    path = write_dataset(tmp_path / "seed.hdf5", geometry_ids=("0", "1", "2", "3"),
                         n_steps=2, cube=3, n_grains=5)
    material = MaterialProperties(crss=150.0)

    with DatasetReader(path) as reader:
        first = analyse_dataset(reader, material, seed=7).points.split
        same = analyse_dataset(reader, material, seed=7).points.split
        other = analyse_dataset(reader, material, seed=8).points.split

    np.testing.assert_array_equal(first, same)
    assert not np.array_equal(first, other) or len(np.unique(first)) == 1


def test_split_survives_the_artefact(tmp_path, shell_bundle):
    points = YieldPointSet(
        sigma6=shell_bundle.points.sigma6,
        group_id=shell_bundle.points.group_id,
    ).with_split(seed=0)

    written = points.save(tmp_path / "p.yieldpoints.npz")
    restored = YieldPointSet.load(written)

    np.testing.assert_array_equal(restored.split, points.split)
