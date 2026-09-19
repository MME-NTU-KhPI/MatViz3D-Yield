"""Yield-point extraction: crystallography, criteria, and the whole pass."""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="dataset ingest needs the 'ingest' extra")

from mvyield.ingest.analysis import (
    NEIGHBOUR_OFFSETS,
    EigenTracker,
    analyse_dataset,
    get_selection_rule,
    schmid_step,
    von_mises_step,
)
from mvyield.ingest.artefacts import IngestResult, StepTable
from mvyield.ingest.material import MaterialProperties
from mvyield.ingest.reader import DATASET_CONVENTION, DatasetReader
from mvyield.ingest.slip import (
    BCC_48,
    euler_to_rotation,
    independent_systems,
    resolved_shear,
    schmid_tensors,
    selfcheck,
    slip_systems,
)
from mvyield.mechanics.voigt import MANDEL_XY_LAST, convert, von_mises
from mvyield.model.bundle import YieldPointSet
from mvyield.settings import ConfigError

from test_ingest import REAL_DATASET, write_dataset

CRSS = 150.0


@pytest.fixture
def material():
    return MaterialProperties(crss=CRSS, structure="BCC_48", sigma_y=275.0)


@pytest.fixture
def dataset(tmp_path):
    """A dataset with several geometries and a load path."""
    return write_dataset(tmp_path / "analysis.hdf5", geometry_ids=("0", "1", "2", "3"),
                         n_steps=4, cube=3, n_grains=5)


# --- Crystallography ---------------------------------------------------------


def test_every_slip_direction_lies_in_its_plane():
    """n . m = 0 is what makes a slip system physical."""
    assert max(selfcheck().values()) == 0.0


def test_slip_vectors_are_unit_length():
    for name in ("BCC", "BCC_48", "FCC"):
        planes, directions = slip_systems(name)
        np.testing.assert_allclose(np.linalg.norm(planes, axis=1), 1.0)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)


def test_unknown_structure_is_refused():
    with pytest.raises(ConfigError, match="no slip systems"):
        slip_systems("diamond")


def test_bcc_48_lists_each_system_twice():
    """Both +b and -b are tabulated, so 48 entries are 24 distinct systems.

    This is why an active-system count that counts table entries can never
    return one.
    """
    planes, directions = slip_systems("BCC_48")
    assert len(planes) == 48
    assert len(independent_systems(planes, directions)) == 24


def test_families_without_reversed_pairs_are_left_alone():
    for name in ("BCC", "FCC"):
        planes, directions = slip_systems(name)
        assert len(independent_systems(planes, directions)) == len(planes)


def test_rotation_matrices_are_orthonormal():
    rng = np.random.default_rng(3)
    rotations = euler_to_rotation(rng.uniform(0, 360, (16, 3)))

    for rotation in rotations:
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_zero_euler_angles_give_the_identity():
    np.testing.assert_allclose(euler_to_rotation([0.0, 0.0, 0.0])[0], np.eye(3), atol=1e-15)


def test_schmid_tensors_agree_with_rotating_the_stress():
    """The fast path must equal the obvious one.

    Rotating the systems once per grain replaces rotating the stress once
    per point. That is only worth doing if the two are the same number, so
    the readable form is kept and checked against.
    """
    rng = np.random.default_rng(11)
    planes, directions = slip_systems("BCC_48")
    rotations = euler_to_rotation(rng.uniform(0, 360, (4, 3)))

    sigma6_raw = rng.normal(0, 100, (7, 6))
    tensors = np.zeros((7, 3, 3))
    slot = DATASET_CONVENTION.slot_of
    tensors[:, 0, 0] = sigma6_raw[:, slot("XX")]
    tensors[:, 1, 1] = sigma6_raw[:, slot("YY")]
    tensors[:, 2, 2] = sigma6_raw[:, slot("ZZ")]
    tensors[:, 0, 1] = tensors[:, 1, 0] = sigma6_raw[:, slot("XY")]
    tensors[:, 1, 2] = tensors[:, 2, 1] = sigma6_raw[:, slot("YZ")]
    tensors[:, 0, 2] = tensors[:, 2, 0] = sigma6_raw[:, slot("XZ")]

    schmid = schmid_tensors(planes, directions, rotations)
    sigma_mandel = convert(sigma6_raw, DATASET_CONVENTION, MANDEL_XY_LAST)

    for grain, rotation in enumerate(rotations):
        rotated = np.einsum("ij,njk,lk->nil", rotation, tensors, rotation)
        direct = resolved_shear(rotated, planes, directions)
        fast = np.abs(sigma_mandel @ schmid[grain].T)
        np.testing.assert_allclose(fast, direct, atol=1e-10)


# --- Eigen tracking ----------------------------------------------------------


def test_tracker_orders_by_magnitude_on_the_first_step():
    tracker = EigenTracker()
    values, _ = tracker.track(np.diag([1.0, 5.0, 3.0]))

    np.testing.assert_allclose(values, [5.0, 3.0, 1.0])


def test_tracker_keeps_axes_in_place_across_steps():
    """A principal axis must not appear to swap with another between steps."""
    tracker = EigenTracker()
    tracker.track(np.diag([5.0, 3.0, 1.0]))
    values, _ = tracker.track(np.diag([2.9, 3.1, 1.0]))

    # Without tracking, sorting by magnitude would report 3.1 first.
    np.testing.assert_allclose(values, [2.9, 3.1, 1.0])


def test_reset_forgets_the_previous_geometry():
    tracker = EigenTracker()
    tracker.track(np.diag([5.0, 3.0, 1.0]))
    tracker.reset()
    values, _ = tracker.track(np.diag([2.9, 3.1, 1.0]))

    np.testing.assert_allclose(values, [3.1, 2.9, 1.0])


# --- Criteria ----------------------------------------------------------------


def test_von_mises_step_scales_to_the_strength(dataset):
    with DatasetReader(dataset) as reader:
        step = reader.step("1", 2)

    sigma6, k_factor, peak, active = von_mises_step(step, 275.0)

    assert von_mises(sigma6, DATASET_CONVENTION)[0] == pytest.approx(275.0)
    assert k_factor * peak == pytest.approx(275.0)
    assert active == 0


def test_schmid_step_brings_the_worst_system_to_the_crss(dataset):
    """The definition of the yield point: one system exactly at the CRSS."""
    with DatasetReader(dataset) as reader:
        geometry = reader.geometry("0")
        step = reader.step("0", 1)

    planes, directions = slip_systems("BCC_48")
    rotations = euler_to_rotation(geometry.euler_deg)
    sigma6, k_factor, tau_max, active = schmid_step(
        step, geometry.voxels, rotations, planes, directions, CRSS
    )

    assert k_factor * tau_max == pytest.approx(CRSS)
    assert 1 <= active <= 24

    # The yield point is one of the step's own states, scaled by k.
    scaled = step.stress * k_factor
    assert np.isclose(scaled, sigma6).all(axis=1).any()


def test_schmid_neighbourhood_can_only_raise_the_resolved_shear(dataset):
    """Searching seven orientations must never find less than searching one."""
    with DatasetReader(dataset) as reader:
        geometry = reader.geometry("0")
        step = reader.step("0", 1)

    planes, directions = slip_systems("BCC_48")
    rotations = euler_to_rotation(geometry.euler_deg)
    tensors = schmid_tensors(planes, directions, rotations)

    _, _, with_neighbours, _ = schmid_step(
        step, geometry.voxels, rotations, planes, directions, CRSS, tensors
    )

    # Own voxel only: the same computation over a one-entry offset list.
    own = np.zeros(1)
    indices = np.rint(step.coords).astype(int)
    grains = geometry.voxels[indices[:, 0], indices[:, 1], indices[:, 2]]
    sigma_mandel = convert(step.stress, DATASET_CONVENTION, MANDEL_XY_LAST)
    own = np.abs(np.einsum("nk,nmk->nm", sigma_mandel, tensors[grains])).max()

    assert with_neighbours >= own - 1e-12


def test_seven_offsets_are_the_point_and_its_faces():
    assert len(NEIGHBOUR_OFFSETS) == 7
    assert NEIGHBOUR_OFFSETS[0] == (0, 0, 0)
    assert all(sum(abs(c) for c in offset) == 1 for offset in NEIGHBOUR_OFFSETS[1:])


# --- Step selection ----------------------------------------------------------


def test_unknown_selection_rule_lists_the_known_ones():
    with pytest.raises(ConfigError, match="available"):
        get_selection_rule("whichever")


def test_every_geometry_gets_one_representative(dataset, material):
    with DatasetReader(dataset) as reader:
        result = analyse_dataset(reader, material)

    assert result.steps.representative.sum() == 4
    assert len(np.unique(result.steps.geometry_id[result.steps.representative])) == 4


def test_selection_rules_can_disagree(dataset, material):
    with DatasetReader(dataset) as reader:
        by_macro = analyse_dataset(reader, material, select_step="max_macro_sz")
        by_last = analyse_dataset(reader, material, select_step="last")

    assert by_last.steps.step[by_last.steps.representative].tolist() == [3, 3, 3, 3]
    assert by_macro.steps.n_rows == by_last.steps.n_rows


# --- Whole-dataset pass ------------------------------------------------------


def test_every_step_becomes_a_yield_point(dataset, material):
    with DatasetReader(dataset) as reader:
        result = analyse_dataset(reader, material)

    assert result.points.n_points == 16  # 4 geometries x 4 steps
    assert result.points.n_groups == 4
    assert result.n_skipped == 0


def test_cloud_is_written_in_the_model_convention(dataset, material):
    """The model applies no conversion of its own, so ingest must land there."""
    with DatasetReader(dataset) as reader:
        result = analyse_dataset(reader, material)

    assert result.points.meta["convention"] == MANDEL_XY_LAST.name
    np.testing.assert_allclose(
        von_mises(result.points.sigma6, MANDEL_XY_LAST),
        result.steps.von_mises,
        rtol=1e-12,
    )


def test_metadata_records_material_and_analysis(dataset, material):
    with DatasetReader(dataset) as reader:
        result = analyse_dataset(reader, material)

    assert result.points.meta["analysis"] == "SCHMID"
    assert result.points.meta["material"]["crss"] == CRSS
    assert "analysis.hdf5" in result.points.meta["source"]


def test_von_mises_mode_needs_no_microstructure(tmp_path, material):
    """A dataset without grains can still be analysed against a strength."""
    path = write_dataset(tmp_path / "flat.hdf5", geometry_ids=("0",), n_steps=2)
    with h5py.File(path, "a") as f:
        del f["0"]["voxels"]

    with DatasetReader(path) as reader:
        result = analyse_dataset(reader, material, analysis="VON_MISES")

    np.testing.assert_allclose(result.steps.von_mises, 275.0, rtol=1e-12)


def test_schmid_without_microstructure_says_what_to_do(tmp_path, material):
    path = write_dataset(tmp_path / "flat.hdf5", geometry_ids=("0",), n_steps=1)
    with h5py.File(path, "a") as f:
        del f["0"]["voxels"]

    with DatasetReader(path) as reader, pytest.raises(ConfigError, match="VON_MISES"):
        analyse_dataset(reader, material)


def test_missing_crss_is_reported(dataset):
    with DatasetReader(dataset) as reader, pytest.raises(ConfigError, match="material_override"):
        analyse_dataset(reader, MaterialProperties(crss=None))


def test_unknown_analysis_is_refused(dataset, material):
    with DatasetReader(dataset) as reader, pytest.raises(ConfigError, match="SCHMID"):
        analyse_dataset(reader, material, analysis="TRESCA")


def test_steps_with_no_stress_are_dropped_not_zeroed(tmp_path, material):
    """A zero point would sit at the origin and drag every density towards it."""
    path = write_dataset(tmp_path / "zeroed.hdf5", geometry_ids=("0",), n_steps=3)
    with h5py.File(path, "a") as f:
        f["0"]["ls_1"]["results"][:, 7:13] = 0.0

    with DatasetReader(path) as reader:
        result = analyse_dataset(reader, material)

    assert result.points.n_points == 2
    assert result.n_skipped == 1
    assert np.linalg.norm(result.points.sigma6, axis=1).min() > 0.0


def test_non_finite_steps_are_dropped(tmp_path, material):
    path = write_dataset(tmp_path / "nan.hdf5", geometry_ids=("0",), n_steps=3)
    with h5py.File(path, "a") as f:
        f["0"]["ls_2"]["results"][3, 7] = np.nan

    with DatasetReader(path) as reader:
        result = analyse_dataset(reader, material)

    assert result.points.n_points == 2
    assert result.skipped == {"non-finite stress": 1}


def test_a_dataset_with_nothing_usable_is_reported(tmp_path, material):
    path = write_dataset(tmp_path / "dead.hdf5", geometry_ids=("0",), n_steps=2)
    with h5py.File(path, "a") as f:
        for step in ("ls_0", "ls_1"):
            f["0"][step]["results"][:, 7:13] = 0.0

    with DatasetReader(path) as reader, pytest.raises(ConfigError, match="no load step"):
        analyse_dataset(reader, material)


def test_progress_is_reported_per_geometry(dataset, material):
    seen = []
    with DatasetReader(dataset) as reader:
        analyse_dataset(reader, material, progress=lambda g, done, total: seen.append((g, done, total)))

    assert seen == [("0", 1, 4), ("1", 2, 4), ("2", 3, 4), ("3", 4, 4)]


# --- Artefacts ---------------------------------------------------------------


def test_artefacts_round_trip(tmp_path, dataset, material):
    with DatasetReader(dataset) as reader:
        result = analyse_dataset(reader, material)

    written = result.save(tmp_path / "models")

    points = YieldPointSet.load(written["points"])
    steps = StepTable.load(written["steps"])

    np.testing.assert_allclose(points.sigma6, result.points.sigma6)
    np.testing.assert_allclose(steps.k_factor, result.steps.k_factor)
    assert steps.geometry_id.tolist() == result.steps.geometry_id.tolist()
    assert points.meta["analysis"] == "SCHMID"


def test_artefacts_are_named_by_content(tmp_path, dataset, material):
    """An unchanged dataset must not invalidate a cached model."""
    with DatasetReader(dataset) as reader:
        first = analyse_dataset(reader, material)
        second = analyse_dataset(reader, material)

    assert first.points.content_hash() == second.points.content_hash()
    assert set(first.save(tmp_path / "a")) == set(second.save(tmp_path / "b"))


def test_table_and_cloud_must_stay_aligned(dataset, material):
    with DatasetReader(dataset) as reader:
        result = analyse_dataset(reader, material)

    with pytest.raises(ValueError, match="aligned"):
        IngestResult(
            points=result.points,
            steps=result.steps.select(np.arange(result.steps.n_rows) < 3),
            material=material,
        )


def test_representatives_selects_one_row_per_geometry(dataset, material):
    with DatasetReader(dataset) as reader:
        result = analyse_dataset(reader, material)

    assert result.steps.representatives().n_rows == 4


def test_description_reports_the_scatter(dataset, material):
    with DatasetReader(dataset) as reader:
        text = analyse_dataset(reader, material).describe()

    assert "yield points" in text
    assert "sigma_vm" in text


# --- Against the reference ---------------------------------------------------


def reference_run(geometries):
    """Analyse the generated dataset and collect the reference beside it."""
    from mvyield.ingest.material import read_material

    with DatasetReader(REAL_DATASET) as reader:
        result = analyse_dataset(reader, read_material(reader), geometries=geometries)
        chosen = geometries if geometries is not None else reader.geometry_ids

        reference = np.concatenate(
            [reader.ground_truth(g)["yield_sigma"] * 1.0e-6 for g in chosen]
        )
        reference_k = np.concatenate([reader.ground_truth(g)["k_factor"] for g in chosen])

    return result, reference, reference_k


@pytest.mark.skipif(not REAL_DATASET.exists(), reason=f"{REAL_DATASET.name} not present")
def test_reproduces_the_generated_reference_step_by_step():
    """Recover ``ground_truth`` from the recorded field, step by step.

    The generator computed its reference yield points with the same
    seven-point orientation search this package uses, on the noise-free
    field that ``synth_clean.hdf5`` stores. So the two must agree to
    floating point, not merely in the mean -- a far sharper statement than
    the summary in ``info.txt``, and the one that would catch a wrong
    rotation, a swapped shear slot or a mistyped Miller index.

    Four geometries are enough to make that point; the whole dataset is
    checked by the ``slow`` test below.
    """
    result, reference, reference_k = reference_run(["0", "1", "2", "3"])

    assert result.points.n_points == 480
    np.testing.assert_allclose(result.steps.k_factor, reference_k, rtol=1e-12)
    np.testing.assert_allclose(
        result.points.sigma6,
        convert(reference, DATASET_CONVENTION, MANDEL_XY_LAST),
        atol=1e-9,
    )


@pytest.mark.slow
@pytest.mark.skipif(not REAL_DATASET.exists(), reason=f"{REAL_DATASET.name} not present")
def test_whole_dataset_matches_its_recorded_summary():
    """All 40 geometries, against ``datasets_synth/info.txt``. Minutes."""
    result, reference, reference_k = reference_run(None)

    assert result.points.n_points == 4800
    np.testing.assert_allclose(result.steps.k_factor, reference_k, rtol=1e-12)

    equivalent = result.steps.von_mises
    assert equivalent.mean() == pytest.approx(268.66, abs=0.01)
    assert equivalent.std() == pytest.approx(6.72, abs=0.01)
