"""Dataset ingest: structure discovery, units, and material provenance."""

from __future__ import annotations

import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="dataset ingest needs the 'ingest' extra")

from mvyield.ingest.material import (
    DEFAULT_STRUCTURE,
    MaterialProperties,
    read_material,
)
from mvyield.ingest.reader import (
    DATASET_CONVENTION,
    DatasetError,
    DatasetReader,
)
from mvyield.mechanics.voigt import MANDEL_XY_LAST, von_mises
from mvyield.paths import find_repo_root
from mvyield.settings import ConfigError

REAL_DATASET = (
    find_repo_root().parent / "BenchmarkFramework" / "datasets_synth" / "synth_clean.hdf5"
)
"""A generated dataset kept outside the package; tests using it skip if absent."""


# --- Fixtures ----------------------------------------------------------------


def stress_pattern(geometry: int, step: int, n_points: int) -> np.ndarray:
    """A per-point stress block in Pa, distinct for every geometry and step.

    Component ``j`` of every point holds ``geometry*100 + step*10 + j`` MPa,
    so a misread column, step or geometry shows up as a wrong number rather
    than as plausible noise.
    """
    values = geometry * 100.0 + step * 10.0 + np.arange(6.0)
    return np.tile(values, (n_points, 1)) * 1.0e6


def write_dataset(
    path,
    geometry_ids=("0", "1"),
    n_steps=3,
    cube=2,
    n_grains=4,
    attrs=None,
    ground_truth=False,
    last_set=False,
    columns=19,
    omit=(),
):
    """Write a miniature dataset with the MatViz3D layout."""
    n_points = cube**3
    coords = np.array(np.unravel_index(np.arange(n_points), (cube, cube, cube))).T

    with h5py.File(path, "w") as f:
        for key, value in (attrs or {}).items():
            f.attrs[key] = value

        if last_set:
            f.create_dataset("last_set", data=np.int32(len(geometry_ids)))

        for geometry_index, geometry_id in enumerate(geometry_ids):
            group = f.create_group(geometry_id)
            group.create_dataset("voxels", data=np.zeros((cube, cube, cube), np.int32))
            group.create_dataset("local_cs", data=np.zeros((n_grains, 3)))
            group.create_dataset("cubeSize", data=np.int32(cube))
            group.create_dataset("P_matrix", data=np.eye(6))

            for step in range(n_steps):
                ls = group.create_group(f"ls_{step}")
                results = np.zeros((n_points, columns))
                if columns >= 19:
                    results[:, 0] = np.arange(1, n_points + 1)
                    results[:, 1:4] = coords
                    results[:, 7:13] = stress_pattern(geometry_index, step, n_points)
                    results[:, 13:19] = 1.0e-4

                if "results" not in omit:
                    ls.create_dataset("results", data=results)
                if "results_avg" not in omit:
                    ls.create_dataset("results_avg", data=results.mean(axis=0))
                if "eps_as_loading" not in omit:
                    ls.create_dataset("eps_as_loading", data=np.full(6, 1.0e-4))

        if ground_truth:
            root = f.create_group("ground_truth")
            for geometry_id in geometry_ids:
                gt = root.create_group(geometry_id)
                gt.create_dataset("yield_sigma", data=np.full((n_steps, 6), 275.0e6))
                gt.create_dataset("k_factor", data=np.full(n_steps, 2.0))
    return path


@pytest.fixture
def dataset(tmp_path):
    """A two-geometry dataset with root attributes, as a generator writes it."""
    return write_dataset(
        tmp_path / "mini.hdf5",
        attrs={
            "crss": 150.0e6,
            "units": "stress: Pa, strain: dimensionless, euler: degrees",
            "generator": "test",
            "noise_level": "clean",
            "gen_config": json.dumps({"c11": 168.4e9, "c12": 121.4e9, "c44": 75.4e9}),
        },
        ground_truth=True,
        last_set=True,
    )


# --- Structure ---------------------------------------------------------------


def test_geometries_sort_numerically(tmp_path):
    """Ten must follow nine, not two."""
    ids = tuple(str(i) for i in range(12))
    path = write_dataset(tmp_path / "many.hdf5", geometry_ids=ids, n_steps=1)

    with DatasetReader(path) as reader:
        assert reader.geometry_ids == ids
        assert reader.n_geometries == 12


def test_bookkeeping_and_reference_groups_are_not_geometries(dataset):
    """A geometry is a group holding load steps; the rest is not one.

    ``last_set`` is a scalar dataset and ``ground_truth`` holds reference
    values, so a structural test excludes both without a list of names to
    keep up to date.
    """
    with DatasetReader(dataset) as reader:
        assert reader.geometry_ids == ("0", "1")


def test_geometry_ids_may_start_at_one(tmp_path):
    """Exported datasets number their RVEs from one."""
    path = write_dataset(tmp_path / "one.hdf5", geometry_ids=("1", "2", "3"), n_steps=1)

    with DatasetReader(path) as reader:
        assert reader.geometry_ids == ("1", "2", "3")
        assert reader.geometry("1").index == 0


def test_geometry_carries_its_microstructure(dataset):
    with DatasetReader(dataset) as reader:
        geometry = reader.geometry("0")

    assert geometry.cube_size == 2
    assert geometry.voxels.shape == (2, 2, 2)
    assert geometry.n_grains == 4
    assert geometry.p_matrix.shape == (6, 6)


# --- Load steps --------------------------------------------------------------


def test_stress_is_converted_to_mpa(dataset):
    """The file stores Pa; everything downstream expects MPa."""
    with DatasetReader(dataset) as reader:
        step = reader.step("1", 2)

    # geometry index 1, step 2, component j -> 100 + 20 + j
    np.testing.assert_allclose(step.stress[0], [120.0, 121.0, 122.0, 123.0, 124.0, 125.0])
    np.testing.assert_allclose(step.stress_avg, step.stress[0])


def test_step_shapes_and_columns(dataset):
    with DatasetReader(dataset) as reader:
        step = reader.step("0", 0)

    assert step.n_points == 8
    assert step.stress.shape == (8, 6)
    assert step.strain.shape == (8, 6)
    assert step.coords.shape == (8, 3)
    assert step.node_id.tolist() == list(range(1, 9))
    np.testing.assert_allclose(step.eps_as_loading, 1.0e-4)


def test_steps_iterate_in_order(dataset):
    with DatasetReader(dataset) as reader:
        assert [s.index for s in reader.steps("0")] == [0, 1, 2]


def test_steps_sort_by_number_not_by_name(tmp_path):
    """``ls_10`` must come after ``ls_9``."""
    path = write_dataset(tmp_path / "steps.hdf5", geometry_ids=("0",), n_steps=12)

    with DatasetReader(path) as reader:
        assert [s.index for s in reader.steps("0")] == list(range(12))


def test_conversion_preserves_von_mises(dataset):
    """Re-expressing in another convention must not change an invariant.

    This is the check that catches a wrong slot order or a missing sqrt(2):
    both leave the array looking reasonable and the invariant wrong.
    """
    with DatasetReader(dataset) as reader:
        step = reader.step("0", 1)

    np.testing.assert_allclose(
        von_mises(step.stress, DATASET_CONVENTION),
        von_mises(step.stress_in(MANDEL_XY_LAST), MANDEL_XY_LAST),
        rtol=1e-12,
    )


def test_stress_scale_can_be_overridden(dataset):
    """A file already written in MPa needs no conversion."""
    with DatasetReader(dataset, stress_scale=1.0) as reader:
        step = reader.step("0", 0)

    np.testing.assert_allclose(step.stress[0, 0], 0.0)
    np.testing.assert_allclose(step.stress[0, 1], 1.0e6)


def test_missing_averages_fall_back_to_zeros(tmp_path):
    """A convenience row is not worth refusing to read the file over."""
    path = write_dataset(tmp_path / "noavg.hdf5", geometry_ids=("0",), n_steps=1,
                         omit=("results_avg", "eps_as_loading"))

    with DatasetReader(path) as reader:
        step = reader.step("0", 0)

    np.testing.assert_allclose(step.stress_avg, 0.0)
    np.testing.assert_allclose(step.eps_as_loading, 0.0)


# --- Refusals ----------------------------------------------------------------


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset not found"):
        DatasetReader(tmp_path / "absent.hdf5").open()


def test_file_without_geometries_is_reported(tmp_path):
    path = tmp_path / "empty.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("last_set", data=np.int32(0))

    with pytest.raises(DatasetError, match="no geometry groups"):
        DatasetReader(path).open()


def test_missing_results_is_reported(tmp_path):
    path = write_dataset(tmp_path / "bad.hdf5", geometry_ids=("0",), n_steps=1,
                         omit=("results",))

    with DatasetReader(path) as reader, pytest.raises(DatasetError, match="no 'results'"):
        reader.step("0", 0)


def test_wrong_column_layout_is_reported(tmp_path):
    path = write_dataset(tmp_path / "narrow.hdf5", geometry_ids=("0",), n_steps=1, columns=7)

    with DatasetReader(path) as reader, pytest.raises(DatasetError, match="column layout"):
        reader.step("0", 0)


def test_unknown_geometry_lists_what_there_is(dataset):
    with DatasetReader(dataset) as reader, pytest.raises(KeyError, match=r"\['0', '1'\]"):
        reader.geometry("7")


def test_step_out_of_range_is_reported(dataset):
    with DatasetReader(dataset) as reader, pytest.raises(IndexError, match="has 3 steps"):
        reader.step("0", 9)


def test_closed_reader_says_so(dataset):
    reader = DatasetReader(dataset)
    with reader:
        pass

    with pytest.raises(DatasetError, match="is closed"):
        reader.geometry_ids


def test_close_is_idempotent(dataset):
    reader = DatasetReader(dataset).open()
    reader.close()
    reader.close()


# --- Reference values --------------------------------------------------------


def test_ground_truth_is_available_when_present(dataset):
    with DatasetReader(dataset) as reader:
        assert reader.has_ground_truth
        assert set(reader.ground_truth("0")) == {"yield_sigma", "k_factor"}


def test_ground_truth_absence_is_explained(tmp_path):
    path = write_dataset(tmp_path / "plain.hdf5", geometry_ids=("0",), n_steps=1)

    with DatasetReader(path) as reader:
        assert not reader.has_ground_truth
        with pytest.raises(KeyError, match="only generated datasets"):
            reader.ground_truth("0")


def test_raw_reads_what_the_reader_does_not_model(dataset):
    with DatasetReader(dataset) as reader:
        row = reader.raw("0", "ls_0", "results_avg")

    assert row.shape == (19,)


# --- Material ----------------------------------------------------------------


def test_crss_comes_from_the_file_in_mpa(dataset):
    with DatasetReader(dataset) as reader:
        material = read_material(reader)

    assert material.crss == pytest.approx(150.0)
    assert material.provenance["crss"] == "dataset attribute 'crss'"


def test_crss_falls_back_to_gen_config(tmp_path):
    path = write_dataset(
        tmp_path / "gc.hdf5",
        geometry_ids=("0",),
        n_steps=1,
        attrs={"gen_config": json.dumps({"crss": 200.0e6})},
    )
    with DatasetReader(path) as reader:
        material = read_material(reader)

    assert material.crss == pytest.approx(200.0)
    assert material.provenance["crss"] == "dataset attribute 'gen_config'"


def test_override_wins_over_the_file_and_says_so(dataset):
    """A value in a config is deliberate, and must be able to correct a file."""
    with DatasetReader(dataset) as reader:
        material = read_material(reader, {"crss": 275.0})

    assert material.crss == pytest.approx(275.0)
    assert material.provenance["crss"] == "case config material_override"


def test_elastic_constants_are_reported_in_gpa(dataset):
    with DatasetReader(dataset) as reader:
        material = read_material(reader)

    assert material.elastic["c11"] == pytest.approx(168.4)
    assert material.elastic["c44"] == pytest.approx(75.4)


def test_exported_dataset_carries_no_material(tmp_path):
    """The C++ writer stores no attributes; saying so beats inventing a value."""
    path = write_dataset(tmp_path / "bare.hdf5", geometry_ids=("1",), n_steps=1)

    with DatasetReader(path) as reader:
        material = read_material(reader)

    assert material.crss is None
    with pytest.raises(ConfigError, match="material_override.crss"):
        material.require_crss()


def test_a_crss_left_in_pascals_is_refused(dataset):
    """Copying 150000000 out of the file into the config must not pass."""
    with DatasetReader(dataset) as reader, pytest.raises(ConfigError, match="Pa/MPa mix-up"):
        read_material(reader, {"crss": 150.0e6})


def test_unknown_override_key_is_refused(dataset):
    with DatasetReader(dataset) as reader, pytest.raises(ConfigError, match="does not accept"):
        read_material(reader, {"crrs": 150.0})


def test_structure_defaults_and_records_that_it_did(dataset):
    with DatasetReader(dataset) as reader:
        material = read_material(reader)

    assert material.structure == DEFAULT_STRUCTURE
    assert material.provenance["structure"] == "package default"


def test_unknown_structure_is_refused(dataset):
    with DatasetReader(dataset) as reader, pytest.raises(ConfigError, match="known structures"):
        read_material(reader, {"structure": "BBC"})


def test_missing_sigma_y_is_explained(dataset):
    with DatasetReader(dataset) as reader:
        material = read_material(reader)

    with pytest.raises(ConfigError, match="von Mises"):
        material.require_sigma_y()


def test_description_carries_provenance(dataset):
    with DatasetReader(dataset) as reader:
        text = read_material(reader, {"structure": "FCC"}).describe()

    assert "dataset attribute 'crss'" in text
    assert "case config material_override" in text


def test_material_serialises_for_the_report(dataset):
    with DatasetReader(dataset) as reader:
        payload = read_material(reader).to_dict()

    assert payload["crss"] == pytest.approx(150.0)
    assert payload["metadata"]["noise_level"] == "clean"
    assert "crss" in payload["provenance"]


def test_material_defaults_stand_alone():
    """A bare MaterialProperties is usable, and honest about knowing nothing."""
    material = MaterialProperties()

    assert material.structure == DEFAULT_STRUCTURE
    assert material.to_dict()["crss"] is None


# --- Against a real dataset --------------------------------------------------


@pytest.mark.skipif(not REAL_DATASET.exists(), reason=f"{REAL_DATASET.name} not present")
def test_generated_dataset_matches_its_recorded_summary():
    """Reproduce the numbers ``datasets_synth/info.txt`` records.

    The file's own ``yield_sigma_mandel`` is read back through this
    package's convention machinery and compared against the raw components:
    an independent check that the dataset convention is what this reader
    claims it is.
    """
    with DatasetReader(REAL_DATASET) as reader:
        assert reader.n_geometries == 40
        assert reader.n_steps("0") == 120
        assert read_material(reader).crss == pytest.approx(150.0)

        equivalent = np.concatenate([
            von_mises(reader.ground_truth(g)["yield_sigma"] * 1.0e-6, DATASET_CONVENTION)
            for g in reader.geometry_ids
        ])

        truth = reader.ground_truth("0")
        raw = von_mises(truth["yield_sigma"] * 1.0e-6, DATASET_CONVENTION)
        mandel = von_mises(truth["yield_sigma_mandel"] * 1.0e-6, MANDEL_XY_LAST)

    assert equivalent.size == 4800
    assert equivalent.mean() == pytest.approx(268.66, abs=0.01)
    assert equivalent.std() == pytest.approx(6.72, abs=0.01)
    np.testing.assert_allclose(raw, mandel, rtol=1e-10)
