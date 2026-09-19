"""Visualisation infrastructure: registry, output, reductions, sampling."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mvyield.mechanics.field import FieldMeta, GeometryHints, StressField, Topology
from mvyield.mechanics.voigt import MANDEL_XY_LAST
from mvyield.paths import RunPaths
from mvyield.settings import PlotConfig
from mvyield.viz import labels, reduce, registry, render, sample
from mvyield.viz.io import export_vtu, save_figure
from mvyield.viz.registry import Space, plot, resolve_preset, select
from mvyield.viz.theme import THEME, apply_rcparams


@pytest.fixture(autouse=True)
def clean_registry():
    """Keep registrations from leaking between tests."""
    saved = dict(registry.REGISTRY)
    registry.REGISTRY.clear()
    yield
    registry.REGISTRY.clear()
    registry.REGISTRY.update(saved)


def make_field(topology=Topology.CLOUD2D, n=64, hints=None, grid_shape=None):
    """Build a small field of the requested topology."""
    if topology is Topology.GRID2D:
        side = int(np.sqrt(n))
        gx, gy = np.meshgrid(np.linspace(-2, 2, side), np.linspace(-2, 2, side))
        points = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
        grid_shape = (side, side)
    elif topology.is_3d:
        rng = np.random.default_rng(0)
        points = rng.uniform(-2, 2, (n, 3))
    else:
        rng = np.random.default_rng(0)
        points = np.column_stack([rng.uniform(-2, 2, (n, 2)), np.zeros(n)])

    sigma6 = np.zeros((len(points), 6))
    sigma6[:, 0] = 150.0 + 30.0 * points[:, 0]

    return StressField(
        points=points,
        sigma6=sigma6,
        topology=topology,
        meta=FieldMeta(source="analytic", convention=MANDEL_XY_LAST, grid_shape=grid_shape),
        hints=hints or GeometryHints(),
    )


# --- Registry ----------------------------------------------------------------


def test_registration_records_conditions():
    @plot(
        "demo",
        space=Space.FIELD,
        groups=["paper"],
        needs=["kde_ray"],
        description="A demo figure",
    )
    def _draw(result, ax, theme):
        pass

    spec = registry.REGISTRY["demo"]
    assert spec.space is Space.FIELD
    assert spec.needs == ("kde_ray",)
    assert spec.description == "A demo figure"


def test_duplicate_registration_is_refused():
    @plot("demo", space=Space.FIELD)
    def _first(result, ax, theme):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @plot("demo", space=Space.FIELD)
        def _second(result, ax, theme):
            pass


def test_description_falls_back_to_the_docstring():
    @plot("demo", space=Space.AGGREGATE)
    def _draw(result, ax, theme):
        """Yield probability histogram."""

    assert registry.REGISTRY["demo"].description == "Yield probability histogram."


def test_field_plots_default_to_per_reduction():
    """A 3D field must be reduced before a field map can be drawn."""

    @plot("a_field", space=Space.FIELD)
    def _field(result, ax, theme):
        pass

    @plot("a_histogram", space=Space.AGGREGATE)
    def _aggregate(result, ax, theme):
        pass

    assert registry.REGISTRY["a_field"].per_reduction
    assert not registry.REGISTRY["a_histogram"].per_reduction


def test_missing_method_skips_a_plot():
    """A figure needing the slice must be skipped, not crash, without it."""

    @plot("needs_slice", space=Space.FIELD, needs=["kde_slice"])
    def _draw(result, ax, theme):
        pass

    selected, skipped = select(["needs_slice"], methods=("kde_ray",))
    assert selected == []
    assert "kde_slice" in skipped["needs_slice"]


def test_incompatible_topology_skips_a_plot():
    """A grid-only figure must step aside for a wire rather than fail."""

    @plot("grid_only", space=Space.FIELD, topology=[Topology.GRID2D])
    def _draw(result, ax, theme):
        pass

    selected, skipped = select(["grid_only"], methods=("kde_ray",), topology=Topology.CLOUD3D)
    assert selected == []
    assert "cloud3d" in skipped["grid_only"]


def test_missing_diagnostic_skips_a_plot():
    @plot("needs_eff_n", space=Space.FIELD, needs_diagnostics=["eff_n"])
    def _draw(result, ax, theme):
        pass

    selected, _ = select(["needs_eff_n"], methods=("kde_ray",), diagnostics=())
    assert selected == []

    selected, _ = select(["needs_eff_n"], methods=("kde_ray",), diagnostics=("eff_n",))
    assert len(selected) == 1


def test_material_plots_apply_to_every_problem():
    """Material figures describe the material, so no geometry can exclude them.

    This is why the dataset-analysis figures need no porting: they were
    never problem-specific, only misfiled.
    """

    @plot("locus", space=Space.MATERIAL, groups=["paper"])
    def _draw(result, ax, theme):
        pass

    for topology in Topology:
        selected, _ = select(["locus"], methods=("kde_ray",), topology=topology)
        assert len(selected) == 1


def test_unregistered_name_is_skipped_not_raised():
    """Presets may name figures a later package will add."""
    selected, skipped = select(["not_yet_written"], methods=("kde_ray",))
    assert selected == []
    assert skipped["not_yet_written"] == "not registered"


# --- Presets -----------------------------------------------------------------


def test_preset_resolves_to_its_list():
    presets = {"paper": ["a", "b"], "minimal": ["a"]}
    assert resolve_preset("paper", presets) == ["a", "b"]


def test_all_preset_returns_the_whole_registry():
    @plot("one", space=Space.FIELD)
    def _one(result, ax, theme):
        pass

    @plot("two", space=Space.FIELD)
    def _two(result, ax, theme):
        pass

    assert set(resolve_preset("all", {})) == {"one", "two"}


def test_include_and_exclude_adjust_a_preset():
    for name in ("a", "b", "c"):
        plot(name, space=Space.FIELD)(lambda result, ax, theme: None)

    presets = {"paper": ["a", "b"]}
    assert resolve_preset("paper", presets, include=("c",), exclude=("a",)) == ["b", "c"]


def test_unknown_preset_is_reported():
    with pytest.raises(KeyError, match="unknown plot preset"):
        resolve_preset("typo", {"paper": []})


def test_unknown_plot_name_in_include_is_reported():
    """A typo in a config must not silently produce no figure."""
    with pytest.raises(KeyError, match="unknown plot"):
        resolve_preset("paper", {"paper": []}, include=("mispelled",))


# --- Output ------------------------------------------------------------------


def test_png_and_svg_are_both_written(tmp_path):
    paths = RunPaths.for_run("t", repo_root=tmp_path).prepare()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    written = save_figure(fig, "demo", paths, PlotConfig())
    assert {p.suffix for p in written} == {".png", ".svg"}
    assert all(p.exists() for p in written)


def test_png_resolution_follows_the_config(tmp_path):
    """The publication floor must reach the file, not just the schema."""
    from PIL import Image

    paths = RunPaths.for_run("t", repo_root=tmp_path).prepare()
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])

    written = save_figure(fig, "demo", paths, PlotConfig(formats=("png",), dpi=300))
    with Image.open(written[0]) as image:
        assert image.info["dpi"][0] == pytest.approx(300, abs=1)


def test_svg_keeps_text_editable(tmp_path):
    """Outlined glyphs cannot be edited at proof stage or searched in a PDF."""
    apply_rcparams()
    paths = RunPaths.for_run("t", repo_root=tmp_path).prepare()

    fig, ax = plt.subplots()
    ax.set_xlabel("Yield probability")

    written = save_figure(fig, "demo", paths, PlotConfig(formats=("svg",), dpi=300))
    content = written[0].read_text(encoding="utf-8")
    assert "Yield probability" in content


def test_reduction_suffix_appears_in_the_filename(tmp_path):
    paths = RunPaths.for_run("t", repo_root=tmp_path).prepare()
    fig, _ = plt.subplots()

    written = save_figure(
        fig, "kde_probability", paths, PlotConfig(formats=("svg",)), suffix="max along z"
    )
    assert written[0].name == "kde_probability__max_along_z.svg"


def test_vtk_export_names_the_install_command(tmp_path):
    """A missing optional dependency must say how to get it.

    Real 3D inspection belongs in ParaView, so this path matters for the
    wire case; failing with a bare ImportError would leave the user guessing.
    """
    try:
        import meshio as _meshio  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"\[vtk\]"):
            export_vtu(np.zeros((4, 3)), {"p": np.zeros(4)}, tmp_path / "f.vtu")
    else:
        written = export_vtu(np.zeros((4, 3)), {"p": np.zeros(4)}, tmp_path / "f.vtu")
        assert written.exists()


def test_vtk_export_checks_field_lengths(tmp_path):
    """A length mismatch must be caught before meshio is reached."""
    with pytest.raises((ValueError, ImportError)):
        export_vtu(np.zeros((10, 3)), {"p": np.zeros(4)}, tmp_path / "f.vtu")


# --- Reductions --------------------------------------------------------------


def test_two_dimensional_field_passes_through():
    field = make_field()
    views = reduce.apply_reductions(field, {"a/probability": np.zeros(field.n_points)}, ())

    assert len(views) == 1
    assert views[0].name == ""


def test_three_dimensional_field_without_reductions_is_refused():
    """Silence here would mean a wire run producing no field maps at all."""
    field = make_field(Topology.CLOUD3D)
    with pytest.raises(ValueError, match=r"no view\.reductions"):
        reduce.apply_reductions(field, {}, ())


def test_slice_selects_points_near_a_plane():
    field = make_field(Topology.CLOUD3D, n=200)
    field.points[:50, 2] = 0.0
    scalars = {"a/probability": np.linspace(0, 1, field.n_points)}

    views = reduce.apply_reductions(
        field, scalars, ({"kind": "slice", "plane": "z", "value": 0.0, "tol": 1e-9},)
    )
    assert views[0].field.n_points == 50
    assert not views[0].field.topology.is_3d


def test_empty_slice_explains_the_coordinate_range():
    field = make_field(Topology.CLOUD3D)
    with pytest.raises(ValueError, match="range"):
        reduce.apply_reductions(
            field, {}, ({"kind": "slice", "plane": "z", "value": 99.0, "tol": 1e-6},)
        )


def test_max_projection_takes_the_envelope():
    """One picture answering 'worst value anywhere along the axis'."""
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [1.0, 0.0, 0.0]])
    field = StressField(
        points=points,
        sigma6=np.zeros((4, 6)),
        topology=Topology.CLOUD3D,
        meta=FieldMeta(source="ansys", convention=MANDEL_XY_LAST),
    )
    scalars = {"a/probability": np.array([0.1, 0.9, 0.4, 0.2])}

    views = reduce.apply_reductions(
        field, scalars, ({"kind": "projection", "axis": "z", "reduce": "max"},)
    )
    result = views[0]

    assert result.field.n_points == 2
    assert result.scalars["a/probability"].max() == pytest.approx(0.9)


def test_projection_handles_nan_and_boolean_fields():
    """Coverage is boolean: a mean would silently turn it into a fraction."""
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    field = StressField(
        points=points,
        sigma6=np.zeros((2, 6)),
        topology=Topology.CLOUD3D,
        meta=FieldMeta(source="ansys", convention=MANDEL_XY_LAST),
    )
    scalars = {
        "a/probability": np.array([np.nan, 0.7]),
        "a/covered": np.array([False, True]),
    }

    views = reduce.apply_reductions(
        field, scalars, ({"kind": "projection", "axis": "z", "reduce": "max"},)
    )
    assert views[0].scalars["a/probability"][0] == pytest.approx(0.7)
    assert views[0].scalars["a/covered"].dtype == bool
    assert views[0].scalars["a/covered"][0]


def test_unroll_maps_to_radius_and_axis():
    field = make_field(Topology.CLOUD3D, n=50)
    views = reduce.apply_reductions(
        field, {}, ({"kind": "unroll", "axis": "z", "name": "rz"},)
    )
    assert (views[0].field.x >= 0).all()


def test_each_reduction_gets_its_own_suffix():
    field = make_field(Topology.CLOUD3D, n=100)
    field.points[:30, 2] = 0.0

    views = reduce.apply_reductions(
        field,
        {"a/probability": np.zeros(field.n_points)},
        (
            {"kind": "slice", "plane": "z", "value": 0.0, "tol": 1e-9, "name": "midspan"},
            {"kind": "projection", "axis": "z", "reduce": "max", "name": "max_along_z"},
        ),
    )
    assert [v.name for v in views] == ["midspan", "max_along_z"]


def test_unknown_reduction_kind_is_reported():
    field = make_field(Topology.CLOUD3D)
    with pytest.raises(ValueError, match="unknown reduction kind"):
        reduce.apply_reductions(field, {}, ({"kind": "fold"},))


# --- Sampling ----------------------------------------------------------------


def test_line_path_spans_its_endpoints():
    path = sample.build_path({"kind": "line", "from": [0, 0, 0], "to": [1, 0, 0], "n": 11})
    assert path.shape == (11, 3)
    assert path[-1, 0] == pytest.approx(1.0)


def test_arc_path_keeps_a_constant_radius():
    path = sample.build_path(
        {"kind": "arc", "center": [0, 0, 0], "radius": 2.0, "plane": "xy", "n": 36}
    )
    np.testing.assert_allclose(np.hypot(path[:, 0], path[:, 1]), 2.0, rtol=1e-12)


def test_ray_path_converts_reference_length_units():
    """The Kirsch ligament profile, expressed as config rather than as code."""
    spec = {
        "kind": "ray",
        "angle_deg": 90.0,
        "r_from": 1.0,
        "r_to": 4.0,
        "units": "hole_radii",
        "n": 50,
    }
    path = sample.build_path(spec, length_scale=2.5)

    assert np.hypot(path[0, 0], path[0, 1]) == pytest.approx(2.5)
    assert np.hypot(path[-1, 0], path[-1, 1]) == pytest.approx(10.0)


def test_reference_length_units_need_a_reference_length():
    spec = {"kind": "ray", "name": "ligament", "r_to": 4.0, "units": "hole_radii"}
    with pytest.raises(ValueError, match="no reference length"):
        sample.build_path(spec, length_scale=None)


def test_sampling_interpolates_a_known_field():
    field = make_field(n=400)
    values = 2.0 * field.x + 1.0
    path = sample.build_path({"kind": "line", "from": [-1, 0, 0], "to": [1, 0, 0], "n": 21})

    sampled = sample.sample_along(field, values, path)
    expected = 2.0 * path[:, 0] + 1.0
    finite = np.isfinite(sampled)

    assert finite.sum() > 15
    np.testing.assert_allclose(sampled[finite], expected[finite], atol=1e-6)


def test_sampling_outside_the_field_returns_nan():
    field = make_field(n=200)
    path = sample.build_path({"kind": "line", "from": [50, 50, 0], "to": [60, 60, 0], "n": 5})
    assert np.isnan(sample.sample_along(field, np.ones(field.n_points), path)).all()


def test_argmax_probe_finds_the_hotspot():
    """The critical point must be found, not assumed.

    On a plate it sits at a known angle; on a wire it does not, so a
    hard-coded location does not transfer.
    """
    field = make_field(n=100)
    values = np.zeros(field.n_points)
    values[42] = 1.0

    index = sample.locate_point({"kind": "argmax", "of": "P"}, field, {"P": values})
    assert index == 42


def test_probe_reports_an_unknown_field():
    field = make_field()
    with pytest.raises(KeyError, match="not computed"):
        sample.locate_point({"kind": "argmax", "of": "P_missing"}, field, {"P": np.zeros(64)})


def test_stress_sampling_defaults_to_nearest():
    """Interpolated stress states never occurred anywhere in the body."""
    field = make_field(n=200)
    path = sample.build_path({"kind": "line", "from": [-1, 0, 0], "to": [1, 0, 0], "n": 10})

    sampled = sample.sample_stress_along(field, path)
    for row in sampled:
        assert np.isclose(field.sigma6, row).all(axis=1).any()


# --- Rendering ---------------------------------------------------------------


@pytest.mark.parametrize("topology", [Topology.GRID2D, Topology.CLOUD2D])
def test_scalar_rendering_works_for_both_2d_topologies(topology):
    field = make_field(topology, n=100)
    fig, ax = plt.subplots()

    mappable = render.render_scalar(
        ax, field, np.linspace(0, 1, field.n_points), label="P", fig=fig
    )
    assert mappable is not None
    plt.close(fig)


def test_rendering_a_3d_field_directly_is_refused():
    """Reaching the renderer with a 3D field is a routing error."""
    field = make_field(Topology.CLOUD3D)
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="Reduce it"):
        render.render_scalar(ax, field, np.zeros(field.n_points), fig=fig)
    plt.close(fig)


def test_rendering_checks_value_length():
    field = make_field()
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="entries for a field"):
        render.render_scalar(ax, field, np.zeros(3), fig=fig)
    plt.close(fig)


def test_axes_normalise_only_when_a_reference_length_exists():
    """A case supplying no reference length must still plot, in mm."""
    plain = make_field()
    scaled = make_field(hints=GeometryHints(length_scale=2.0, length_symbol="a"))

    fig, ax = plt.subplots()
    render.render_scalar(ax, plain, np.zeros(plain.n_points), fig=fig)
    assert "mm" in ax.get_xlabel()
    plt.close(fig)

    fig, ax = plt.subplots()
    render.render_scalar(ax, scaled, np.zeros(scaled.n_points), fig=fig)
    assert "/a" in ax.get_xlabel()
    plt.close(fig)


def test_hole_outline_masks_the_triangulation():
    """Without masking, the plot colours a disc where there is no material."""
    rng = np.random.default_rng(1)
    angle = rng.uniform(0, 2 * np.pi, 300)
    radius = rng.uniform(1.0, 3.0, 300)
    points = np.column_stack([radius * np.cos(angle), radius * np.sin(angle), np.zeros(300)])

    field = StressField(
        points=points,
        sigma6=np.tile([150.0, 0, 0, 0, 0, 0], (300, 1)),
        topology=Topology.CLOUD2D,
        meta=FieldMeta(source="analytic", convention=MANDEL_XY_LAST),
        hints=GeometryHints(outlines=[render.circle_outline(1.0)]),
    )

    triangulation = render._triangulate(field.x, field.y, field.hints)
    assert triangulation.mask is not None and triangulation.mask.any()


# --- Labels ------------------------------------------------------------------


def test_direction_keys_are_ascii():
    """Language must not live inside dictionary keys.

    The legacy tables used Cyrillic strings as keys, so translating the
    labels would have broken every lookup.
    """
    for key in labels.DIRECTION_LABELS:
        assert key.isascii() and key.islower()


def test_every_figure_string_is_ascii():
    for text in (
        *labels.LABELS.values(),
        *labels.METHOD_LABELS.values(),
        *labels.DIRECTION_LABELS.values(),
    ):
        assert text.isascii(), f"non-ASCII label: {text!r}"


def test_ray_variants_are_distinguishable():
    """Two modes of one method still need distinct names on a shared figure."""
    assert labels.method_label("kde_ray", localized=True) != labels.method_label(
        "kde_ray", localized=False
    )


def test_unknown_label_passes_through():
    assert labels.label("a_one_off_label") == "a_one_off_label"


def test_provenance_note_states_the_source():
    note = labels.provenance_note("kappa", 20.0, "cv_loglik")
    assert "20" in note and "cross-validated" in note


def test_methods_keep_their_colour_across_figures():
    assert THEME.method_colour("kde_slice") != THEME.method_colour("kde_ray")
