"""Cases: the analytical Kirsch field and the ANSYS import."""

from __future__ import annotations

import numpy as np
import pytest

from mvyield.cases import available_cases, build_field
from mvyield.cases.ansys import read_table
from mvyield.cases.kirsch import (
    Domain,
    kirsch_polar,
    polar_to_cartesian,
    resolve_domain,
)
from mvyield.mechanics.field import Topology
from mvyield.mechanics.voigt import MANDEL_XY_FIRST, MANDEL_XY_LAST, von_mises
from mvyield.settings import ConfigError, FieldConfig

SIGMA_0 = 150.0


def kirsch_config(**overrides) -> FieldConfig:
    """A Kirsch field config sized in hole radii, needing no dataset."""
    settings = {
        "source": "analytic",
        "sigma_applied": SIGMA_0,
        "domain_from": "config",
        "hole_radius": {"auto": False, "fraction": 0.15, "fallback": 1.0},
        "grid": {"resolution": 81},
    }
    settings.update(overrides)
    return FieldConfig(**settings)


# --- Registry ----------------------------------------------------------------


def test_both_cases_are_available():
    assert available_cases() == ("ansys", "kirsch")


def test_unknown_case_lists_the_known_ones():
    with pytest.raises(ConfigError, match="available"):
        build_field("plate_with_notch", kirsch_config())


# --- The Kirsch solution -----------------------------------------------------


def test_stress_concentration_is_three_at_the_hole_edge():
    """The whole reason the problem is worth solving.

    On the transverse axis at r = a the hoop stress is 3 sigma_0, so a
    remote load a third of the yield strength already yields at the hole.
    """
    _, hoop, shear = kirsch_polar(
        np.array([1.0]), np.array([np.pi / 2]), hole_radius=1.0, sigma_applied=SIGMA_0
    )

    assert hoop[0] == pytest.approx(3.0 * SIGMA_0)
    assert shear[0] == pytest.approx(0.0, abs=1e-12)


def test_hole_edge_is_traction_free():
    """Nothing pushes on a free surface, at any angle."""
    theta = np.linspace(0.0, 2 * np.pi, 37)
    radial, _, shear = kirsch_polar(np.ones_like(theta), theta, 1.0, SIGMA_0)

    np.testing.assert_allclose(radial, 0.0, atol=1e-12)
    np.testing.assert_allclose(shear, 0.0, atol=1e-12)


def test_edge_stress_is_compressive_along_the_load_axis():
    """The classical -sigma_0 at theta = 0, the other half of the answer."""
    _, hoop, _ = kirsch_polar(np.array([1.0]), np.array([0.0]), 1.0, SIGMA_0)

    assert hoop[0] == pytest.approx(-SIGMA_0)


def test_remote_field_is_recovered_far_from_the_hole():
    """Far away the hole is invisible: uniaxial tension along x."""
    theta = np.linspace(0.0, 2 * np.pi, 24)
    radius = np.full_like(theta, 400.0)

    sigma_xx, sigma_yy, tau_xy = polar_to_cartesian(
        *kirsch_polar(radius, theta, 1.0, SIGMA_0), theta
    )

    np.testing.assert_allclose(sigma_xx, SIGMA_0, rtol=2e-5)
    np.testing.assert_allclose(sigma_yy, 0.0, atol=SIGMA_0 * 2e-5)
    np.testing.assert_allclose(tau_xy, 0.0, atol=SIGMA_0 * 2e-5)


def test_solution_is_scale_free():
    """Doubling plate and hole together changes nothing."""
    theta = np.linspace(0.0, 2 * np.pi, 16)
    small = kirsch_polar(np.full_like(theta, 2.5), theta, 1.0, SIGMA_0)
    large = kirsch_polar(np.full_like(theta, 5.0), theta, 2.0, SIGMA_0)

    for one, other in zip(small, large):
        np.testing.assert_allclose(one, other, atol=1e-12)


def test_rotation_preserves_von_mises():
    """Polar to Cartesian is a rotation, so an invariant must not move."""
    theta = np.linspace(0.1, 2 * np.pi, 20)
    radial, hoop, shear = kirsch_polar(np.full_like(theta, 1.7), theta, 1.0, SIGMA_0)
    sigma_xx, sigma_yy, tau_xy = polar_to_cartesian(radial, hoop, shear, theta)

    polar_equivalent = np.sqrt(radial**2 + hoop**2 - radial * hoop + 3 * shear**2)
    cartesian_equivalent = np.sqrt(
        sigma_xx**2 + sigma_yy**2 - sigma_xx * sigma_yy + 3 * tau_xy**2
    )
    np.testing.assert_allclose(polar_equivalent, cartesian_equivalent, rtol=1e-12)


# --- The field -------------------------------------------------------------


def test_field_is_a_regular_grid():
    field = build_field("kirsch", kirsch_config())

    assert field.topology is Topology.GRID2D
    assert field.meta.grid_shape == (81, 81)
    assert field.n_points == 81 * 81
    np.testing.assert_allclose(field.z, 0.0)


def test_points_inside_the_hole_are_marked_not_removed():
    """The grid must stay rectangular so field maps can use imshow."""
    field = build_field("kirsch", kirsch_config())
    inside = ~field.valid_mask()

    assert inside.any()
    assert field.n_points == 81 * 81
    np.testing.assert_allclose(field.sigma6[inside], 0.0)
    assert np.hypot(field.x, field.y)[inside].max() < 1.0


def test_field_is_plane_stress():
    """zz, yz and xz are zero, and stay zero whatever the slot order."""
    field = build_field("kirsch", kirsch_config())
    convention = field.meta.convention

    for label in ("ZZ", "YZ", "XZ"):
        np.testing.assert_allclose(field.sigma6[:, convention.slot_of(label)], 0.0, atol=1e-12)


def test_peak_equivalent_stress_is_near_three_sigma():
    field = build_field("kirsch", kirsch_config(grid={"resolution": 401}))
    equivalent = von_mises(field.sigma6, field.meta.convention)[field.valid_mask()]

    assert equivalent.max() == pytest.approx(3.0 * SIGMA_0, rel=0.02)


def test_hints_carry_the_hole_for_the_figures():
    """A plotting function should not need a hole_radius argument."""
    field = build_field("kirsch", kirsch_config())

    assert field.hints.length_scale == pytest.approx(1.0)
    assert field.hints.length_symbol == "a"
    assert len(field.hints.outlines) == 1
    assert field.hints.outlines[0].kind == "circle"
    assert field.hints.outlines[0].radius == pytest.approx(1.0)


def test_metadata_records_how_the_field_was_made():
    field = build_field("kirsch", kirsch_config())

    assert field.meta.source == "analytic"
    assert field.meta.extra["case"] == "kirsch"
    assert field.meta.extra["sigma_applied"] == SIGMA_0
    assert field.meta.extra["domain_source"] == "config"


def test_field_honours_the_declared_convention():
    """The same physics, two slot orders, one invariant."""
    last = build_field("kirsch", kirsch_config(convention="mandel_xy_last"))
    first = build_field("kirsch", kirsch_config(convention="mandel_xy_first"))

    np.testing.assert_allclose(
        von_mises(last.sigma6, MANDEL_XY_LAST),
        von_mises(first.sigma6, MANDEL_XY_FIRST),
        rtol=1e-12,
    )
    assert not np.allclose(last.sigma6, first.sigma6)


# --- Sizing the window -------------------------------------------------------


def test_dataset_extent_sizes_the_hole():
    config = kirsch_config(domain_from="dataset",
                           hole_radius={"auto": True, "fraction": 0.15})
    domain = resolve_domain(config, extent=14.0)

    assert domain.hole_radius == pytest.approx(0.15 * 14.0)
    assert domain.source == "dataset extent"


def test_dataset_sizing_without_a_dataset_says_what_to_do():
    config = kirsch_config(domain_from="dataset")

    with pytest.raises(ConfigError, match="domain_from"):
        resolve_domain(config, extent=None)


def test_manual_radius_overrides_the_fraction():
    config = kirsch_config(domain_from="dataset",
                           hole_radius={"auto": False, "fallback": 2.0})
    domain = resolve_domain(config, extent=14.0)

    assert domain.hole_radius == pytest.approx(2.0)


def test_window_defaults_to_four_radii():
    domain = resolve_domain(kirsch_config())

    assert domain.half_width_in_radii == pytest.approx(4.0)


def test_unknown_domain_source_is_refused():
    with pytest.raises(ConfigError, match="domain_from"):
        resolve_domain(kirsch_config(domain_from="guess"))


def test_a_hole_larger_than_the_plate_is_refused():
    with pytest.raises(ConfigError, match="does not contain the hole"):
        Domain(half_width=1.0, hole_radius=2.0)


def test_degenerate_resolution_is_refused():
    with pytest.raises(ConfigError, match="resolution"):
        build_field("kirsch", kirsch_config(grid={"resolution": 1}))


# --- The ANSYS import --------------------------------------------------------


PRNSOL = """\
 PRINT S    NODAL SOLUTION PER NODE

  ***** POST1 NODAL STRESS LISTING *****

  NODE       X       Y       Z          SX          SY          SZ         SXY         SYZ         SXZ
     1     0.0     0.0     0.0   1.000E+08   0.000E+00   0.000E+00   0.000E+00   0.000E+00   0.000E+00
     2     1.0     0.0     0.0   2.000E+08   1.000E+08   0.000E+00   5.000E+07   0.000E+00   0.000E+00

  ***** POST1 NODAL STRESS LISTING *****

  NODE       X       Y       Z          SX          SY          SZ         SXY         SYZ         SXZ
     3     2.0     1.0     0.0   0.000E+00   0.000E+00   0.000E+00   0.000E+00   1.000E+08   0.000E+00
"""


@pytest.fixture
def export(tmp_path):
    path = tmp_path / "wire_nodal_stress.csv"
    path.write_text(PRNSOL, encoding="utf-8")
    return path


def ansys_config(path, **overrides) -> FieldConfig:
    settings = {
        "source": "ansys",
        "path": path,
        "format": "prnsol_csv",
        "convention": "mandel_xy_first",
        "units": {"stress": "Pa", "length": "m"},
    }
    settings.update(overrides)
    return FieldConfig(**settings)


def test_paginated_export_is_read_whole(export):
    """PRNSOL repeats its header between banner lines; all rows must land."""
    table = read_table(export, ["X", "Y", "SX"])

    assert len(table["X"]) == 3
    np.testing.assert_allclose(table["SX"], [1e8, 2e8, 0.0])


def test_import_converts_units(export):
    """Pa and metres in, MPa and millimetres out."""
    field = build_field("ansys", ansys_config(export))

    assert field.meta.stress_unit == "MPa"
    assert field.sigma6[0, field.meta.convention.slot_of("XX")] == pytest.approx(100.0)
    np.testing.assert_allclose(field.points[1], [1000.0, 0.0, 0.0])


def test_mandel_factor_is_applied_exactly_once(export):
    """ANSYS writes engineering shear; the model space is Mandel-scaled."""
    field = build_field("ansys", ansys_config(export))
    convention = field.meta.convention

    assert convention.mandel
    # Node 2 has SXY = 5e7 Pa = 50 MPa engineering.
    assert field.sigma6[1, convention.slot_of("XY")] == pytest.approx(50.0 * np.sqrt(2.0))
    # Node 3 has SYZ = 1e8 Pa = 100 MPa engineering.
    assert field.sigma6[2, convention.slot_of("YZ")] == pytest.approx(100.0 * np.sqrt(2.0))


def test_shear_order_follows_the_declared_convention(export):
    """The export's SXY column must not land in a YZ slot."""
    first = build_field("ansys", ansys_config(export, convention="mandel_xy_first"))
    last = build_field("ansys", ansys_config(export, convention="mandel_xy_last"))

    np.testing.assert_allclose(
        von_mises(first.sigma6, MANDEL_XY_FIRST),
        von_mises(last.sigma6, MANDEL_XY_LAST),
        rtol=1e-12,
    )
    assert first.sigma6[1, MANDEL_XY_FIRST.slot_of("XY")] == pytest.approx(
        last.sigma6[1, MANDEL_XY_LAST.slot_of("XY")]
    )


def test_planar_export_is_recognised(export):
    """Every z equal means a plane, which can be drawn without a reduction."""
    field = build_field("ansys", ansys_config(export))

    assert field.topology is Topology.CLOUD2D


def test_three_dimensional_export_is_recognised(tmp_path, export):
    path = tmp_path / "solid.csv"
    path.write_text(PRNSOL.replace("     3     2.0     1.0     0.0",
                                   "     3     2.0     1.0     7.0"), encoding="utf-8")
    field = build_field("ansys", ansys_config(path))

    assert field.topology is Topology.CLOUD3D


def test_missing_export_says_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="exported field not found"):
        build_field("ansys", ansys_config(tmp_path / "absent.csv"))


def test_wrong_column_names_are_reported(export):
    config = ansys_config(export, columns={"xyz": ["XX", "YY", "ZZ"]})

    with pytest.raises(ConfigError, match="no header naming"):
        build_field("ansys", config)


def test_wrong_column_count_is_reported(export):
    with pytest.raises(ConfigError, match="must name 3 columns"):
        build_field("ansys", ansys_config(export, columns={"xyz": ["X", "Y"]}))


def test_unknown_units_list_the_known_ones(export):
    config = ansys_config(export, units={"stress": "bar", "length": "m"})

    with pytest.raises(ConfigError, match="known units"):
        build_field("ansys", config)


def test_unknown_format_is_refused(export):
    with pytest.raises(ConfigError, match="not supported"):
        build_field("ansys", ansys_config(export, format="rst"))


def test_header_without_rows_is_reported(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("NODE X Y Z SX SY SZ SXY SYZ SXZ\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="no numeric rows"):
        build_field("ansys", ansys_config(path))


def test_frame_is_carried_not_assumed(export):
    """A result in a local frame cannot be compared with a global model."""
    field = build_field("ansys", ansys_config(export, frame="cylindrical"))

    assert field.meta.frame == "cylindrical"
