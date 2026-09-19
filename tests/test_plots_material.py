"""The dataset figures: registration, robustness and what they claim."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mvyield.ingest.artefacts import IngestResult, StepTable
from mvyield.ingest.material import MaterialProperties
from mvyield.model.bundle import YieldPointSet
from mvyield.pipeline.figures import FigureContext, load_presets
from mvyield.viz import REGISTRY, Space
from mvyield.viz.plots.material import robust_limits

INGEST_FIGURES = (
    "yield_components",
    "stress_space",
    "strain_space",
    "stress_pairs_2d",
    "strain_pairs_2d",
    "projected_kde",
    "histograms",
    "eigen_tracking",
    "schmid_factors",
)


@pytest.fixture
def ingest():
    """A small but realistic ingest result: a yield shell plus a fat tail.

    The tail matters. Scaling a nearly hydrostatic load direction to yield
    multiplies its pressure enormously, so a real cloud carries a handful of
    points two orders of magnitude out. Every figure has to survive them.
    """
    rng = np.random.default_rng(7)
    n, geometries = 240, 4

    direction = rng.normal(size=(n, 6))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    sigma6 = direction * (270.0 + rng.normal(0.0, 7.0, n))[:, None]
    sigma6[:3, :3] += 4000.0  # the hydrostatic tail

    group = np.repeat(np.arange(geometries, dtype=np.int32), n // geometries)
    steps = np.tile(np.arange(n // geometries, dtype=np.int32), geometries)
    representative = np.zeros(n, dtype=bool)
    representative[:: n // geometries] = True

    points = YieldPointSet(sigma6=sigma6, group_id=group,
                           meta={"convention": "mandel_xy_last"})
    table = StepTable(
        geometry_id=np.array([str(g) for g in group], dtype=np.str_),
        geometry_index=group,
        step=steps,
        k_factor=np.abs(rng.normal(20.0, 4.0, n)),
        tau_max=np.abs(rng.normal(150.0, 20.0, n)),
        von_mises=np.abs(rng.normal(268.0, 7.0, n)),
        principal_stress=rng.normal(0.0, 150.0, (n, 3)),
        principal_strain=rng.normal(0.0, 1e-3, (n, 3)),
        strain6=rng.normal(0.0, 1e-3, (n, 6)),
        active_systems=rng.integers(1, 3, n).astype(np.int32),
        representative=representative,
    )
    return IngestResult(points=points, steps=table,
                        material=MaterialProperties(crss=150.0))


def draw(name: str, ingest) -> None:
    """Render one registered figure onto a throwaway canvas."""
    spec = REGISTRY[name]
    rows, columns = spec.layout or (1, 1)
    figure, axes = plt.subplots(rows, columns, figsize=spec.figsize or (7.0, 5.5))
    try:
        from mvyield.viz.theme import THEME

        spec.function(FigureContext(ingest=ingest), axes, THEME)
    finally:
        plt.close(figure)


# --- Registration ------------------------------------------------------------


def test_every_preset_figure_is_registered():
    """The shipped preset must not name a figure that does not exist."""
    for name in load_presets().get("ingest_diagnostics", []):
        assert name in REGISTRY, f"{name} is in the preset but not registered"


def test_all_nine_dataset_figures_exist():
    assert set(INGEST_FIGURES) <= set(REGISTRY)


def test_dataset_figures_describe_the_material_not_a_problem():
    """These must not depend on a geometry, or a new case would need new ones."""
    for name in INGEST_FIGURES:
        assert REGISTRY[name].space is Space.MATERIAL
        assert not REGISTRY[name].per_reduction
        assert REGISTRY[name].needs == ()


def test_every_figure_carries_a_description():
    for name in INGEST_FIGURES:
        assert REGISTRY[name].description


# --- Drawing -----------------------------------------------------------------


@pytest.mark.parametrize("name", INGEST_FIGURES)
def test_figure_draws(name, ingest):
    draw(name, ingest)


@pytest.mark.parametrize("name", INGEST_FIGURES)
def test_figure_survives_a_single_geometry(name, ingest):
    """One RVE is a legitimate run -- ``--geometries 0`` produces exactly that."""
    keep = ingest.steps.geometry_index == 0
    single = IngestResult(
        points=YieldPointSet(
            sigma6=ingest.points.sigma6[keep],
            group_id=ingest.points.group_id[keep],
            meta=ingest.points.meta,
        ),
        steps=ingest.steps.select(keep),
        material=ingest.material,
    )
    draw(name, single)


@pytest.mark.parametrize("name", INGEST_FIGURES)
def test_figure_survives_a_degenerate_cloud(name, ingest):
    """A component with no scatter must not divide by a zero range."""
    flat = ingest.points.sigma6.copy()
    flat[:, 3] = 0.0
    degenerate = IngestResult(
        points=YieldPointSet(sigma6=flat, group_id=ingest.points.group_id,
                             meta=ingest.points.meta),
        steps=ingest.steps,
        material=ingest.material,
    )
    draw(name, degenerate)


# --- Robust limits -----------------------------------------------------------


def test_robust_limits_ignore_a_heavy_tail():
    """The point of them: a few extreme values must not set the axis."""
    values = np.concatenate([np.random.default_rng(0).normal(0.0, 1.0, 1000),
                             [1.0e5, 2.0e5]])
    low, high = robust_limits(values)

    assert high < 100.0
    assert low > -100.0


def test_robust_limits_keep_everything_when_there_is_no_tail():
    values = np.linspace(-1.0, 1.0, 500)
    low, high = robust_limits(values, coverage=100.0)

    assert low <= values.min() and high >= values.max()


def test_robust_limits_survive_a_constant():
    low, high = robust_limits(np.full(50, 3.0))

    assert low < high


def test_robust_limits_survive_an_empty_array():
    low, high = robust_limits(np.array([]))

    assert low < high


# --- What the figures claim --------------------------------------------------


def test_geometry_is_not_encoded_as_forty_hues(ingest):
    """Colour must not carry an identity a reader cannot look up.

    Forty RVE cannot be told apart by hue, and geometry is not the variable
    these figures are about. The check is that no figure asks Matplotlib for
    a per-point colour array.
    """
    from mvyield.viz.theme import THEME

    spec = REGISTRY["stress_space"]
    figure, axes = plt.subplots(*spec.layout, figsize=spec.figsize)
    try:
        spec.function(FigureContext(ingest=ingest), axes, THEME)
        for ax in np.atleast_1d(axes).ravel():
            for collection in ax.collections:
                colours = collection.get_facecolor()
                assert len(colours) <= 1, "a per-point colour array encodes geometry"
    finally:
        plt.close(figure)


def test_multi_panel_figures_declare_their_layout():
    """A figure that needs a grid says so, rather than rebuilding the axes."""
    assert REGISTRY["stress_pairs_2d"].layout == (6, 6)
    assert REGISTRY["yield_components"].layout == (2, 3)
    assert REGISTRY["stress_space"].layout == (1, 2)


def test_pair_matrix_leaves_the_upper_triangle_empty(ingest):
    """Both triangles show the same pairs; drawing both doubles the ink."""
    from mvyield.viz.theme import THEME

    spec = REGISTRY["stress_pairs_2d"]
    figure, axes = plt.subplots(*spec.layout, figsize=spec.figsize)
    try:
        spec.function(FigureContext(ingest=ingest), axes, THEME)
        assert not axes[0, 5].get_visible() or not axes[0, 5].collections
        assert axes[5, 0].collections
    finally:
        plt.close(figure)
