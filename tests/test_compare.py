"""Method comparison, and the figures of the field and the surface."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mvyield.mechanics.voigt import MANDEL_XY_LAST
from mvyield.model.base import EstimatorOutput
from mvyield.model.bundle import ModelBundle, YieldPointSet
from mvyield.model.transform import SpaceTransform
from mvyield.pipeline.compare import (
    STANDARD_DIRECTIONS,
    VON_MISES_ANISOTROPY,
    compare_probability_fields,
    format_report,
    save_report,
    yield_locus,
)
from mvyield.pipeline.evaluate import ResultBundle
from mvyield.pipeline.figures import FigureContext
from mvyield.viz import REGISTRY, Space

from test_cases import kirsch_config

PAPER_FIGURES = (
    "yield_zone",
    "kde_probability",
    "three_methods",
    "probability_profile",
    "yield_locus_polar",
    "pi_plane_density",
)


@pytest.fixture
def shell_bundle():
    """A model fitted to an isotropic shell, plus the analytic baseline."""
    from mvyield.pipeline.build_model import build_model
    from mvyield.settings import ModelConfig

    rng = np.random.default_rng(3)
    n = 600
    direction = rng.normal(size=(n, 6))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    sigma6 = direction * (275.0 + rng.normal(0.0, 6.0, n))[:, None]

    points = YieldPointSet(sigma6=sigma6, group_id=np.arange(n, dtype=np.int32) % 5,
                           meta={"convention": "mandel_xy_last"})
    config = ModelConfig(methods=("analytic", "kde_ray"), primary="kde_ray",
                         analytic={"mode": "from_dataset"}, kde_ray={"power": 8.0})
    return build_model(points, config, calibrate=False)


def make_result(probabilities: dict, covered: dict) -> ResultBundle:
    """A ResultBundle over a tiny Kirsch field."""
    from mvyield.cases import build_field

    field = build_field("kirsch", kirsch_config(grid={"resolution": 20}))
    n = field.n_points

    outputs = {
        name: EstimatorOutput(
            probability=np.full(n, value, dtype=np.float64),
            spread=np.full(n, 5.0),
            covered=np.full(n, covered[name], dtype=bool),
        )
        for name, value in probabilities.items()
    }
    return ResultBundle(field=field, outputs=outputs, primary=next(iter(probabilities)))


# --- The rule that makes a comparison mean anything --------------------------


def test_metrics_use_only_the_region_both_methods_cover():
    """Scoring each method on its own subset can reverse the ranking.

    Here one method is right everywhere it claims coverage and the other is
    wrong, but only outside the first one's region. A comparison that did
    not intersect the masks would see no disagreement at all.
    """
    n = 100
    truth = np.linspace(0.0, 1.0, n)

    honest = EstimatorOutput(
        probability=truth.copy(), spread=np.ones(n),
        covered=np.arange(n) < 50,
    )
    optimistic = EstimatorOutput(
        probability=np.where(np.arange(n) < 50, truth, truth + 0.5),
        spread=np.ones(n), covered=np.ones(n, dtype=bool),
    )

    from mvyield.pipeline.compare import _compare_pair

    report = _compare_pair(honest, optimistic, "honest", "optimistic", 0.5)

    assert report["agreement"]["n_common"] == 50
    assert report["agreement"]["MAE"] == pytest.approx(0.0, abs=1e-12)
    assert report["coverage"]["covered_optimistic_pct"] == 100.0
    assert report["coverage"]["covered_honest_pct"] == 50.0


def test_disjoint_coverage_is_reported_not_averaged():
    n = 20
    a = EstimatorOutput(probability=np.zeros(n), spread=np.ones(n),
                        covered=np.arange(n) < 10)
    b = EstimatorOutput(probability=np.ones(n), spread=np.ones(n),
                        covered=np.arange(n) >= 10)

    from mvyield.pipeline.compare import _compare_pair

    report = _compare_pair(a, b, "a", "b", 0.5)

    assert "MAE" not in report["agreement"]
    assert "no covered region" in report["agreement"]["note"]
    assert report["coverage"]["jaccard"] == 0.0


def test_every_pair_of_methods_is_compared():
    result = make_result(
        {"analytic": 0.2, "kde_ray": 0.3, "kde_slice": 0.4},
        {"analytic": True, "kde_ray": True, "kde_slice": True},
    )
    pairs = compare_probability_fields(result, 0.5)

    assert len(pairs) == 3


def test_a_single_method_has_nothing_to_compare():
    result = make_result({"analytic": 0.2}, {"analytic": True})

    assert compare_probability_fields(result, 0.5) == {}


# --- The locus ---------------------------------------------------------------


def test_isotropic_baseline_has_a_tension_shear_ratio_of_one(shell_bundle):
    """The check that the reference constant is right, not sqrt(3).

    Radii here are deviatoric norms, and in that metric von Mises is a
    sphere: uniaxial tension and pure shear both yield at
    ``sqrt(6)/3 sigma_y``. The analytic estimator is isotropic by
    construction, so anything but one would mean the metric is misread.
    """
    locus = yield_locus(shell_bundle)

    assert locus["anisotropy"]["analytic"] == pytest.approx(1.0, abs=1e-9)
    assert locus["anisotropy"]["von_mises_expected"] == VON_MISES_ANISOTROPY
    assert VON_MISES_ANISOTROPY == 1.0


def test_locus_covers_every_standard_direction(shell_bundle):
    locus = yield_locus(shell_bundle)

    assert [row["direction"] for row in locus["rows"]] == list(STANDARD_DIRECTIONS)
    for row in locus["rows"]:
        assert np.isfinite(row["analytic"])


def test_reference_is_converted_from_load_to_radius(shell_bundle):
    """A paper quotes a uniaxial sigma; the model measures a deviatoric norm.

    Comparing them raw reports a unit mismatch as a model error: the same
    material quoted as 312.9 MPa in tension and 179.6 MPa in shear looks
    wildly anisotropic until both are converted, after which they agree.
    """
    locus = yield_locus(
        shell_bundle,
        reference={"x_uniaxial": 312.9, "xy_shear": 179.6},
    )
    rows = {row["direction"]: row for row in locus["rows"]}

    tension, shear = rows["x_uniaxial"], rows["xy_shear"]

    assert tension["reference_load"] == pytest.approx(312.9)
    assert tension["reference"] == pytest.approx(312.9 * np.sqrt(6.0) / 3.0, rel=1e-9)
    assert shear["reference"] == pytest.approx(179.6 * np.sqrt(2.0), rel=1e-9)

    # Converted, the two agree to a few per cent; raw they differ by 74 %.
    assert abs(tension["reference"] - shear["reference"]) / tension["reference"] < 0.02


def test_reference_accepts_a_display_label(shell_bundle):
    """The shipped case file keys the reference by label, not identifier."""
    locus = yield_locus(shell_bundle, reference={"X uniaxial": 312.9})
    rows = {row["direction"]: row for row in locus["rows"]}

    assert "reference" in rows["x_uniaxial"]


def test_unknown_reference_direction_is_warned_not_fatal(shell_bundle, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        locus = yield_locus(shell_bundle, reference={"nonsense": 100.0})

    assert "not a known direction" in caplog.text
    assert "vs_reference" not in locus or not locus["vs_reference"]


def test_report_serialises_with_non_finite_values(tmp_path, shell_bundle):
    import json

    locus = yield_locus(shell_bundle)
    locus["rows"][0]["analytic"] = float("nan")

    path = save_report({"locus": locus}, tmp_path / "cmp.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["locus"]["rows"][0]["analytic"] is None


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_report_states_the_common_coverage_rule(lang, shell_bundle):
    result = make_result(
        {"kde_ray": 0.4, "analytic": 0.6},
        {"kde_ray": True, "analytic": True},
    )
    lines = format_report(
        {"locus": yield_locus(shell_bundle),
         "fields": compare_probability_fields(result, 0.5)},
        lang,
    )
    text = "\n".join(lines)

    lowered = text.lower()
    assert "covered" in lowered or "покрит" in lowered
    assert "tension" in lowered or "розтяг" in lowered


# --- The paper figures -------------------------------------------------------


def test_paper_preset_figures_are_registered():
    for name in PAPER_FIGURES:
        assert name in REGISTRY


def test_field_figures_are_drawn_once_per_reduction():
    """A 3D field needs one map per view; an aggregate does not."""
    for name in ("yield_zone", "kde_probability", "three_methods"):
        assert REGISTRY[name].space is Space.FIELD
        assert REGISTRY[name].per_reduction

    assert REGISTRY["yield_locus_polar"].space is Space.MATERIAL
    assert not REGISTRY["pi_plane_density"].per_reduction


def test_three_methods_declares_what_it_needs():
    """It compares the slice against the rest, so it needs the slice."""
    assert REGISTRY["three_methods"].needs == ("kde_slice",)


def draw(name: str, context) -> None:
    spec = REGISTRY[name]
    rows, columns = spec.layout or (1, 1)
    figure, axes = plt.subplots(rows, columns, figsize=spec.figsize or (7.0, 5.5))
    try:
        from mvyield.viz.theme import THEME

        spec.function(context, axes, THEME)
    finally:
        plt.close(figure)


@pytest.fixture
def paper_context(shell_bundle):
    from mvyield.settings import load_case
    from mvyield.paths import find_repo_root

    cfg = load_case(find_repo_root() / "configs" / "cases" / "kirsch_bcc.yaml")
    cfg.field.domain_from = "config"
    cfg.field.hole_radius = {"auto": False, "fallback": 1.0}
    cfg.field.grid = {"resolution": 20}

    result = make_result(
        {"kde_slice": 0.3, "kde_ray": 0.4, "analytic": 0.5},
        {"kde_slice": True, "kde_ray": True, "analytic": True},
    )
    return FigureContext(
        result=result, config=cfg, bundle=shell_bundle,
        extra={"locus": yield_locus(shell_bundle,
                                    reference=cfg.compare.reference_locus),
               "points": shell_bundle.points},
    )


@pytest.mark.parametrize("name", PAPER_FIGURES)
def test_paper_figure_draws(name, paper_context):
    draw(name, paper_context)


def test_locus_figure_explains_an_empty_run(paper_context):
    paper_context.extra = {}
    draw("yield_locus_polar", paper_context)


def test_pi_plane_falls_back_to_the_bundle_cloud(paper_context):
    """Mode B has no ingest result, but the model still carries the cloud."""
    assert paper_context.ingest is None
    draw("pi_plane_density", paper_context)


def test_profile_explains_itself_without_probe_paths(paper_context):
    paper_context.config.probes.paths = ()
    draw("probability_profile", paper_context)


def test_yield_zone_uses_the_configured_threshold(paper_context):
    """The figure must follow compare.p_crit, not a hard-coded 0.5."""
    from mvyield.viz.theme import THEME

    paper_context.config.compare.p_crit = 0.25
    figure, ax = plt.subplots()
    try:
        REGISTRY["yield_zone"].function(paper_context, ax, THEME)
        assert "0.25" in ax.get_title()
    finally:
        plt.close(figure)
