"""The three modes end to end: caching, the report, replotting, the CLI."""

from __future__ import annotations

import logging
import shutil

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="mode A needs the 'ingest' extra")

import matplotlib

matplotlib.use("Agg")

from mvyield import cli
from mvyield.paths import RunPaths, find_repo_root
from mvyield.pipeline.ingest import dataset_key, run_ingest
from mvyield.pipeline.model import config_fingerprint, run_build_model
from mvyield.pipeline.replot import load_results, replot
from mvyield.pipeline.report import write_summary
from mvyield.pipeline.run import run_mode_a, run_mode_c
from mvyield.pipeline.scaffold import write_case
from mvyield.settings import ConfigError, load_case
from mvyield.viz import registry

from test_ingest import write_dataset

CASE = """\
case: kirsch
name: smoke
dataset:
  source: {dataset}
  analysis: SCHMID
  material_override: {{crss: 150.0}}
field:
  source: analytic
  sigma_applied: 300.0
  domain_from: config
  hole_radius: {{auto: false, fallback: 1.0}}
  grid: {{resolution: 24}}
model:
  path: auto
  deviatoric: true
  methods: [analytic]
  primary: analytic
  analytic: {{mode: from_dataset}}
plots:
  preset: minimal
  formats: [png]
report:
  lang: {lang}
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A self-contained project: a repo root, a dataset and a case file."""
    root = tmp_path / "project"
    (root / "configs" / "cases").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")

    # A real project has one, and the figure presets are resolved from it.
    shutil.copy(
        find_repo_root() / "configs" / "defaults.yaml", root / "configs" / "defaults.yaml"
    )

    dataset = write_dataset(root / "data" / "mini.hdf5",
                            geometry_ids=("0", "1", "2"), n_steps=4, cube=3, n_grains=5)

    def make(lang="en"):
        path = root / "configs" / "cases" / "smoke.yaml"
        path.write_text(CASE.format(dataset=dataset.as_posix(), lang=lang), encoding="utf-8")
        return path

    monkeypatch.setattr("mvyield.paths.find_repo_root", lambda start=None: root)
    monkeypatch.setattr("mvyield.cli.find_repo_root", lambda start=None: root)
    monkeypatch.chdir(root)

    make.root = root
    make.dataset = dataset
    return make


def paths_for(root, name="smoke", kind="run") -> RunPaths:
    return RunPaths.for_run(name, kind=kind, repo_root=root).prepare()


def load(project, lang="en"):
    return load_case(project(lang))


# --- Caching -----------------------------------------------------------------


def test_dataset_key_depends_on_the_analysis(project):
    cfg = load(project)
    first = dataset_key(project.dataset, cfg.dataset)

    cfg.dataset.analysis = "VON_MISES"
    assert dataset_key(project.dataset, cfg.dataset) != first


def test_dataset_key_depends_on_the_material_override(project):
    cfg = load(project)
    first = dataset_key(project.dataset, cfg.dataset)

    cfg.dataset.material_override = {"crss": 200.0}
    assert dataset_key(project.dataset, cfg.dataset) != first


def test_second_ingest_reuses_the_artefact(project, caplog):
    cfg = load(project)
    paths = paths_for(project.root)

    first = run_ingest(cfg, paths)
    with caplog.at_level(logging.INFO):
        second = run_ingest(cfg, paths)

    assert "reusing dataset analysis" in caplog.text
    np.testing.assert_allclose(first.points.sigma6, second.points.sigma6)
    assert second.material.crss == pytest.approx(150.0)
    assert second.material.provenance["crss"] == "case config material_override"


def test_force_re_analyses(project, caplog):
    cfg = load(project)
    paths = paths_for(project.root)
    run_ingest(cfg, paths)

    with caplog.at_level(logging.INFO):
        run_ingest(cfg, paths, force=True)

    assert "reusing" not in caplog.text


def test_partial_analysis_is_not_cached(project):
    cfg = load(project)
    paths = paths_for(project.root)

    partial = run_ingest(cfg, paths, geometries=["0"])
    assert partial.points.n_groups == 1

    full = run_ingest(cfg, paths)
    assert full.points.n_groups == 3


def test_model_is_reused_when_nothing_changed(project, caplog):
    cfg = load(project)
    paths = paths_for(project.root)
    ingest = run_ingest(cfg, paths)
    key = dataset_key(project.dataset, cfg.dataset)

    run_build_model(cfg, ingest.points, paths, key)
    with caplog.at_level(logging.INFO):
        run_build_model(cfg, ingest.points, paths, key)

    assert "reusing model" in caplog.text


def test_a_changed_model_setting_forces_a_refit(project, caplog):
    cfg = load(project)
    paths = paths_for(project.root)
    ingest = run_ingest(cfg, paths)
    key = dataset_key(project.dataset, cfg.dataset)

    run_build_model(cfg, ingest.points, paths, key)
    cfg.model.deviatoric = False

    with caplog.at_level(logging.INFO):
        run_build_model(cfg, ingest.points, paths, key)

    assert "different inputs" in caplog.text


def test_fingerprint_ignores_settings_outside_the_model(project):
    cfg = load(project)
    before = config_fingerprint(cfg.model)

    cfg.plots.preset = "paper"
    cfg.probes.paths = ({"name": "p", "kind": "line"},)

    assert config_fingerprint(cfg.model) == before


# --- Modes -------------------------------------------------------------------


def test_mode_a_writes_a_report(project):
    cfg = load(project)
    outcome = run_mode_a(cfg, paths_for(project.root, kind="ingest"))

    assert outcome.ingest.points.n_points == 12
    assert outcome.paths.summary_file.exists()
    assert "Yield-point cloud" in outcome.paths.summary_file.read_text(encoding="utf-8")


def test_mode_c_runs_the_whole_chain(project):
    cfg = load(project)
    outcome = run_mode_c(cfg, paths_for(project.root), calibrate=False)

    assert outcome.ingest is not None
    assert outcome.bundle is not None
    assert outcome.result.field.n_points == 24 * 24
    assert (outcome.paths.data_dir / "results.npz").exists()
    assert outcome.paths.summary_file.exists()


def test_report_follows_the_configured_language(project):
    outcome = run_mode_c(load(project, lang="uk"), paths_for(project.root), calibrate=False)
    text = outcome.paths.summary_file.read_text(encoding="utf-8")

    assert "Звіт про прогін" in text
    assert "Хмара точок текучості" in text


def test_report_records_material_provenance(project):
    outcome = run_mode_c(load(project), paths_for(project.root), calibrate=False)
    text = outcome.paths.summary_file.read_text(encoding="utf-8")

    assert "case config material_override" in text


def test_report_warns_that_coverage_is_not_comparable(project):
    """Two methods, two meanings of 'covered'; the table must say so."""
    cfg = load(project)
    cfg.model.methods = ("analytic", "kde_ray")
    cfg.model.kde_ray = {"power": 8.0}
    cfg.model.__post_init__()

    outcome = run_mode_c(cfg, paths_for(project.root), calibrate=False)
    text = outcome.paths.summary_file.read_text(encoding="utf-8")

    assert "not comparable" in text


def test_run_log_lands_in_the_run_directory(project):
    cfg = load(project)
    paths = paths_for(project.root)
    from mvyield.pipeline.report import attach_log_file

    handler = attach_log_file(paths)
    logging.getLogger("mvyield.test").info("a recorded line")
    logging.getLogger().removeHandler(handler)
    handler.close()

    assert "a recorded line" in paths.log_file.read_text(encoding="utf-8")


def test_minimal_preset_draws_the_field_figures(project):
    outcome = run_mode_c(load(project), paths_for(project.root), calibrate=False)

    drawn = {path.stem for path in outcome.figures}
    assert drawn == {"yield_zone", "kde_probability"}


def test_summary_says_when_nothing_was_drawn(project, monkeypatch):
    """An empty selection must be stated, not left as a missing folder."""
    monkeypatch.setitem(cfg_presets(project), "minimal", [])
    outcome = run_mode_c(load(project), paths_for(project.root), calibrate=False)

    assert outcome.figures == []
    assert "selected none" in outcome.paths.summary_file.read_text(encoding="utf-8")


def cfg_presets(project) -> dict:
    """The preset table the run will resolve against."""
    import yaml

    path = project.root / "configs" / "defaults.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    presets = data["plot_presets"]

    class _Writable(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            data["plot_presets"] = dict(self)
            path.write_text(yaml.safe_dump(data), encoding="utf-8")

    return _Writable(presets)


def test_mode_a_draws_the_dataset_figures(project):
    """Mode A resolves 'ingest_diagnostics' and draws all nine."""
    outcome = run_mode_a(load(project), paths_for(project.root, kind="ingest"))

    assert len(outcome.figures) == 9
    assert all(path.exists() for path in outcome.figures)


# --- Figures -----------------------------------------------------------------


@pytest.fixture
def one_plot():
    """Register a throwaway figure so the drawing path can be exercised."""
    saved = dict(registry.REGISTRY)
    registry.REGISTRY.clear()

    drawn: list = []

    @registry.plot("smoke_map", space=registry.Space.FIELD, groups=["minimal"],
                   description="A test figure")
    def _draw(context, ax, theme):
        drawn.append(context)
        ax.plot([0, 1], [0, 1])

    yield drawn

    registry.REGISTRY.clear()
    registry.REGISTRY.update(saved)


def test_registered_figures_are_drawn_and_written(project, one_plot, monkeypatch):
    cfg = load(project)
    monkeypatch.setitem(cfg.raw, "plot_presets", {"minimal": ["smoke_map"]})

    outcome = run_mode_c(cfg, paths_for(project.root), calibrate=False)

    assert len(one_plot) == 1
    assert outcome.figures and outcome.figures[0].suffix == ".png"
    assert outcome.figures[0].exists()


def test_a_figure_that_raises_costs_only_itself(project, monkeypatch, caplog):
    """A run that produced numbers must not be lost to one bad plot."""
    saved = dict(registry.REGISTRY)
    registry.REGISTRY.clear()

    @registry.plot("broken", space=registry.Space.AGGREGATE, groups=["minimal"])
    def _draw(context, ax, theme):
        raise RuntimeError("no")

    cfg = load(project)
    monkeypatch.setitem(cfg.raw, "plot_presets", {"minimal": ["broken"]})

    try:
        with caplog.at_level(logging.ERROR):
            outcome = run_mode_c(cfg, paths_for(project.root), calibrate=False)
    finally:
        registry.REGISTRY.clear()
        registry.REGISTRY.update(saved)

    assert "figure 'broken' failed" in caplog.text
    assert outcome.paths.summary_file.exists()


# --- Replotting --------------------------------------------------------------


def test_replot_reads_the_stored_arrays(project, one_plot, monkeypatch):
    cfg = load(project)
    monkeypatch.setitem(cfg.raw, "plot_presets", {"minimal": ["smoke_map"]})
    outcome = run_mode_c(cfg, paths_for(project.root), calibrate=False)

    one_plot.clear()
    written = replot(outcome.paths.run_dir)

    assert len(one_plot) == 1
    assert written


def test_replot_recovers_the_field_it_stored(project):
    outcome = run_mode_c(load(project), paths_for(project.root), calibrate=False)
    restored = load_results(outcome.paths.data_dir / "results.npz",
                            load(project))

    np.testing.assert_allclose(restored.field.sigma6, outcome.result.field.sigma6)
    np.testing.assert_allclose(
        restored.probability(), outcome.result.probability(), equal_nan=True
    )
    assert restored.field.topology is outcome.result.field.topology


def test_replot_without_stored_results_says_what_to_do(project, tmp_path):
    outcome = run_mode_c(load(project), paths_for(project.root), calibrate=False)
    (outcome.paths.data_dir / "results.npz").unlink()

    with pytest.raises(FileNotFoundError, match="export.results_npz"):
        replot(outcome.paths.run_dir)


def test_replot_refuses_a_directory_that_is_not_a_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.resolved.yaml"):
        replot(tmp_path)


# --- Scaffolding -------------------------------------------------------------


@pytest.mark.parametrize("template", ["kirsch", "ansys"])
def test_generated_case_files_load(tmp_path, template):
    """A starter file must parse; only its dataset path needs filling in."""
    target = write_case("fresh", template, tmp_path)
    cfg = load_case(target)

    assert cfg.name == "fresh"
    assert cfg.case == template


def test_scaffold_refuses_to_overwrite(tmp_path):
    write_case("fresh", "kirsch", tmp_path)

    with pytest.raises(ConfigError, match="already exists"):
        write_case("fresh", "kirsch", tmp_path)


def test_unknown_template_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="available"):
        write_case("fresh", "notch", tmp_path)


# --- The command line --------------------------------------------------------


def test_all_runs_from_the_command_line(project, capsys):
    case = project()
    assert cli.main(["all", "--case", str(case), "--no-calibrate"]) == 0

    out = capsys.readouterr().out
    assert "Done. Results in" in out
    assert list((project.root / "runs").glob("*_run_smoke")) != []


def test_ingest_runs_from_the_command_line(project):
    assert cli.main(["ingest", "--case", str(project()), "--geometries", "0,1"]) == 0
    assert list((project.root / "runs").glob("*_ingest_smoke")) != []


def test_dry_run_computes_nothing(project):
    assert cli.main(["all", "--case", str(project()), "--dry-run"]) == 0
    assert not (project.root / "runs").exists()


def test_missing_dataset_exits_with_a_message(project, caplog):
    case = project()
    with caplog.at_level(logging.ERROR):
        code = cli.main(["all", "--case", str(case), "--set", "dataset.source=data/absent.hdf5"])

    assert code == 2
    assert "MVY_DATA_DIR" in caplog.text


def test_run_without_a_model_says_how_to_build_one(project, caplog):
    with caplog.at_level(logging.ERROR):
        code = cli.main(["run", "--case", str(project())])

    assert code == 2
    assert "mvy build-model" in caplog.text


def test_build_model_then_run_uses_it(project, capsys):
    case = project()
    assert cli.main(["build-model", "--case", str(case)]) == 0
    capsys.readouterr()

    assert cli.main(["run", "--case", str(case)]) == 0
    assert "Done. Results in" in capsys.readouterr().out


def test_init_case_writes_a_runnable_skeleton(project, capsys):
    assert cli.main(["init-case", "wire2", "--from", "ansys"]) == 0

    target = project.root / "configs" / "cases" / "wire2.yaml"
    assert target.exists()
    assert load_case(target).case == "ansys"


def test_plot_list_works_without_a_run(capsys):
    assert cli.main(["plot", "--list"]) == 0

    listing = capsys.readouterr().out
    assert "yield_components" in listing
    assert "material" in listing


def test_plot_without_a_directory_is_refused(caplog):
    with caplog.at_level(logging.ERROR):
        assert cli.main(["plot"]) == 2
    assert "run directory" in caplog.text
