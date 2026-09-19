"""The run report: what was asked for, what was used, what came out.

``summary.md`` exists so that a run directory can be read six months later
without the session that produced it. It therefore records provenance --
which dataset, which material values and where each came from, which
hyper-parameters and how they were chosen -- alongside the numbers.

Language follows ``report.lang``. Figures stay English by design, so only
this file and the console are translated; a caption and a summary line that
disagree about language is worse than either choice.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from mvyield.paths import RunPaths
from mvyield.pipeline.evaluate import ResultBundle
from mvyield.settings import CaseConfig

log = logging.getLogger(__name__)

TEXT: dict[str, dict[str, str]] = {
    "en": {
        "title": "Run report",
        "case": "Case",
        "generated": "Generated",
        "inputs": "Inputs",
        "dataset": "Dataset",
        "field": "Stress field",
        "material": "Material",
        "cloud": "Yield-point cloud",
        "model": "Model",
        "space": "Model space",
        "methods": "Methods",
        "hyperparams": "Hyper-parameters",
        "kernel": "Fitted kernels",
        "cloud_shape": "Cloud radius",
        "results": "Results",
        "method": "Method",
        "p_mean": "mean P(G)",
        "p_max": "max P(G)",
        "covered": "covered",
        "yielding": "Area above the decision threshold",
        "coverage_note": (
            "`covered` means something different per method and the columns are "
            "not comparable. `kde_ray` asks whether the dataset supports the "
            "*direction*; `kde_slice` also requires the query *radius* to lie "
            "within the yield radii observed along it, so a safely elastic "
            "field is largely uncovered by construction. Read `eff_n` for the "
            "slice method's statistical support."
        ),
        "outputs": "Files",
        "figures": "figures",
        "no_field": "No stress field was evaluated in this run.",
        "skipped": "Load steps skipped",
        "points": "points",
        "geometries": "geometries",
        "not_drawn": "No figures were drawn: the preset selected none that apply here.",
    },
    "uk": {
        "title": "Звіт про прогін",
        "case": "Кейс",
        "generated": "Створено",
        "inputs": "Вхідні дані",
        "dataset": "Датасет",
        "field": "Поле напружень",
        "material": "Матеріал",
        "cloud": "Хмара точок текучості",
        "model": "Модель",
        "space": "Простір моделі",
        "methods": "Методи",
        "hyperparams": "Гіперпараметри",
        "kernel": "Ядра після підгонки",
        "cloud_shape": "Радіус хмари",
        "results": "Результати",
        "method": "Метод",
        "p_mean": "середнє P(G)",
        "p_max": "макс. P(G)",
        "covered": "покрито",
        "yielding": "Площа понад порогом рішення",
        "coverage_note": (
            "`covered` означає різне для різних методів, колонки **не** "
            "порівнюються між собою. `kde_ray` питає, чи датасет підтримує "
            "*напрямок*; `kde_slice` додатково вимагає, щоб *радіус* запиту "
            "потрапив у діапазон радіусів текучості вздовж цього напрямку — "
            "тож надійно пружне поле майже все «непокрите» за побудовою. "
            "Статистичну забезпеченість методу перерізу показує `eff_n`."
        ),
        "outputs": "Файли",
        "figures": "графіків",
        "no_field": "У цьому прогоні поле напружень не оцінювалося.",
        "skipped": "Пропущено кроків навантаження",
        "points": "точок",
        "geometries": "геометрій",
        "not_drawn": "Графіки не малювалися: пресет не обрав жодного придатного тут.",
    },
}


def write_summary(
    cfg: CaseConfig,
    paths: RunPaths,
    result: ResultBundle | None = None,
    ingest=None,
    bundle=None,
    figures: list[Path] | None = None,
    comparison: dict | None = None,
) -> Path:
    """Write ``summary.md`` for one run.

    Parameters
    ----------
    cfg : CaseConfig
        The resolved case.
    paths : RunPaths
        Prepared run layout.
    result : ResultBundle, optional
        Field evaluation, when one was run.
    ingest : IngestResult, optional
        Dataset analysis, when one was run.
    bundle : ModelBundle, optional
        The fitted model.
    figures : list of Path, optional
        Figure files written.

    Returns
    -------
    Path
        The file written.
    """
    from datetime import datetime

    words = TEXT.get(cfg.report_lang, TEXT["en"])
    lines: list[str] = [
        f"# {words['title']}: {cfg.name}",
        "",
        f"- **{words['case']}**: `{cfg.case}` ({cfg.name})",
        f"- **{words['generated']}**: {datetime.now():%Y-%m-%d %H:%M}",
        "",
    ]

    lines += _inputs_section(cfg, words, ingest)
    lines += _model_section(words, bundle)
    lines += _results_section(cfg, words, result)

    if comparison:
        from mvyield.pipeline.compare import format_report

        lines += format_report(comparison, cfg.report_lang)

    lines += _outputs_section(words, paths, figures)

    paths.summary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", paths.summary_file.name)
    return paths.summary_file


def attach_log_file(paths: RunPaths) -> logging.Handler:
    """Send every log record to the run directory as well as the console.

    Returned so the caller can detach it: leaving handlers attached across
    runs in one process makes a later run write into an earlier directory.

    The root logger is opened up to DEBUG, because a handler cannot see what
    the logger already discarded. Console verbosity is unaffected -- that is
    set on the console handler itself -- so the file gets the full record
    while the terminal stays readable.
    """
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if root.level > logging.DEBUG:
        root.setLevel(logging.DEBUG)

    handler = logging.FileHandler(paths.log_file, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    return handler


# --- Sections ----------------------------------------------------------------


def _inputs_section(cfg, words, ingest) -> list[str]:
    """Dataset, field and material, each with where it came from."""
    lines = [f"## {words['inputs']}", ""]
    lines.append(f"- **{words['dataset']}**: `{cfg.dataset.source or '-'}` "
                 f"({cfg.dataset.analysis}, `{cfg.dataset.select_step}`)")
    lines.append(f"- **{words['field']}**: `{cfg.field.source}`"
                 + (f" <- `{cfg.field.path}`" if cfg.field.path else ""))

    if ingest is not None:
        lines += ["", f"### {words['material']}", "", "```", ingest.material.describe(), "```", ""]
        lines.append(
            f"- **{words['cloud']}**: {ingest.points.n_points} {words['points']}, "
            f"{ingest.points.n_groups} {words['geometries']}"
        )
        equivalent = ingest.steps.von_mises
        mean, std = float(np.mean(equivalent)), float(np.std(equivalent))
        lines.append(
            f"- **sigma_vm**: {mean:.2f} ± {std:.2f} MPa "
            f"(CV {100.0 * std / mean:.2f} %)" if mean > 0 else "- **sigma_vm**: -"
        )
        if ingest.skipped:
            detail = ", ".join(f"{count} ({reason})" for reason, count in sorted(ingest.skipped.items()))
            lines.append(f"- **{words['skipped']}**: {detail}")

    lines.append("")
    return lines


def _model_section(words, bundle) -> list[str]:
    """Space, methods and how each hyper-parameter was chosen."""
    if bundle is None:
        return []

    lines = [f"## {words['model']}", ""]
    lines.append(f"- **{words['space']}**: {bundle.transform.describe()}")
    lines.append(f"- **{words['methods']}**: {', '.join(bundle.methods)} "
                 f"(primary: {bundle.primary})")

    cloud = bundle.diagnostics
    if cloud:
        lines.append(
            f"- **{words['cloud_shape']}**: {cloud['r_mean']:.1f} ± "
            f"{cloud['r_std']:.1f} MPa (CV {cloud['r_cv_pct']:.2f} %), "
            f"rank {cloud['rank']}"
        )

    kernels = _kernel_lines(bundle)
    if kernels:
        lines += ["", f"### {words['kernel']}", ""]
        lines += [f"- `{name}`: {text}" for name, text in kernels]

    if bundle.hyperparams:
        lines += ["", f"### {words['hyperparams']}", "",
                  "| name | value | source | objective |",
                  "| --- | --- | --- | --- |"]
        for name, record in sorted(bundle.hyperparams.items()):
            objective = record.get("objective")
            lines.append(
                f"| `{name}` | {record['value']:g} | {record.get('source', '-')} | "
                f"{objective:.4g} |" if objective is not None else
                f"| `{name}` | {record['value']:g} | {record.get('source', '-')} | - |"
            )
    lines.append("")
    return lines


def _kernel_lines(bundle) -> list[tuple[str, str]]:
    """What each fitted estimator became, in physical terms.

    Rebuilt from the stored parameters rather than restated here: the
    half-weight angle a reader needs is derived from ``kappa`` or
    ``power``, and deriving it in two places is how the report and the
    console come to disagree.
    """
    from mvyield.model.registry import load_estimator

    cloud = bundle.transform.forward(bundle.points.sigma6)
    described: list[tuple[str, str]] = []

    for method in bundle.methods:
        try:
            estimator = load_estimator(method, cloud, bundle.params.get(method, {}))
        except Exception:  # noqa: BLE001 - a bundle missing a method's params
            continue
        describe = getattr(estimator, "describe", None)
        if describe is not None:
            described.append((method, describe()))
    return described


def _results_section(cfg, words, result) -> list[str]:
    """Per-method summary and the headline decision."""
    lines = [f"## {words['results']}", ""]

    if result is None:
        lines += [words["no_field"], ""]
        return lines

    lines += [
        f"| {words['method']} | {words['p_mean']} | {words['p_max']} | {words['covered']} |",
        "| --- | --- | --- | --- |",
    ]
    for name, stats in result.summary().items():
        covered = 100.0 * stats["n_covered"] / max(stats["n_points"], 1)
        marker = " **(primary)**" if name == result.primary else ""
        lines.append(
            f"| `{name}`{marker} | {stats['p_mean']:.4f} | {stats['p_max']:.4f} | "
            f"{covered:.0f} % |"
        )

    if len(result.outputs) > 1:
        lines += ["", f"> {words['coverage_note']}"]

    for name in result.methods:
        effective = result.outputs[name].diagnostics.get("eff_n")
        if effective is not None and len(effective):
            positive = effective[effective > 0]
            if positive.size:
                lines.append(
                    f"- `{name}` eff_n: median {np.median(positive):.1f}, "
                    f"max {positive.max():.1f}"
                )

    probability = result.probability()
    finite = np.isfinite(probability)
    if finite.any():
        share = 100.0 * float(np.mean(probability[finite] > cfg.compare.p_crit))
        lines += ["", f"- **{words['yielding']}** (P > {cfg.compare.p_crit:g}): {share:.2f} %"]

    lines.append("")
    return lines


def _outputs_section(words, paths, figures) -> list[str]:
    """What the run wrote, relative to the run directory."""
    lines = [f"## {words['outputs']}", ""]
    lines.append(f"- `{paths.log_file.name}`")
    lines.append(f"- `{paths.resolved_config.name}`")

    if figures:
        lines.append(f"- `{paths.plots_dir.name}/` — {len(figures)} {words['figures']}")
    else:
        lines.append(f"- {words['not_drawn']}")

    for item in sorted(paths.data_dir.glob("*")):
        lines.append(f"- `{paths.data_dir.name}/{item.name}`")
    lines.append("")
    return lines
