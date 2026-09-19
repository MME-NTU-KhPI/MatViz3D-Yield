"""Comparing methods: where they agree, and where they admit to not knowing.

The one rule that decides whether a comparison means anything
--------------------------------------------------------------
Accuracy is measured **only where every method being compared declares the
point covered**. Skipping this inverts the conclusion. On the synthetic test
that motivated the legacy module:

===========================  =========  ==================
each method on its own subset  MAE       coverage
===========================  =========  ==================
ray, angle-weighted            0.052     100 % of 2000
slice                          0.132      37 % of 2000
===========================  =========  ==================

and on the 295 points both covered, ray scores 0.151 against the slice's
0.123 -- the opposite ranking. The angle-weighted ray declares nearly every
point covered, including points well outside the data, so most of its
apparent accuracy in the first table is credit for extrapolation it never
flagged as extrapolation.

Coverage is therefore reported as its own result rather than as a footnote
to accuracy. "How well do they agree where both have data" and "how
honestly does each admit to having none" are two findings, and the second
is not the lesser one.

Directions are keyed by ASCII identifier
----------------------------------------
The legacy tables used Cyrillic display strings as dictionary keys, putting
the interface language inside the program logic: translating a label would
have broken every lookup. Here the key is an identifier and the label is
looked up for display only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from mvyield.mechanics.voigt import SQRT2
from mvyield.model.bundle import ModelBundle
from mvyield.model.registry import load_estimator
from mvyield.pipeline.evaluate import ResultBundle
from mvyield.settings import CaseConfig
from mvyield.viz.labels import DIRECTION_LABELS, direction_label

log = logging.getLogger(__name__)

EPS = 1e-12

STANDARD_DIRECTIONS: dict[str, np.ndarray] = {
    "x_uniaxial": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "y_uniaxial": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    "z_uniaxial": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    "yz_shear": np.array([0.0, 0.0, 0.0, SQRT2, 0.0, 0.0]),
    "xz_shear": np.array([0.0, 0.0, 0.0, 0.0, SQRT2, 0.0]),
    "xy_shear": np.array([0.0, 0.0, 0.0, 0.0, 0.0, SQRT2]),
    "biaxial_xy": np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    "pure_shear": np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
}
"""Loading directions probed for the locus, in ``MANDEL_XY_LAST``."""

TENSION_KEYS = ("x_uniaxial", "y_uniaxial", "z_uniaxial")
SHEAR_KEYS = ("xy_shear", "yz_shear", "xz_shear")

VON_MISES_ANISOTROPY = 1.0
"""Tension-to-shear radius ratio of an isotropic von Mises surface.

**One, not sqrt(3).** The radii here are deviatoric norms in the model
space, and in that metric von Mises is a sphere. At yield under uniaxial
tension ``sigma_y`` the deviator is
``(2/3, -1/3, -1/3) sigma_y``, of norm ``sqrt(6)/3 sigma_y = 0.8165
sigma_y``; pure shear yields at ``tau = sigma_y / sqrt(3)``, whose Mandel
vector has norm ``sqrt(2) tau = 0.8165 sigma_y``. The same number, so the
ratio is exactly one.

The legacy comparison carried ``sqrt(3)`` here, which belongs to a different
metric -- radius measured as the scalar load magnitude, ``sigma`` for
tension against ``tau`` for shear. Against that reference every method looks
about 40 % too isotropic, and the measured 1.03 to 1.05 would read as a
gross failure instead of what it is: a real three-to-five per cent
tension-shear anisotropy of the BCC polycrystal.

The isotropic baseline returning exactly 1.000 is the check that the
reference is right, since that estimator is isotropic by construction.
"""


# --- Field agreement ---------------------------------------------------------


def compare_probability_fields(
    result: ResultBundle,
    p_crit: float = 0.5,
) -> dict:
    """Agreement between every pair of methods, on their common coverage."""
    methods = result.methods
    if len(methods) < 2:
        return {}

    pairs = {}
    for index, first in enumerate(methods):
        for second in methods[index + 1 :]:
            pairs[f"{first}__vs__{second}"] = _compare_pair(
                result.outputs[first], result.outputs[second], first, second, p_crit
            )
    return pairs


def _compare_pair(output_a, output_b, label_a: str, label_b: str, p_crit: float) -> dict:
    """Coverage, agreement and decision agreement for one pair."""
    p_a = np.asarray(output_a.probability, dtype=np.float64)
    p_b = np.asarray(output_b.probability, dtype=np.float64)
    cov_a = np.asarray(output_a.covered, dtype=bool)
    cov_b = np.asarray(output_b.covered, dtype=bool)

    both = cov_a & cov_b & np.isfinite(p_a) & np.isfinite(p_b)

    report = {
        "n_points": int(p_a.size),
        "label_a": label_a,
        "label_b": label_b,
        "coverage": {
            f"covered_{label_a}_pct": 100.0 * float(cov_a.mean()),
            f"covered_{label_b}_pct": 100.0 * float(cov_b.mean()),
            "covered_both_pct": 100.0 * float(both.mean()),
            "covered_neither_pct": 100.0 * float((~cov_a & ~cov_b).mean()),
            f"only_{label_a}_pct": 100.0 * float((cov_a & ~cov_b).mean()),
            f"only_{label_b}_pct": 100.0 * float((~cov_a & cov_b).mean()),
            "jaccard": float((cov_a & cov_b).sum() / max((cov_a | cov_b).sum(), 1)),
        },
    }

    if int(both.sum()) < 2:
        report["agreement"] = {
            "n_common": int(both.sum()),
            "note": "the methods share no covered region, so they cannot be compared",
        }
        report["decision"] = {}
        return report

    a, b = p_a[both], p_b[both]
    difference = b - a

    report["agreement"] = {
        "n_common": int(both.sum()),
        "MAE": float(np.mean(np.abs(difference))),
        "RMSE": float(np.sqrt(np.mean(difference**2))),
        "max_abs_diff": float(np.max(np.abs(difference))),
        "R2": _r_squared(a, b),
        "bias_b_minus_a": float(np.mean(difference)),
        "diff_p05": float(np.percentile(difference, 5)),
        "diff_p95": float(np.percentile(difference, 95)),
        "argmax_diff_index": int(np.flatnonzero(both)[int(np.argmax(np.abs(difference)))]),
    }

    yields_a, yields_b = a >= p_crit, b >= p_crit
    report["decision"] = {
        "p_crit": p_crit,
        "agree_pct": 100.0 * float((yields_a == yields_b).mean()),
        f"only_{label_a}_yields_pct": 100.0 * float((yields_a & ~yields_b).mean()),
        f"only_{label_b}_yields_pct": 100.0 * float((~yields_a & yields_b).mean()),
    }
    return report


# --- The yield locus ---------------------------------------------------------


def yield_locus(
    bundle: ModelBundle,
    directions: dict[str, np.ndarray] | None = None,
    reference: dict[str, float] | None = None,
) -> dict:
    """Yield radius along standard directions, for every fitted method.

    One bundle supplies every method, so the radii are guaranteed to come
    from the same cloud and the same space -- which is what makes the
    comparison between them a statement about the methods rather than about
    their inputs.

    Parameters
    ----------
    bundle : ModelBundle
        Fitted model.
    directions : dict, optional
        Identifier to a 6-vector. Defaults to :data:`STANDARD_DIRECTIONS`.
    reference : dict, optional
        Independent values in MPa, keyed by identifier or by display label,
        given as **load magnitudes**: the uniaxial stress or the shear
        stress at yield, as a paper would quote them. They are converted to
        deviatoric radii here, per direction, before any error is computed.

    Returns
    -------
    dict
        ``rows``, ``anisotropy`` and, when a reference is given,
        ``vs_reference``.
    """
    directions = directions or STANDARD_DIRECTIONS
    reference = _normalise_reference(reference)
    cloud = bundle.transform.forward(bundle.points.sigma6)

    estimators = {
        name: load_estimator(name, cloud, bundle.params.get(name, {}))
        for name in bundle.methods
    }

    rows = []
    for key, raw in directions.items():
        row = {"direction": key, "label": direction_label(key)}
        projected = bundle.transform.forward(np.asarray(raw, dtype=np.float64)[None, :])[0]

        for name, estimator in estimators.items():
            row[name] = _safe_radius(estimator, projected, name, key)

        if key in reference:
            # A reference value is a load magnitude -- the uniaxial sigma or
            # the shear tau at yield -- while a radius here is a deviatoric
            # norm. The two differ by a factor that depends on the
            # direction, so comparing them raw reports a unit mismatch as a
            # model error. See LOAD_TO_RADIUS.
            factor = float(np.linalg.norm(projected))
            row["reference_load"] = float(reference[key])
            row["reference"] = float(reference[key]) * factor
            row["load_to_radius"] = factor
            for name in estimators:
                if np.isfinite(row[name]):
                    row[f"err_{name}"] = row[name] - row["reference"]
        rows.append(row)

    report: dict = {
        "rows": rows,
        "methods": list(bundle.methods),
        "anisotropy": {
            name: _anisotropy(rows, name) for name in bundle.methods
        },
    }
    report["anisotropy"]["von_mises_expected"] = VON_MISES_ANISOTROPY

    if reference:
        report["vs_reference"] = {}
        for name in bundle.methods:
            errors = [row[f"err_{name}"] for row in rows if f"err_{name}" in row]
            if errors:
                report["vs_reference"][name] = {
                    "MAE_MPa": float(np.mean(np.abs(errors))),
                    "bias_MPa": float(np.mean(errors)),
                    "max_abs_MPa": float(np.max(np.abs(errors))),
                }
    return report


def _safe_radius(estimator, direction: np.ndarray, method: str, key: str) -> float:
    """Yield radius, or NaN with an explanation rather than a crash.

    A direction a method cannot answer for is a result, not an error: the
    row shows a gap and the rest of the table still stands.
    """
    try:
        return float(estimator.yield_radius(direction))
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        log.warning("%s has no yield radius along %s: %s", method, key, exc)
        return float("nan")


def _anisotropy(rows: list[dict], method: str) -> float:
    """Mean tensile radius over mean shear radius."""
    tension = [row[method] for row in rows if row["direction"] in TENSION_KEYS]
    shear = [row[method] for row in rows if row["direction"] in SHEAR_KEYS]

    tension = [v for v in tension if np.isfinite(v) and v > EPS]
    shear = [v for v in shear if np.isfinite(v) and v > EPS]
    if not tension or not shear:
        return float("nan")
    return float(np.mean(tension) / np.mean(shear))


def _normalise_reference(reference: dict | None) -> dict[str, float]:
    """Accept a reference keyed by identifier or by display label."""
    if not reference:
        return {}

    by_label = {label: key for key, label in DIRECTION_LABELS.items()}
    resolved: dict[str, float] = {}

    for key, value in reference.items():
        identifier = key if key in STANDARD_DIRECTIONS else by_label.get(key)
        if identifier is None:
            log.warning(
                "compare.reference_locus names %r, which is not a known direction; "
                "known: %s",
                key,
                sorted(STANDARD_DIRECTIONS),
            )
            continue
        resolved[identifier] = float(value)
    return resolved


# --- Putting it together -----------------------------------------------------


def compare_methods(result: ResultBundle, bundle: ModelBundle, cfg: CaseConfig) -> dict:
    """Everything the comparison produces, for the report and the figures."""
    return {
        "p_crit": cfg.compare.p_crit,
        "methods": list(result.methods),
        "primary": result.primary,
        "fields": compare_probability_fields(result, cfg.compare.p_crit),
        "locus": yield_locus(bundle, reference=cfg.compare.reference_locus),
    }


def save_report(report: dict, path: Path) -> Path:
    """Write the comparison as JSON beside the run's other data.

    Non-finite values become ``null``. Python's encoder writes them as the
    bare tokens ``NaN`` and ``Infinity``, which no JSON parser outside
    Python is required to accept -- so a direction a method could not answer
    for would make the whole file unreadable to anything else.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


def format_report(report: dict, lang: str = "en") -> list[str]:
    """Render the comparison as Markdown lines for ``summary.md``."""
    words = _WORDS.get(lang, _WORDS["en"])
    lines = [f"## {words['title']}", ""]

    locus = report.get("locus") or {}
    rows, methods = locus.get("rows", []), locus.get("methods", [])

    if rows:
        header = " | ".join([words["direction"], *methods]
                            + ([words["reference"]] if any("reference" in r for r in rows) else []))
        lines += [f"| {header} |", "| " + " | ".join(["---"] * (len(header.split("|")))) + " |"]
        for row in rows:
            cells = [row["label"]] + [_mpa(row.get(m)) for m in methods]
            if any("reference" in r for r in rows):
                cells.append(_mpa(row.get("reference")))
            lines.append("| " + " | ".join(cells) + " |")

        if any("reference_load" in row for row in rows):
            lines += ["", f"> {words['reference_note']}"]

        ratios = locus.get("anisotropy", {})
        expected = ratios.get("von_mises_expected", VON_MISES_ANISOTROPY)
        shown = ", ".join(
            f"`{name}` {ratios[name]:.3f}" for name in methods if np.isfinite(ratios.get(name, np.nan))
        )
        lines += ["", f"- **{words['anisotropy']}**: {shown} "
                      f"({words['expected']} {expected:.3f})"]

    for pair, block in (report.get("fields") or {}).items():
        agreement = block.get("agreement", {})
        coverage = block.get("coverage", {})
        lines += ["", f"### {pair.replace('__vs__', ' vs ')}", ""]
        lines.append(
            f"- **{words['common']}**: {coverage.get('covered_both_pct', 0.0):.1f} % "
            f"(Jaccard {coverage.get('jaccard', 0.0):.3f})"
        )
        if "MAE" in agreement:
            lines.append(
                f"- **{words['agreement']}** ({agreement['n_common']} {words['points']}): "
                f"MAE {agreement['MAE']:.4f}, RMSE {agreement['RMSE']:.4f}, "
                f"R² {agreement['R2']:.3f}, bias {agreement['bias_b_minus_a']:+.4f}"
            )
            decision = block.get("decision", {})
            if decision:
                lines.append(
                    f"- **{words['decision']}** (P > {decision['p_crit']:g}): "
                    f"{decision['agree_pct']:.1f} % {words['agree']}"
                )
        else:
            lines.append(f"- {agreement.get('note', '')}")

    lines += ["", f"> {words['caveat']}", ""]
    return lines


_WORDS = {
    "en": {
        "title": "Method comparison",
        "direction": "Direction",
        "reference": "Reference",
        "anisotropy": "Tension / shear ratio",
        "expected": "von Mises expects",
        "common": "Covered by both",
        "agreement": "Agreement on the common region",
        "decision": "Same yield/no-yield decision",
        "agree": "of points",
        "points": "points",
        "reference_note": (
            "Radii are deviatoric norms. `compare.reference_locus` is read as a "
            "load magnitude -- the uniaxial sigma or the shear tau at yield -- "
            "and converted per direction before the comparison, so the column "
            "above is the reference in the model's own metric, not the number "
            "as written in the config."
        ),
        "caveat": (
            "Accuracy is measured only where every method declares the point "
            "covered. Scoring each method on its own subset rewards a method "
            "for extrapolating without saying so, and can reverse the ranking."
        ),
    },
    "uk": {
        "title": "Порівняння методів",
        "direction": "Напрямок",
        "reference": "Еталон",
        "anisotropy": "Відношення розтяг / зсув",
        "expected": "фон Мізес очікує",
        "common": "Покрито обома",
        "agreement": "Узгодженість на спільній зоні",
        "decision": "Однакове рішення «тече / не тече»",
        "agree": "точок",
        "points": "точок",
        "reference_note": (
            "Радіуси — це девіаторні норми. `compare.reference_locus` читається "
            "як **величина навантаження** (одновісне σ або дотичне τ при "
            "текучості) і переводиться в радіус окремо для кожного напрямку. "
            "Тож у колонці вище — еталон у метриці моделі, а не число з конфігу."
        ),
        "caveat": (
            "Точність міряється лише там, де **кожен** метод визнав точку "
            "покритою. Оцінювання кожного методу на його власній підмножині "
            "винагороджує за неоголошену екстраполяцію і може перевернути "
            "порядок методів."
        ),
    },
}


def _mpa(value) -> str:
    """Format a radius in MPa, or a dash when a method had no answer."""
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.1f}"


def _r_squared(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Coefficient of determination, guarding a constant reference."""
    residual = float(np.sum((reference - prediction) ** 2))
    total = float(np.sum((reference - np.mean(reference)) ** 2))
    if total <= EPS:
        return 1.0 if residual <= EPS else 0.0
    return 1.0 - residual / total


def _json_safe(value):
    """Convert a structure into one the JSON encoder accepts strictly.

    Walks the whole structure rather than relying on the encoder's
    ``default`` hook: that hook is called only for types the encoder cannot
    handle, and it handles floats -- including NaN -- perfectly well, just
    not portably.
    """
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    return str(value)
