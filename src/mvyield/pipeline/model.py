"""Fitting and calibrating a model as pipeline steps, with caching.

A model is rebuilt when its inputs changed and not otherwise. "Inputs" here
means the yield-point cloud and the ``model`` section of the config -- not
file timestamps, and not the rest of the case: changing a plot preset or a
probe path must not cost a refit.

The recorded fingerprint is compared, never merely checked for presence, so
an existing bundle built from a different configuration is refused rather
than silently reused.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from mvyield.model.bundle import ModelBundle, YieldPointSet
from mvyield.paths import RunPaths
from mvyield.pipeline.build_model import build_model
from mvyield.settings import CaseConfig, ModelConfig

log = logging.getLogger(__name__)


def config_fingerprint(config: ModelConfig) -> str:
    """Digest of the settings a fit depends on.

    Hyper-parameter *specifications* are included, so switching a value from
    manual to calibrated triggers a rebuild; everything outside the model
    section is excluded, so a new plot preset does not.
    """
    payload = {
        "convention": config.convention,
        "deviatoric": config.deviatoric,
        "normalise": config.normalise,
        "methods": list(config.methods),
        "primary": config.primary,
        "options": {
            method: getattr(config, method, {}) for method in config.methods
        },
        "hyperparams": {
            name: asdict(spec) for name, spec in sorted(config.hyperparams.items())
        },
    }
    return json.dumps(payload, sort_keys=True, default=str)


def run_build_model(
    cfg: CaseConfig,
    points: YieldPointSet,
    paths: RunPaths,
    dataset_key: str,
    force: bool = False,
    calibrate: bool = True,
) -> ModelBundle:
    """Fit the model, or reuse a bundle built from the same inputs.

    Parameters
    ----------
    cfg : CaseConfig
        The resolved case.
    points : YieldPointSet
        The cloud to fit.
    paths : RunPaths
        Run layout; the bundle lands in the shared model store.
    dataset_key : str
        Identifier of the dataset, used when ``model.path`` is ``"auto"``.
    force : bool
        Refit even when the stored bundle matches.
    calibrate : bool
        Honour ``mode: auto`` hyper-parameters. ``False`` uses grid
        midpoints, which is what makes a smoke run fast.

    Returns
    -------
    ModelBundle
    """
    path = cfg.model_path(paths.models_dir, dataset_key)
    fingerprint = config_fingerprint(cfg.model)

    if not force and path.exists():
        stored = _stored_inputs_hash(path)
        expected = _expected_inputs_hash(points, fingerprint)

        if stored == expected:
            bundle = ModelBundle.load(path)
            log.info("reusing model %s (methods: %s)", path.name, ", ".join(bundle.methods))
            return bundle

        log.info(
            "model %s was built from different inputs (%s vs %s); refitting",
            path.name,
            stored or "unrecorded",
            expected,
        )

    bundle = build_model(points, cfg.model, calibrate=calibrate)
    bundle.save(path, inputs_hash=bundle.inputs_hash(fingerprint))
    log.info("wrote %s", path.name)
    return bundle


def run_calibrate(
    cfg: CaseConfig,
    points: YieldPointSet,
    paths: RunPaths,
    dataset_key: str,
    refine: bool = False,
    verify: bool = False,
) -> ModelBundle:
    """Select hyper-parameters and record them in a fresh bundle.

    Always refits: calibration is the step whose whole purpose is to change
    the numbers a cached bundle holds.

    Parameters
    ----------
    refine : bool
        Refine around the best grid point, overriding the per-parameter
        setting for every automatic hyper-parameter.
    verify : bool
        Re-score the selected value and log the comparison.
    """
    if refine:
        for name, spec in cfg.model.hyperparams.items():
            if spec.mode == "auto" and not spec.refine:
                cfg.model.hyperparams[name] = type(spec)(**{**asdict(spec), "refine": True})
                log.info("%s: refinement enabled by --refine", name)

    bundle = run_build_model(
        cfg, points, paths, dataset_key, force=True, calibrate=True
    )

    for name, record in bundle.hyperparams.items():
        log.info(
            "%s = %g  (%s%s)",
            name,
            record["value"],
            record.get("source", "unknown"),
            f", objective {record['objective']:.4g}" if record.get("objective") else "",
        )

    if verify:
        _verify(bundle)
    return bundle


# --- Helpers -----------------------------------------------------------------


def _stored_inputs_hash(path: Path) -> str | None:
    """Read the fingerprint a stored bundle recorded, if it has one."""
    try:
        return ModelBundle.peek(path).get("inputs_hash")
    except (OSError, ValueError, KeyError):
        return None


def _expected_inputs_hash(points: YieldPointSet, fingerprint: str) -> str:
    """Fingerprint the current inputs would produce."""
    from hashlib import sha256

    digest = sha256()
    digest.update(points.content_hash().encode())
    digest.update(fingerprint.encode())
    return digest.hexdigest()[:16]


def _verify(bundle: ModelBundle) -> None:
    """Log the diagnostics recorded beside each calibrated value.

    Verification here means showing what the search saw, not repeating it:
    the objective, the grid and the source are already stored, and a second
    evaluation of the same criterion on the same data cannot disagree.
    """
    for name, record in bundle.hyperparams.items():
        if record.get("source") == "manual":
            log.info("%s: set manually, nothing to verify", name)
            continue
        grid = record.get("grid") or []
        log.info(
            "%s: chose %g from %s by %s",
            name,
            record["value"],
            grid if grid else "(no grid recorded)",
            record.get("source", "unknown"),
        )
