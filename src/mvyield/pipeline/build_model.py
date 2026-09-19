"""Fit a model bundle from a yield-point artefact.

The space transform is built once here and stored in the bundle, so every
estimator is fitted against the same prepared cloud and every later query
goes through the identical mapping. That is the whole reason the transform
is an object rather than a pair of config flags.

Only the requested methods are fitted. Asking for the slice alone fits the
slice alone -- in the legacy code the slice was constructed from an already
built ray model, so one could not be had without the other.
"""

from __future__ import annotations

import logging

import numpy as np

from mvyield.model.base import EstimatorContext
from mvyield.model.bundle import ModelBundle, YieldPointSet
from mvyield.model.calibration import CalibrationError, get_calibrator
from mvyield.model.registry import fit_estimator
from mvyield.model.transform import SpaceTransform
from mvyield.settings import HyperParam, ModelConfig

log = logging.getLogger(__name__)


def build_model(
    points: YieldPointSet,
    config: ModelConfig,
    calibrate: bool = True,
) -> ModelBundle:
    """Fit the requested estimators over a yield-point cloud.

    Parameters
    ----------
    points : YieldPointSet
        Cloud produced by dataset ingest.
    config : ModelConfig
        Space settings, method list and hyper-parameter specifications.
    calibrate : bool
        Whether to honour ``mode: auto`` specifications. When ``False``,
        automatic parameters fall back to the middle of their grid, which
        keeps unit tests and dry runs fast.

    Returns
    -------
    ModelBundle
        Fitted model with resolved hyper-parameters and their provenance.
    """
    transform = _build_transform(points, config)
    sigma_model = transform.forward(points.sigma6)
    context = EstimatorContext(sigma_model=sigma_model, transform=transform)

    log.info("model space: %s", transform.describe())
    log.info("cloud: %d points from %d geometries", points.n_points, points.n_groups)

    bundle = ModelBundle(
        points=points,
        transform=transform,
        methods=config.methods,
        primary=config.primary,
    )

    resolved = _resolve_hyperparams(points, transform, config, bundle, calibrate)

    for method in config.methods:
        options = dict(getattr(config, method, {}) or {})
        options.update(_hyperparams_for(method, resolved))

        estimator = fit_estimator(method, context, options)
        bundle.params[method] = estimator.to_params()

        # What the estimator became, not what it was asked to be. A config
        # sets 'power: 8' or 'kappa: 20', neither of which says how much of
        # the sphere is being averaged over; the estimator can say.
        describe = getattr(estimator, "describe", None)
        log.info("fitted %-10s %s", method, describe() if describe else "")

    bundle.diagnostics = _diagnostics(sigma_model, transform)
    _log_cloud(bundle.diagnostics)
    return bundle


def _log_cloud(diagnostics: dict) -> None:
    """Report the shape of the cloud every estimator was fitted to.

    Computed anyway for the bundle; printing it costs nothing and is the
    first thing to look at when a probability field is surprising. A rank
    below the space's dimension means the dataset never explored some
    direction, and every answer along it is extrapolation.
    """
    log.info(
        "cloud radius %.1f +- %.1f MPa (CV %.2f %%), range %.1f..%.1f, rank %d",
        diagnostics["r_mean"],
        diagnostics["r_std"],
        diagnostics["r_cv_pct"],
        diagnostics["r_min"],
        diagnostics["r_max"],
        diagnostics["rank"],
    )


def _build_transform(points: YieldPointSet, config: ModelConfig) -> SpaceTransform:
    """Build the space transform, fitting the scale when requested."""
    transform = SpaceTransform(
        convention=config.voigt_convention,
        deviatoric=config.deviatoric,
    )
    if config.normalise:
        transform = transform.fitted_scale(points.sigma6)
    return transform


def _resolve_hyperparams(
    points: YieldPointSet,
    transform: SpaceTransform,
    config: ModelConfig,
    bundle: ModelBundle,
    calibrate: bool,
) -> dict[str, float]:
    """Resolve every hyper-parameter and record how it was chosen."""
    resolved: dict[str, float] = {}

    for name, spec in config.hyperparams.items():
        if spec.mode == "manual":
            value = float(spec.value)
            bundle.record_hyperparam(name, value, source="manual")

        elif spec.mode == "frozen":
            raise CalibrationError(
                f"hyper-parameter {name!r} is 'frozen', but this is a fresh build "
                f"with no earlier value to reuse. Set it manually, or run "
                f"'mvy calibrate' first."
            )

        elif not calibrate:
            value = _grid_midpoint(spec)
            bundle.record_hyperparam(name, value, source="manual")
            log.warning("calibration disabled; %s set to grid midpoint %g", name, value)

        else:
            result = get_calibrator(name).calibrate(
                points, transform, spec.grid or (), refine=spec.refine
            )
            value = result.value
            bundle.record_hyperparam(
                name,
                value,
                source=result.source,
                objective=result.objective,
                grid=list(spec.grid or ()),
                curve=result.curve,
            )
            log.info("%s", result.describe(name))
            if result.diagnostics.get("note"):
                log.warning("%s: %s", name, result.diagnostics["note"])

        resolved[name] = float(bundle.hyperparam(name))

    return resolved


def _hyperparams_for(method: str, resolved: dict[str, float]) -> dict[str, float]:
    """Select the resolved hyper-parameters belonging to one method."""
    owners = {"kde_ray": ("power",), "kde_slice": ("kappa",)}
    return {name: resolved[name] for name in owners.get(method, ()) if name in resolved}


def _grid_midpoint(spec: HyperParam) -> float:
    """Middle candidate of a search grid, used when calibration is skipped."""
    grid = sorted(spec.grid or ())
    if not grid:
        raise CalibrationError("an automatic hyper-parameter needs a search grid")
    return float(grid[len(grid) // 2])


def _diagnostics(sigma_model: np.ndarray, transform: SpaceTransform) -> dict:
    """Summary statistics of the fitted cloud, stored in the bundle."""
    radii = np.linalg.norm(sigma_model, axis=1)
    mean = float(np.mean(radii))

    return {
        "n_points": len(sigma_model),
        "r_mean": mean,
        "r_std": float(np.std(radii)),
        "r_cv_pct": float(100.0 * np.std(radii) / mean) if mean > 0 else float("nan"),
        "r_min": float(radii.min()),
        "r_max": float(radii.max()),
        "rank": int(np.linalg.matrix_rank(np.cov(sigma_model, rowvar=False))),
        "space": transform.describe(),
    }
