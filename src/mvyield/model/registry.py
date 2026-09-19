"""Method name to estimator, and the two ways one comes into being.

An estimator is constructed twice in a model's life: fitted from the yield
cloud when the model is built, and rebuilt from stored parameters when a
field is evaluated. Both paths go through this module, so adding a method
means adding one entry here rather than editing a chain of branches in the
pipeline.

Option keys are checked against the estimator's own signature. A typo in
``model.kde_slice`` would otherwise be dropped in silence and the run would
finish on defaults -- the kind of failure that survives all the way to a
figure.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np

from mvyield.model.base import EstimatorContext, YieldEstimator
from mvyield.model.gaussian import IsotropicBaseline
from mvyield.model.kde_ray import RayMarginalEstimator
from mvyield.model.kde_slice import SliceEstimator
from mvyield.settings import ConfigError

ESTIMATORS: dict[str, type] = {
    "analytic": IsotropicBaseline,
    "kde_ray": RayMarginalEstimator,
    "kde_slice": SliceEstimator,
}
"""Every method that can be fitted, keyed by the name used in configs."""

PENDING: dict[str, str] = {
    "pinn": "the neural yield surface needs the 'pinn' extra and is not built yet",
}
"""Names accepted by the config but not yet implemented, with the reason."""


def fit_estimator(
    name: str,
    context: EstimatorContext,
    options: dict | None = None,
) -> YieldEstimator:
    """Fit one estimator over the shared yield cloud.

    Parameters
    ----------
    name : str
        Method name from ``model.methods``.
    context : EstimatorContext
        Cloud and space, already prepared by the pipeline.
    options : dict, optional
        The method's config section, with any calibrated hyper-parameter
        already substituted for its specification.

    Returns
    -------
    YieldEstimator
    """
    estimator = _lookup(name)
    options = dict(options or {})

    if estimator is IsotropicBaseline:
        return _fit_baseline(context, options)

    return estimator.fit(context, **_checked(estimator.fit, options, name))


def load_estimator(
    name: str,
    sigma_model: np.ndarray,
    params: dict,
) -> YieldEstimator:
    """Rebuild a fitted estimator from stored bundle parameters.

    The cloud is passed separately because the kernel estimators keep it as
    their state and the bundle stores it once, not once per method. Whether
    a method wants it is read from its own ``from_params`` signature rather
    than hard-coded here.
    """
    estimator = _lookup(name)
    signature = inspect.signature(estimator.from_params)

    if "sigma_model" in signature.parameters:
        return estimator.from_params(sigma_model, dict(params))
    return estimator.from_params(dict(params))


def available_methods() -> tuple[str, ...]:
    """Names that can actually be fitted, for error messages and ``doctor``."""
    return tuple(sorted(ESTIMATORS))


# --- Helpers -----------------------------------------------------------------


def _lookup(name: str) -> type:
    """Resolve a method name to its estimator class."""
    if name in ESTIMATORS:
        return ESTIMATORS[name]
    if name in PENDING:
        raise NotImplementedError(f"method {name!r}: {PENDING[name]}")
    raise ConfigError(
        f"unknown method {name!r}; registered methods are {list(available_methods())}"
    )


def _fit_baseline(context: EstimatorContext, options: dict) -> IsotropicBaseline:
    """Build the analytic baseline according to ``model.analytic.mode``.

    This one method is not configured by keyword arguments but by a mode,
    because ``from_dataset`` and ``manual`` are different constructors
    rather than different values of the same parameter.
    """
    mode = str(options.get("mode", "from_dataset"))

    if mode == "from_dataset":
        return IsotropicBaseline.fit(context)

    if mode == "manual":
        mean, std = options.get("mean"), options.get("std")
        if mean is None or std is None:
            raise ConfigError(
                "model.analytic.mode is 'manual', so model.analytic.mean and "
                "model.analytic.std must both be given, in MPa"
            )
        return IsotropicBaseline.manual(
            mean=float(mean),
            std=float(std),
            deviatoric=context.transform.deviatoric,
        )

    if mode == "off":
        raise ConfigError(
            "model.analytic.mode is 'off', but 'analytic' is listed in "
            "model.methods. Drop it from the list, or set the mode to "
            "'from_dataset' or 'manual'."
        )

    raise ConfigError(
        f"model.analytic.mode must be from_dataset, manual or off; got {mode!r}"
    )


def _checked(function: Callable, options: dict, name: str) -> dict:
    """Reject option keys the estimator does not accept.

    Silently ignoring an unrecognised key turns a misspelled parameter into
    a run that quietly used the default, which is indistinguishable from a
    correct run until someone checks the numbers.
    """
    accepted = set(inspect.signature(function).parameters) - {"cls", "context"}
    unknown = sorted(set(options) - accepted)

    if unknown:
        raise ConfigError(
            f"model.{name} does not accept {unknown}; "
            f"accepted keys are {sorted(accepted)}"
        )
    return options
