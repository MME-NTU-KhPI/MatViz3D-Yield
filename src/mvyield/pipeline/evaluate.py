"""Evaluate a stress field against every requested method.

The space transform is applied **once**, to the whole field, before any
estimator runs. Estimators then work on prepared coordinates and cannot
disagree about the space. In the legacy code each method transformed the
field itself, which kept them consistent only as long as both read the same
module-level config.

Results land in a :class:`ResultBundle` that carries the field alongside
every method's output. That is what lets figures be redrawn later without
recomputing anything: the expensive part is here, and the plotting stage
only reads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mvyield.mechanics.field import StressField
from mvyield.model.base import EstimatorOutput
from mvyield.model.bundle import ModelBundle
from mvyield.model.registry import load_estimator

log = logging.getLogger(__name__)


@dataclass
class ResultBundle:
    """Everything one evaluation produced.

    Parameters
    ----------
    field : StressField
        The evaluated field.
    outputs : dict of str to EstimatorOutput
        One entry per method that ran.
    primary : str
        Method used for headline decisions.
    model_summary : dict
        Copy of the model's diagnostics and hyper-parameter provenance, so a
        stored result is interpretable without the bundle beside it.
    """

    field: StressField
    outputs: dict[str, EstimatorOutput]
    primary: str
    model_summary: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check that the primary method actually ran."""
        if self.primary not in self.outputs:
            raise ValueError(
                f"primary method {self.primary!r} produced no output; "
                f"available: {sorted(self.outputs)}"
            )

    @property
    def methods(self) -> tuple[str, ...]:
        """Methods present, primary first so reports lead with it."""
        others = sorted(name for name in self.outputs if name != self.primary)
        return (self.primary, *others)

    @property
    def runs_comparison(self) -> bool:
        """Whether a method comparison is meaningful for this result."""
        return len(self.outputs) >= 2

    def probability(self, method: str | None = None) -> np.ndarray:
        """Probability field for a method, defaulting to the primary one."""
        return self.outputs[method or self.primary].probability

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-method aggregate statistics for the report."""
        return {name: self.outputs[name].summary() for name in self.methods}

    def save(self, path: Path) -> Path:
        """Write results so figures can be redrawn without recomputing."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {
            "points": self.field.points,
            "sigma6": self.field.sigma6,
            "valid": self.field.valid_mask(),
        }

        # The presentation hints travel with the arrays so that redrawing is
        # faithful. Without the reference length a probe path given in hole
        # radii cannot be rebuilt, and the figure that uses it silently
        # becomes a different figure.
        hints = self.field.hints
        if hints.length_scale is not None:
            arrays["hints/length_scale"] = np.asarray([hints.length_scale])
        for index, outline in enumerate(hints.outlines):
            arrays[f"hints/outline/{index}/points"] = np.asarray(outline.points)
            if outline.radius is not None:
                arrays[f"hints/outline/{index}/radius"] = np.asarray([outline.radius])
        for name in self.methods:
            output = self.outputs[name]
            arrays[f"{name}/probability"] = output.probability
            arrays[f"{name}/spread"] = output.spread
            arrays[f"{name}/covered"] = output.covered
            for key, values in output.diagnostics.items():
                arrays[f"{name}/diag/{key}"] = values

        np.savez_compressed(path, **arrays)
        return path


def evaluate_field(
    field: StressField,
    bundle: ModelBundle,
    methods: tuple[str, ...] | None = None,
) -> ResultBundle:
    """Evaluate a stress field with every requested method.

    Parameters
    ----------
    field : StressField
        Raw stresses from a case, in the field's own convention.
    bundle : ModelBundle
        Fitted model carrying the cloud, the space and the parameters.
    methods : tuple of str, optional
        Subset of the bundle's methods to run. Defaults to all of them.

    Returns
    -------
    ResultBundle
    """
    requested = tuple(methods) if methods else bundle.methods

    unknown = [name for name in requested if name not in bundle.methods]
    if unknown:
        raise ValueError(
            f"the model bundle carries no parameters for {unknown}; it was "
            f"built with {list(bundle.methods)}. Rebuild it with "
            f"'mvy build-model' if you need those methods."
        )

    _check_scale(field, bundle)

    # One transform, applied once, shared by every estimator.
    sigma_model = bundle.transform.forward(field.sigma6, source=field.meta.convention)
    cloud_model = bundle.transform.forward(bundle.points.sigma6)

    outputs: dict[str, EstimatorOutput] = {}
    for name in requested:
        log.info("evaluating %s over %d points", name, field.n_points)
        estimator = load_estimator(name, cloud_model, bundle.params.get(name, {}))
        outputs[name] = estimator.evaluate(sigma_model)

    primary = bundle.primary if bundle.primary in outputs else requested[0]

    return ResultBundle(
        field=field,
        outputs=outputs,
        primary=primary,
        model_summary={
            "hyperparams": bundle.hyperparams,
            "diagnostics": bundle.diagnostics,
            "space": bundle.transform.to_dict(),
        },
    )


def _check_scale(field: StressField, bundle: ModelBundle) -> None:
    """Warn when field and model magnitudes are implausibly far apart.

    A unit mismatch -- an ANSYS export left in Pa against a model in MPa --
    produces a field of zeros or ones rather than an error, which is exactly
    the kind of failure that survives to a figure. Comparing medians catches
    it in one line, and a warning rather than an exception is right here:
    an unusual but deliberate loading case should not be blocked.
    """
    field_median = float(np.median(np.linalg.norm(field.sigma6, axis=1)))
    cloud_median = float(np.median(np.linalg.norm(bundle.points.sigma6, axis=1)))

    if field_median <= 0.0 or cloud_median <= 0.0:
        return

    ratio = field_median / cloud_median
    if not 0.01 < ratio < 100.0:
        log.warning(
            "field and model magnitudes differ by a factor of %.3g "
            "(field median %.4g, model median %.4g %s). Check field.units -- "
            "a Pa/MPa mix-up looks exactly like this.",
            ratio,
            field_median,
            cloud_median,
            field.meta.stress_unit,
        )
