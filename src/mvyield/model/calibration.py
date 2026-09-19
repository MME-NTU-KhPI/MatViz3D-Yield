"""Hyper-parameter selection, with provenance.

The two methods arrived in this codebase in different states. The slice had
a real criterion -- leave-one-out log-likelihood over a grid of ``kappa``.
The ray had ``build_reference_baseline.py``, which prints a table of mode
and effective sample size across ``power`` in (2, 4, 8, 16, 32) and leaves a
human to pick one by eye. That is a report, not a calibrator.

Keeping that asymmetry is fine for now. Hiding it is not: a reader of a
figure needs to know whether ``power = 8`` was cross-validated or chosen by
inspection, and that distinction disappears the moment a number is copied
into a config by hand. So both go through one interface, both record where
the value came from, and the ray implementation refuses ``mode: auto``
outright rather than inventing a criterion it does not have. When a
criterion for the ray weighting does appear, it replaces one registry entry
and no call site changes.

Two rules the implementations share:

* **Cross-validation splits by geometry, never by point.** Voxels from one
  RVE see nearly the same stress state, so a per-point split would place
  near-duplicates on both sides and select a parameter that memorises the
  training geometries. The split is fixed once at ingest and reused here.
* **The config states intent; the bundle records fact.** Calibration never
  rewrites a YAML file. The chosen value, the criterion, the searched grid
  and the timestamp go into the model bundle, and the command prints a line
  that can be pasted into a config if the user wants to pin it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .basis import subspace_basis, to_subspace
from .bundle import SPLIT_TRAIN, SPLIT_VAL, YieldPointSet
from .transform import SpaceTransform

EPS = 1e-12


class CalibrationError(RuntimeError):
    """Raised when a hyper-parameter cannot be selected automatically."""


@dataclass
class CalibrationResult:
    """The outcome of selecting one hyper-parameter.

    Parameters
    ----------
    value : float
        Selected value.
    source : str
        Criterion used: ``"manual"``, ``"cv_loglik"``, ``"frozen"``.
    objective : float, optional
        Score at the selected value; ``None`` when nothing was optimised.
    curve : dict of float to float, optional
        The full sweep, kept for the diagnostic plot. For a manual choice
        this is the sweep table the user would otherwise read in the
        console.
    diagnostics : dict
        Supporting numbers, such as effective sample size per candidate.
    """

    value: float
    source: str
    objective: float | None = None
    curve: dict[float, float] | None = None
    diagnostics: dict = field(default_factory=dict)

    def describe(self, name: str) -> str:
        """One-line summary for the console and the report."""
        if self.objective is None:
            return f"{name} = {self.value:g}  [{self.source}]"
        return f"{name} = {self.value:g}  [{self.source}, objective {self.objective:.4g}]"

    def as_config_line(self, name: str) -> str:
        """Return the line to paste into a case config to pin this value."""
        return f"{name}: {{mode: manual, value: {self.value:g}}}"


class Calibrator(Protocol):
    """Select one hyper-parameter for a data-driven estimator."""

    parameter: str
    """Name of the hyper-parameter this calibrator selects."""

    def calibrate(
        self,
        points: YieldPointSet,
        transform: SpaceTransform,
        grid: tuple[float, ...],
        refine: bool = False,
    ) -> CalibrationResult:
        """Search ``grid`` and return the selected value with provenance."""
        ...


@dataclass
class SweepReportCalibrator:
    """Report a sweep for the ray weighting, without selecting a value.

    This reproduces what ``build_reference_baseline.py`` did: it evaluates
    the estimated radius and the effective sample size across the candidate
    grid, in the standard directions, and returns the table.

    It deliberately does **not** return a value. There is no criterion for
    the ray exponent in this codebase, and inventing one here -- picking the
    largest ``power`` that keeps ``eff_n`` above a threshold, say -- would
    dress up a threshold chosen by hand as an optimisation. Better to say
    plainly that the choice is manual and show the numbers behind it.

    Parameters
    ----------
    min_eff_n : float
        Effective sample size below which a candidate is flagged as
        unusable in the report.
    n_directions : int
        Probe directions sampled from the cloud.
    """

    min_eff_n: float = 15.0
    n_directions: int = 32
    seed: int = 0

    parameter: str = "power"

    def calibrate(
        self,
        points: YieldPointSet,
        transform: SpaceTransform,
        grid: tuple[float, ...],
        refine: bool = False,
    ) -> CalibrationResult:
        """Produce the sweep table and refuse to pick a value.

        Raises
        ------
        CalibrationError
            Always. The message carries the sweep so the user can choose,
            and names the config key to set.
        """
        report = self.sweep(points, transform, grid)

        lines = [f"  {'power':>8s}  {'radius, MPa':>12s}  {'eff_n':>8s}  {'usable':>7s}"]
        for candidate in grid:
            row = report[candidate]
            usable = "yes" if row["eff_n_median"] >= self.min_eff_n else "NO"
            lines.append(
                f"  {candidate:8.3g}  {row['radius_median']:12.1f}  "
                f"{row['eff_n_median']:8.1f}  {usable:>7s}"
            )

        raise CalibrationError(
            "the ray weighting exponent has no automatic criterion in this "
            "build, so 'mode: auto' cannot be honoured. The sweep below is "
            "the same information the legacy reference-baseline script "
            "printed; choose a value and set it explicitly:\n"
            "  model.kde_ray.power: {mode: manual, value: <chosen>}\n\n" + "\n".join(lines)
        )

    def sweep(
        self,
        points: YieldPointSet,
        transform: SpaceTransform,
        grid: tuple[float, ...],
    ) -> dict[float, dict[str, float]]:
        """Evaluate radius and effective sample size across the grid."""
        sigma = transform.forward(points.sigma6)
        radii = np.linalg.norm(sigma, axis=1)
        valid = radii > 1e-9

        unit = np.zeros_like(sigma)
        unit[valid] = sigma[valid] / radii[valid, None]

        rng = np.random.default_rng(self.seed)
        count = min(self.n_directions, int(valid.sum()))
        probes = unit[valid][rng.choice(int(valid.sum()), size=count, replace=False)]

        cosine = np.clip(probes @ unit.T, 0.0, None)

        report: dict[float, dict[str, float]] = {}
        for candidate in grid:
            weights = cosine**candidate
            weight_sum = weights.sum(axis=1)
            eff_n = weight_sum**2 / np.maximum((weights**2).sum(axis=1), 1e-30)

            safe = np.where(weight_sum > EPS, weight_sum, 1.0)
            radius = (weights @ radii) / safe

            report[candidate] = {
                "radius_median": float(np.median(radius)),
                "eff_n_median": float(np.median(eff_n)),
                "eff_n_p05": float(np.percentile(eff_n, 5)),
            }
        return report


@dataclass
class PolarLogLikCalibrator:
    """Select the slice concentration by held-out log-likelihood.

    Scores each candidate ``kappa`` by the log-likelihood of validation
    radii under the conditional density built from training points only::

        w_j   = exp(kappa (cos theta_ij - 1))
        m_i   = weighted mean radius
        tau_i = weighted standard deviation, floored
        score = mean over i of  -0.5 ((r_i - m_i)/tau_i)^2 - log(tau_i)

    The legacy script did this leave-one-out over the whole cloud. Held-out
    scoring by geometry is used here instead, for the reason given in the
    module docstring: leave-one-out on correlated voxels understates the
    error, and with an existing grouped split the honest version costs
    nothing extra.

    Parameters
    ----------
    tau_floor : float
        Lower bound on the radial width, MPa.
    max_train, max_eval : int
        Caps on the number of points used, since the scoring matrix is
        ``n_eval x n_train``. The estimate is a median over directions and
        converges quickly, so a full cloud is rarely needed.
    """

    tau_floor: float = 2.0
    max_train: int = 4000
    max_eval: int = 1500
    seed: int = 0

    parameter: str = "kappa"

    def calibrate(
        self,
        points: YieldPointSet,
        transform: SpaceTransform,
        grid: tuple[float, ...],
        refine: bool = False,
    ) -> CalibrationResult:
        """Search the grid, optionally refining around the best candidate."""
        train_radii, train_unit, eval_radii, eval_unit = self._prepare(points, transform)

        curve = {
            float(candidate): self._score(
                candidate, train_radii, train_unit, eval_radii, eval_unit
            )
            for candidate in grid
        }

        best = max(curve, key=lambda k: curve[k])

        if refine:
            neighbours = sorted(grid)
            position = neighbours.index(best)
            low = neighbours[max(position - 1, 0)]
            high = neighbours[min(position + 1, len(neighbours) - 1)]
            if high > low:
                for candidate in np.geomspace(low, high, 7)[1:-1]:
                    curve[float(candidate)] = self._score(
                        float(candidate), train_radii, train_unit, eval_radii, eval_unit
                    )
                best = max(curve, key=lambda k: curve[k])

        if not np.isfinite(curve[best]):
            raise CalibrationError(
                "every candidate scored non-finite; the cloud may be degenerate "
                "or the grid may be far from a usable range"
            )

        if best in (min(grid), max(grid)) and len(grid) > 1:
            edge = "lower" if best == min(grid) else "upper"
            note = f"selected value sits at the {edge} edge of the search grid; widen it"
        else:
            note = ""

        return CalibrationResult(
            value=float(best),
            source="cv_loglik",
            objective=float(curve[best]),
            curve=curve,
            diagnostics={
                "n_train": len(train_radii),
                "n_eval": len(eval_radii),
                "split": "by group_id",
                "note": note,
            },
        )

    def _prepare(self, points: YieldPointSet, transform: SpaceTransform):
        """Build training and evaluation sets from the grouped split."""
        basis = subspace_basis(transform.deviatoric, transform.convention)

        if points.split is None:
            raise CalibrationError(
                "the yield-point set carries no split; calibration needs held-out "
                "geometries. Re-run 'mvy ingest' so the artefact records one."
            )

        rng = np.random.default_rng(self.seed)
        train = self._subset(points, SPLIT_TRAIN, basis, transform, self.max_train, rng)
        evaluate = self._subset(points, SPLIT_VAL, basis, transform, self.max_eval, rng)

        if len(train) < 10 or len(evaluate) < 10:
            raise CalibrationError(
                f"too few points after splitting ({len(train)} train, "
                f"{len(evaluate)} validation) to score a hyper-parameter"
            )

        train_radii = np.linalg.norm(train, axis=1)
        eval_radii = np.linalg.norm(evaluate, axis=1)
        return (
            train_radii,
            train / np.maximum(train_radii[:, None], EPS),
            eval_radii,
            evaluate / np.maximum(eval_radii[:, None], EPS),
        )

    @staticmethod
    def _subset(points, which, basis, transform, cap, rng) -> np.ndarray:
        """Project one split partition into subspace coordinates, capped."""
        mask = points.split == which
        sigma = transform.forward(points.sigma6[mask])
        y = to_subspace(sigma, basis)
        if len(y) > cap:
            y = y[rng.choice(len(y), size=cap, replace=False)]
        return y

    def _score(
        self,
        kappa: float,
        train_radii: np.ndarray,
        train_unit: np.ndarray,
        eval_radii: np.ndarray,
        eval_unit: np.ndarray,
    ) -> float:
        """Mean held-out log-likelihood of the radius for one ``kappa``."""
        cosine = np.clip(eval_unit @ train_unit.T, -1.0, 1.0)
        log_weights = kappa * (cosine - 1.0)

        weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)

        mean = weights @ train_radii
        variance = (weights * (train_radii[None, :] - mean[:, None]) ** 2).sum(axis=1)

        local_eff_n = 1.0 / np.maximum((weights**2).sum(axis=1), EPS)
        correction = np.where(
            local_eff_n > 1.5, local_eff_n / np.maximum(local_eff_n - 1.0, EPS), 1.0
        )
        tau = np.maximum(np.sqrt(np.maximum(variance * correction, 0.0)), self.tau_floor)

        score = -0.5 * ((eval_radii - mean) / tau) ** 2 - np.log(tau)
        return float(np.mean(score))


CALIBRATORS: dict[str, type] = {
    "power": SweepReportCalibrator,
    "kappa": PolarLogLikCalibrator,
}
"""Which calibrator handles which hyper-parameter."""


def get_calibrator(parameter: str, **kwargs) -> Calibrator:
    """Build the calibrator registered for a hyper-parameter."""
    try:
        return CALIBRATORS[parameter](**kwargs)
    except KeyError:
        raise KeyError(
            f"no calibrator registered for {parameter!r}; known: {sorted(CALIBRATORS)}"
        ) from None
