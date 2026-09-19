"""The estimator contract: one cloud, one space, several independent methods.

Ray marginalisation and the conditional slice share the yield-point cloud and
the space it lives in, and nothing else. Their parameters differ, their
diagnostics differ, and neither needs the other to produce an answer.

The legacy code did not reflect that. ``SliceChannel.from_ray`` built the
slice *out of* an already constructed ray model, so asking for the slice
alone still paid for the ray, and the two could not be reasoned about
separately. Here each method is a :class:`YieldEstimator` fitted over a
shared cloud, and ``methods: [kde_slice]`` computes exactly one.

The space transform is applied **once**, by the pipeline, before any
estimator sees the data. Estimators therefore receive vectors already in
model coordinates and never call :meth:`SpaceTransform.forward` themselves,
which removes the possibility of one method working in a deviatoric space
while another does not.

``gradient`` is optional and returns ``None`` for the kernel estimators. It
exists because a yield surface without a normal is of limited use in
plasticity: the flow rule needs ``df/dsigma``. A kernel mixture does not
supply one usefully, a trained network does, and this is the seam where the
latter plugs in without the evaluation code changing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from .transform import SpaceTransform


@dataclass
class EstimatorOutput:
    """What one estimator returns for a batch of query points.

    Parameters
    ----------
    probability : ndarray of shape (M,)
        Yield probability. ``NaN`` marks points the method declines to
        answer for, which is deliberate: a silent extrapolation is worse
        than an honest gap.
    spread : ndarray of shape (M,)
        Width of the estimated yield-radius distribution, MPa. The methods
        define this differently -- see each estimator -- so it is comparable
        only in order of magnitude across methods.
    covered : ndarray of shape (M,), bool
        Whether the query lies inside the region the dataset supports.
    diagnostics : dict of str to ndarray
        Per-point method-specific fields, such as ``eff_n`` for the slice.
    """

    probability: np.ndarray
    spread: np.ndarray
    covered: np.ndarray
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check that every array agrees on the number of points."""
        sizes = {
            len(self.probability),
            len(self.spread),
            len(self.covered),
            *(len(v) for v in self.diagnostics.values()),
        }
        if len(sizes) > 1:
            raise ValueError(f"estimator output arrays disagree on length: {sizes}")

    @property
    def n_points(self) -> int:
        """Number of query points."""
        return len(self.probability)

    def summary(self) -> dict[str, float]:
        """Aggregate statistics for the run report."""
        finite = np.isfinite(self.probability)
        return {
            "n_points": float(self.n_points),
            "n_valid": float(finite.sum()),
            "n_covered": float(self.covered.sum()),
            "p_mean": float(np.mean(self.probability[finite]))
            if finite.any()
            else float("nan"),
            "p_max": float(np.max(self.probability[finite])) if finite.any() else float("nan"),
            "spread_median": float(np.nanmedian(self.spread)),
        }


@runtime_checkable
class YieldEstimator(Protocol):
    """One data-driven estimator over a shared yield-point cloud."""

    name: str
    """Method identifier, matching the entries of ``model.methods``."""

    def evaluate(self, sigma_model: np.ndarray) -> EstimatorOutput:
        """Estimate yield probability for query points in model space.

        Parameters
        ----------
        sigma_model : ndarray of shape (M, 6)
            Query stresses already mapped through the space transform.

        Returns
        -------
        EstimatorOutput
        """
        ...

    def probability_at(self, direction: np.ndarray, radius: float) -> float:
        """Yield probability along one direction at one radius.

        Used for locus extraction and point probes, where the direction is
        given rather than derived from a stress state.
        """
        ...

    def yield_radius(self, direction: np.ndarray) -> float:
        """Most probable yield radius along a direction, MPa."""
        ...

    def to_params(self) -> dict:
        """Parameters to store in the model bundle."""
        ...

    def gradient(self, sigma_model: np.ndarray) -> np.ndarray | None:
        """Yield-surface normal, or ``None`` when the method cannot supply one."""
        ...


@dataclass
class EstimatorContext:
    """Everything an estimator is fitted against.

    Bundling these means an estimator is constructed from one object rather
    than from a config plus a cloud plus a transform that must be kept in
    step by hand.

    Parameters
    ----------
    sigma_model : ndarray of shape (N, 6)
        Yield points already in model coordinates.
    transform : SpaceTransform
        The mapping that produced them, carried along so estimators can
        record the space they were fitted in.
    weight : ndarray of shape (N,), optional
        Per-point weights, currently unused by the kernel estimators and
        reserved for weighted fitting.
    """

    sigma_model: np.ndarray
    transform: SpaceTransform
    weight: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate the cloud."""
        self.sigma_model = np.asarray(self.sigma_model, dtype=np.float64)
        if self.sigma_model.ndim != 2 or self.sigma_model.shape[1] != 6:
            raise ValueError(
                f"sigma_model must have shape (N, 6), got {self.sigma_model.shape}"
            )
        if len(self.sigma_model) < 2:
            raise ValueError("at least two yield points are required to fit a model")

    @property
    def n_points(self) -> int:
        """Number of yield points."""
        return len(self.sigma_model)
