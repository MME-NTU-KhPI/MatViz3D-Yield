"""Isotropic Gaussian baseline: the demonstration channel.

The classical treatment assumes a single scalar yield strength scattered
normally, and compares it against the von Mises equivalent stress::

    P(G) = Phi((sigma_vm - mu) / sigma)

This is not a data-driven result and is not presented as one. It exists as a
contrast: the whole point of the two kernel methods is that a yield surface
has *shape*, and the cheapest way to show that is to put the shapeless model
beside them.

In the legacy config the two numbers were literals, copied by hand from a
paper. That is the wrong place for them -- change the dataset and they
silently describe a different material. Here the default estimates them from
the point cloud itself, so the baseline always refers to the same material
as the methods it is being compared against:

``from_dataset`` (default)
    ``mu`` and ``sigma`` are the mean and standard deviation of the yield
    radius over the cloud.
``manual``
    Values supplied in the config, for reproducing a published figure.
``off``
    The channel is not computed and does not appear in any plot.

Because the estimate comes from the cloud, plots label it explicitly as an
isotropic baseline fitted to N points, so no reader mistakes it for the
data-driven result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm as _norm

from mvyield.mechanics import voigt

from .base import EstimatorContext, EstimatorOutput

EPS = 1e-9


@dataclass
class IsotropicBaseline:
    """Scalar yield strength with normal scatter.

    Parameters
    ----------
    mean : float
        Mean yield radius, MPa.
    std : float
        Standard deviation of the yield radius, MPa.
    source : str
        ``"from_dataset"`` or ``"manual"``, recorded so a figure caption can
        state where the numbers came from.
    n_points : int
        Size of the cloud the estimate came from; zero for manual values.
    deviatoric : bool
        Whether the model space dropped the hydrostatic part. Decides how
        the query magnitude is measured -- see :meth:`evaluate`.
    """

    mean: float
    std: float
    source: str = "from_dataset"
    n_points: int = 0
    deviatoric: bool = True

    name: str = "analytic"

    def __post_init__(self) -> None:
        """Validate the scatter."""
        if self.std <= 0.0:
            raise ValueError(f"baseline std must be positive, got {self.std}")

    @classmethod
    def fit(cls, context: EstimatorContext) -> IsotropicBaseline:
        """Estimate the mean and scatter from the yield-point cloud."""
        magnitude = cls._magnitude(context.sigma_model, context.transform.deviatoric)
        mean = float(np.mean(magnitude))
        std = float(np.std(magnitude))

        if std < EPS:
            raise ValueError(
                "the yield-point cloud has no scatter in magnitude, so an "
                "isotropic baseline is undefined; set analytic.mode to 'manual' "
                "or 'off'"
            )
        return cls(
            mean=mean,
            std=std,
            source="from_dataset",
            n_points=context.n_points,
            deviatoric=context.transform.deviatoric,
        )

    @classmethod
    def manual(cls, mean: float, std: float, deviatoric: bool = True) -> IsotropicBaseline:
        """Build from values supplied in the config."""
        return cls(mean=float(mean), std=float(std), source="manual", deviatoric=deviatoric)

    @classmethod
    def from_params(cls, params: dict) -> IsotropicBaseline:
        """Rebuild from stored bundle parameters."""
        return cls(
            mean=float(params["mean"]),
            std=float(params["std"]),
            source=str(params.get("source", "from_dataset")),
            n_points=int(params.get("n_points", 0)),
            deviatoric=bool(params.get("deviatoric", True)),
        )

    def to_params(self) -> dict:
        """Parameters to store in the model bundle."""
        return {
            "mean": self.mean,
            "std": self.std,
            "source": self.source,
            "n_points": self.n_points,
            "deviatoric": self.deviatoric,
        }

    def evaluate(self, sigma_model: np.ndarray) -> EstimatorOutput:
        """Evaluate the baseline over a field.

        Notes
        -----
        Coverage is reported as ``True`` everywhere. The baseline has no
        notion of dataset support -- it extrapolates cheerfully in every
        direction, which is exactly the limitation the comparison is meant
        to expose. Reporting a coverage mask here would suggest a
        discrimination the model does not have.
        """
        sigma_model = np.asarray(sigma_model, dtype=np.float64)
        magnitude = self._magnitude(sigma_model, self.deviatoric)

        probability = _norm.cdf((magnitude - self.mean) / self.std)
        spread = np.full(len(sigma_model), self.std, dtype=np.float64)
        covered = np.ones(len(sigma_model), dtype=bool)

        return EstimatorOutput(probability=probability, spread=spread, covered=covered)

    def probability_at(self, direction: np.ndarray, radius: float) -> float:
        """Yield probability at a given radius.

        The direction is accepted and ignored: an isotropic model has the
        same answer everywhere, which is the property under examination.
        """
        return float(_norm.cdf((float(radius) - self.mean) / self.std))

    def yield_radius(self, direction: np.ndarray) -> float:
        """Most probable yield radius: the mean, in every direction."""
        return self.mean

    def gradient(self, sigma_model: np.ndarray) -> np.ndarray | None:
        """Not provided; the baseline is a scalar comparison, not a surface."""
        return None

    def describe(self) -> str:
        """Caption text stating where the numbers came from.

        The scatter is quoted as a coefficient of variation as well: a
        standard deviation in MPa cannot be compared against another
        material, and the CV is what the dataset summaries report.
        """
        spread = f"sigma = {self.std:.1f} MPa (CV {100.0 * self.std / self.mean:.2f} %)"
        space = "deviatoric radius" if self.deviatoric else "stress magnitude"

        if self.source == "manual":
            return f"Isotropic baseline on the {space}, mu = {self.mean:.1f} MPa, {spread}, manual"
        return (
            f"Isotropic baseline on the {space}, mu = {self.mean:.1f} MPa, {spread}, "
            f"estimated from {self.n_points} dataset points"
        )

    @staticmethod
    def _magnitude(sigma_model: np.ndarray, deviatoric: bool) -> np.ndarray:
        """Scalar magnitude of a stress state in the model space.

        In a deviatoric space the Euclidean norm already is the equivalent
        measure, up to the constant ``sqrt(2/3)`` that cancels between the
        query and the fitted mean. In a full space the norm would count
        hydrostatic pressure, which does not drive yielding, so the von
        Mises invariant is used instead.
        """
        if deviatoric:
            return np.linalg.norm(sigma_model, axis=-1)
        return voigt.von_mises(sigma_model, voigt.MANDEL_XY_LAST)
