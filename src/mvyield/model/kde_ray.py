"""Ray marginalisation: the yield radius as a marginal along a loading ray.

Two variants share this module because they are the same method with the
weighting switched on or off, not two methods.

Global projection (``localized=False``)
---------------------------------------
Integrates the kernel density over the hyperplane orthogonal to the query
direction. Every dataset point contributes with weight one, whether or not
it lies near the ray::

    s_i     = sigma_i . d
    sigma_d = sqrt(d.T @ H @ d)
    P(G)    = (1/N) * sum_i Phi((r_star - s_i) / sigma_d)

Closed form, so no numerical integration and no repeated Cholesky
factorisation. It is the textbook starting point and is kept for the paper,
but on a thin yield shell it behaves poorly: in five or six dimensions the
projections of a shell concentrate near zero, so the marginal describes the
whole cloud rather than the surface radius in that particular direction.

Angle-weighted projection (``localized=True``, the default)
-----------------------------------------------------------
Keeps only points whose own direction is close to the query direction, and
estimates the radius from *their own radii* rather than from their
projections::

    w_i    = max(0, cos(angle(sigma_i, d)))**power
    r_mean = weighted mean of ||sigma_i||
    r_std  = weighted standard deviation
    P(G)   = Phi((r_star - r_mean) / r_std)

``eff_n = (sum w)^2 / sum w^2`` is Kish's effective sample size: how many
points actually inform the estimate. Below ``min_eff_n`` the result is
``NaN`` rather than a number, since a confident-looking value from three
points is worse than an admitted gap.

The angular weight is a loose approximation of the weights the conditional
slice derives exactly -- see :mod:`mvyield.model.kde_slice`, which makes the
same idea rigorous with a von Mises-Fisher kernel and a dimensionless
concentration parameter.

All formulas are ported unchanged from ``solver/probability.py`` and
``material/kde_yield_surface.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm as _norm

from .base import EstimatorContext, EstimatorOutput

EPS = 1e-9
DEFAULT_P_LOW = 0.005
"""Density floor for coverage, matching the legacy coverage checker."""


@dataclass
class RayMarginalEstimator:
    """Yield probability by marginalising the density along a loading ray.

    Parameters
    ----------
    sigma_data : ndarray of shape (N, 6)
        Yield points in model coordinates.
    bandwidth_h : float
        Scott's scalar factor.
    bandwidth_matrix : ndarray of shape (6, 6)
        ``H = h**2 * Cov``.
    localized : bool
        Use the angle-weighted variant.
    power : float
        Exponent of the cosine weight. Higher means a narrower cone and a
        more local estimate, at the cost of effective sample size.
    min_eff_n : float
        Below this effective sample size the estimate is withheld.
    p_low : float
        Density floor used by the coverage test of the global variant.
    batch_size : int
        Query points per pass. Memory scales as ``batch_size * N``.
    """

    sigma_data: np.ndarray
    bandwidth_h: float
    bandwidth_matrix: np.ndarray
    localized: bool = True
    power: float = 8.0
    min_eff_n: float = 15.0
    p_low: float = DEFAULT_P_LOW
    batch_size: int = 1500

    name: str = "kde_ray"

    def __post_init__(self) -> None:
        """Cache the per-point radii used by the localized variant."""
        self.sigma_data = np.asarray(self.sigma_data, dtype=np.float64)
        self.bandwidth_matrix = np.asarray(self.bandwidth_matrix, dtype=np.float64)
        self._data_norms = np.linalg.norm(self.sigma_data, axis=1)
        self._safe_norms = np.where(self._data_norms > EPS, self._data_norms, 1.0)

    # --- Construction --------------------------------------------------------

    @classmethod
    def fit(
        cls,
        context: EstimatorContext,
        localized: bool = True,
        power: float = 8.0,
        min_eff_n: float = 15.0,
        p_low: float = DEFAULT_P_LOW,
        batch_size: int = 1500,
    ) -> RayMarginalEstimator:
        """Fit the bandwidth from the cloud using Scott's rule.

        Notes
        -----
        The exponent uses the ambient dimension six, matching the legacy
        build. Only ``d.T @ H @ d`` is ever evaluated, so the singular
        hydrostatic direction is harmless here -- it contributes zero
        variance and no inverse is taken. The slice estimator, which does
        invert, works in the subspace instead.
        """
        sigma = context.sigma_model
        n, dim = sigma.shape

        h = n ** (-1.0 / (dim + 4))
        covariance = np.cov(sigma, rowvar=False)
        matrix = (h**2) * covariance

        return cls(
            sigma_data=sigma,
            bandwidth_h=float(h),
            bandwidth_matrix=matrix,
            localized=localized,
            power=power,
            min_eff_n=min_eff_n,
            p_low=p_low,
            batch_size=batch_size,
        )

    @classmethod
    def from_params(cls, sigma_model: np.ndarray, params: dict) -> RayMarginalEstimator:
        """Rebuild from stored bundle parameters."""
        return cls(
            sigma_data=sigma_model,
            bandwidth_h=float(params["bandwidth_h"]),
            bandwidth_matrix=np.asarray(params["bandwidth_matrix"], dtype=np.float64),
            localized=bool(params.get("localized", True)),
            power=float(params.get("power", 8.0)),
            min_eff_n=float(params.get("min_eff_n", 15.0)),
            p_low=float(params.get("p_low", DEFAULT_P_LOW)),
            batch_size=int(params.get("batch_size", 1500)),
        )

    # --- Kernel geometry -----------------------------------------------------

    def angular_weight(self, angle_deg: np.ndarray) -> np.ndarray:
        """Weight given to a cloud point at this angle from the query.

        ``w = max(0, cos theta) ** power``. Exposed so that the console, the
        report and the diagnostic figure all state the same kernel rather
        than three copies of the same formula.
        """
        cosine = np.cos(np.deg2rad(np.asarray(angle_deg, dtype=np.float64)))
        return np.clip(cosine, 0.0, None) ** self.power

    def half_weight_angle(self) -> float:
        """Angle at which a neighbour counts half as much, degrees.

        ``power`` is dimensionless and says nothing about how much of the
        sphere is being averaged over. This does: it is the width of the
        cone the estimate actually borrows from, and it is what decides
        whether an anisotropic surface survives the averaging.
        """
        if self.power <= 0.0:
            return 90.0
        return float(np.degrees(np.arccos(0.5 ** (1.0 / self.power))))

    def describe(self) -> str:
        """One line stating the kernel in physical terms."""
        if not self.localized:
            return (
                f"global projection, no angular weighting; "
                f"Scott h = {self.bandwidth_h:.4f} in 6D"
            )
        return (
            f"cos^p weighting, p = {self.power:g} "
            f"(half weight at {self.half_weight_angle():.1f} deg); "
            f"Scott h = {self.bandwidth_h:.4f} in 6D; "
            f"min eff_n = {self.min_eff_n:g}"
        )

    def to_params(self) -> dict:
        """Parameters to store in the model bundle."""
        return {
            "bandwidth_h": self.bandwidth_h,
            "bandwidth_matrix": self.bandwidth_matrix,
            "localized": self.localized,
            "power": self.power,
            "min_eff_n": self.min_eff_n,
            "p_low": self.p_low,
            "batch_size": self.batch_size,
        }

    # --- Field evaluation ----------------------------------------------------

    def evaluate(self, sigma_model: np.ndarray) -> EstimatorOutput:
        """Evaluate a whole field in one pass.

        The expensive step is the projection of the dataset onto the batch
        of query directions. Probability, density and coverage all follow
        from the same projection, so it is computed once rather than once
        per quantity.
        """
        sigma_model = np.asarray(sigma_model, dtype=np.float64)
        if sigma_model.ndim != 2 or sigma_model.shape[1] != 6:
            raise ValueError(f"expected a field of shape (M, 6), got {sigma_model.shape}")

        return (
            self._evaluate_localized(sigma_model)
            if self.localized
            else self._evaluate_global(sigma_model)
        )

    def _evaluate_global(self, sigma_model: np.ndarray) -> EstimatorOutput:
        """Plain projection: every dataset point contributes equally."""
        n_points = len(sigma_model)
        radius = np.linalg.norm(sigma_model, axis=1)
        valid = radius > EPS

        probability = np.zeros(n_points, dtype=np.float64)
        spread = np.zeros(n_points, dtype=np.float64)
        covered = np.zeros(n_points, dtype=bool)

        for start, end, batch, batch_r, batch_valid in self._batches(
            sigma_model, radius, valid
        ):
            directions = self._unit_directions(batch, batch_r, batch_valid)

            projection = directions @ self.sigma_data.T
            scaled = directions @ self.bandwidth_matrix
            sigma_d = np.sqrt(np.maximum(np.einsum("bi,bi->b", scaled, directions), 1e-15))

            z = (batch_r[:, None] - projection) / sigma_d[:, None]
            batch_probability = np.mean(_norm.cdf(z), axis=1)
            density = np.mean(_norm.pdf(z), axis=1) / sigma_d

            in_range = (batch_r >= projection.min(axis=1)) & (
                batch_r <= projection.max(axis=1)
            )

            batch_probability[~batch_valid] = 0.0
            probability[start:end] = batch_probability
            spread[start:end] = sigma_d
            covered[start:end] = batch_valid & in_range & (density >= self.p_low)

        return EstimatorOutput(probability=probability, spread=spread, covered=covered)

    def _evaluate_localized(self, sigma_model: np.ndarray) -> EstimatorOutput:
        """Angle-weighted variant: only points near the ray contribute."""
        n_points = len(sigma_model)
        radius = np.linalg.norm(sigma_model, axis=1)
        valid = radius > EPS

        probability = np.full(n_points, np.nan, dtype=np.float64)
        spread = np.full(n_points, np.nan, dtype=np.float64)
        effective_n = np.zeros(n_points, dtype=np.float64)
        covered = np.zeros(n_points, dtype=bool)

        for start, end, batch, batch_r, batch_valid in self._batches(
            sigma_model, radius, valid
        ):
            directions = self._unit_directions(batch, batch_r, batch_valid)

            projection = directions @ self.sigma_data.T
            cosine = projection / self._safe_norms[None, :]
            weights = np.clip(cosine, 0.0, None) ** self.power

            weight_sum = weights.sum(axis=1)
            batch_eff_n = weight_sum**2 / np.maximum((weights**2).sum(axis=1), 1e-30)

            safe_sum = np.where(weight_sum > EPS, weight_sum, 1.0)
            r_mean = (weights @ self._data_norms) / safe_sum
            r_var = (weights @ (self._data_norms**2)) / safe_sum - r_mean**2
            r_std = np.sqrt(np.maximum(r_var, 1e-6))

            batch_probability = _norm.cdf((batch_r - r_mean) / r_std)
            batch_covered = batch_valid & (weight_sum > EPS) & (batch_eff_n >= self.min_eff_n)

            probability[start:end] = np.where(batch_covered, batch_probability, np.nan)
            spread[start:end] = r_std
            effective_n[start:end] = batch_eff_n
            covered[start:end] = batch_covered

        return EstimatorOutput(
            probability=probability,
            spread=spread,
            covered=covered,
            diagnostics={"eff_n": effective_n},
        )

    # --- Single-direction queries -------------------------------------------

    def probability_at(self, direction: np.ndarray, radius: float) -> float:
        """Yield probability along one direction at one radius."""
        direction = self._unit(direction)
        if radius < EPS:
            return 0.0

        if not self.localized:
            projection = self.sigma_data @ direction
            return float(np.mean(_norm.cdf((radius - projection) / self._sigma_d(direction))))

        weights = self._angular_weights(direction)
        if weights.sum() < EPS:
            return float("nan")
        r_mean = np.average(self._data_norms, weights=weights)
        r_std = max(
            float(np.sqrt(np.average((self._data_norms - r_mean) ** 2, weights=weights))),
            1e-6,
        )
        return float(_norm.cdf((radius - r_mean) / r_std))

    def yield_radius(self, direction: np.ndarray, n_grid: int = 500) -> float:
        """Most probable yield radius along a direction, MPa.

        The localized variant returns the weighted mean radius; the global
        variant returns the mode of the marginal density, found on a grid.
        """
        direction = self._unit(direction)

        if self.localized:
            weights = self._angular_weights(direction)
            eff_n = weights.sum() ** 2 / max(float((weights**2).sum()), 1e-30)
            if eff_n < self.min_eff_n:
                raise ValueError(
                    f"too few dataset points near this direction "
                    f"(eff_n = {eff_n:.1f} < {self.min_eff_n}); lower 'power' "
                    f"or check dataset coverage"
                )
            return float(np.average(self._data_norms, weights=weights))

        projection = self.sigma_data @ direction
        sigma_d = self._sigma_d(direction)
        grid = np.linspace(
            max(0.0, projection.min() - 3 * sigma_d),
            projection.max() + 3 * sigma_d,
            n_grid,
        )
        z = (grid[:, None] - projection[None, :]) / sigma_d
        density = np.mean(_norm.pdf(z), axis=1) / sigma_d
        return float(grid[int(np.argmax(density))])

    def marginal_std(self, direction: np.ndarray) -> float:
        """Width of the marginal distribution along a direction, MPa."""
        return self._sigma_d(self._unit(direction))

    def gradient(self, sigma_model: np.ndarray) -> np.ndarray | None:
        """Not available for a kernel mixture; returns ``None``.

        A finite-difference normal could be produced here, but it would be
        noisy and expensive, and presenting it alongside an analytic one
        would invite treating the two as equivalent. A method that has no
        useful gradient should say so.
        """
        return None

    # --- Internals -----------------------------------------------------------

    def _batches(self, field: np.ndarray, radius: np.ndarray, valid: np.ndarray):
        """Iterate over query batches, skipping those with no valid point."""
        for start in range(0, len(field), self.batch_size):
            end = min(start + self.batch_size, len(field))
            batch_valid = valid[start:end]
            if not np.any(batch_valid):
                continue
            yield start, end, field[start:end], radius[start:end], batch_valid

    @staticmethod
    def _unit_directions(
        batch: np.ndarray, radius: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        """Return unit directions for a batch, leaving null rows at zero."""
        directions = np.zeros_like(batch)
        safe = np.where(radius > EPS, radius, 1.0)
        directions[valid] = batch[valid] / safe[valid, None]
        return directions

    @staticmethod
    def _unit(direction: np.ndarray) -> np.ndarray:
        """Normalise a direction already expressed in model space."""
        direction = np.asarray(direction, dtype=np.float64).ravel()
        magnitude = float(np.linalg.norm(direction))
        if magnitude < EPS:
            raise ValueError("direction vector is null")
        return direction / magnitude

    def _sigma_d(self, direction: np.ndarray) -> float:
        """``sqrt(d.T @ H @ d)``, the marginal width along a direction."""
        value = float(direction @ self.bandwidth_matrix @ direction)
        return float(np.sqrt(max(value, 1e-15)))

    def _angular_weights(self, direction: np.ndarray) -> np.ndarray:
        """Cosine weights of every dataset point relative to a direction."""
        cosine = np.zeros(len(self.sigma_data))
        valid = self._data_norms > EPS
        cosine[valid] = (self.sigma_data[valid] @ direction) / self._data_norms[valid]
        return np.clip(cosine, 0.0, None) ** self.power

    def local_radii(self, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return dataset radii and their angular weights, for plotting."""
        direction = self._unit(direction)
        return self._data_norms, self._angular_weights(direction)
