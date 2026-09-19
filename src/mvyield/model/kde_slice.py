"""Conditional slice: the density of the yield radius on the ray itself.

Marginalisation against slicing
-------------------------------
Marginalisation integrates the density over the hyperplane orthogonal to the
query direction, so every point contributes with weight one. Slicing
evaluates the density *on* the ray ``sigma = r * u``. With ``A = H^-1`` and

    a = u.T A u,   b_i = u.T A y_i,   c_i = y_i.T A y_i

expanding the quadratic form gives

    -0.5 (a r^2 - 2 b_i r + c_i) = -(a/2)(r - b_i/a)^2 - 0.5 (c_i - b_i^2/a)

which is closed-form and no more expensive than the marginal::

    p_slice(r|u) ~ sum_i w_i phi((r - mu_i)/tau) / tau
    mu_i = b_i / a           centre of point i's contribution on the ray
    tau  = 1 / sqrt(a)       slice width, shared by all points
    w_i  = exp(-delta_i^2/2) with delta_i^2 = c_i - b_i^2/a

``delta_i`` is the Mahalanobis distance from the point to the line. So the
two methods differ in exactly two things: weights instead of ones, and
``tau`` instead of ``sigma_d``. Cauchy-Schwarz gives ``tau <= sigma_d``
always, so marginalisation systematically inflates the uncertainty. The
``cos**power`` weighting of the ray estimator is a loose approximation of
these ``w_i``.

Truncation at r >= 0
--------------------
``delta_i`` measures distance to the *line*, not the ray, so a point on the
far side of the cloud is weighted like a neighbour. On a synthetic shell 57%
of the weight fell on the antipodal branch at ``mu_i ~ -313 MPa``, and the
integral over ``[0, inf)`` came to 0.43 instead of 1. Physically the yield
radius is positive and the ``-u`` branch is a different loading state --
tension against compression, an asymmetry that is real for BCC and must not
be mixed. The mixture is therefore truncated at ``r >= 0`` and renormalised
in closed form::

    Z     = sum_i w_i Phi(mu_i/tau)
    F(r*) = sum_i w_i [Phi((r*-mu_i)/tau) - Phi(-mu_i/tau)] / Z

The ray method never sees the second branch separately; it simply blends it
in.

Bandwidth modes
---------------
Scott's rule calibrates for a filled cloud, but the data lie on a thin shell
-- roughly 10 MPa thick at a radius near 275 MPa. The kernel comes out wider
than the shell and the slice degenerates back into a projection. Shrinking
an isotropic kernel does not help, because it narrows radially and angularly
at once: on synthetic data ``scale = 0.3`` already drops the effective
sample size to 2.6.

``global``
    ``H_eff = scale**2 * H``, the textbook slice. Needed as a reference
    point, poor on a shell.
``adapted``
    ``H_eff(u) = sigma_r**2 u u.T + sigma_t**2 (I - u u.T)``, giving
    ``tau = sigma_r`` radially and a separate angular width ``sigma_t``.
``polar`` (default)
    The centre of a point's contribution is *its own radius* rather than its
    projection, so shell curvature no longer biases the estimate toward the
    centre. Angular distance enters only through a von Mises-Fisher weight
    ``w ~ exp(kappa (cos theta - 1))`` on the sphere. ``kappa`` is
    dimensionless and therefore transfers between materials and stress
    scales, which is what a width in MPa cannot do. ``tau`` is then
    estimated *from the data* as the weighted spread of radii -- the
    physical microstructural scatter, not a free knob.

Numerical care
--------------
``delta_i**2`` reaches hundreds of thousands, since most points lie far from
any given ray. A naive ``exp(-delta**2/2)`` underflows to zero and the
normalisation becomes 0/0, so the weights are handled in log space.

All formulas are ported unchanged from ``material/kde_slice_surface.py`` and
``solver/probability_slice.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm as _norm

from .base import EstimatorContext, EstimatorOutput
from .basis import subspace_basis, subspace_direction, to_subspace

EPS = 1e-12
INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)
DEFAULT_TOP_K = 4096
DEFAULT_P_LOW = 0.005

RANK_TOL = 1e-12
"""Relative floor on the smallest bandwidth eigenvalue.

A rank-deficient cloud almost never yields an exactly zero eigenvalue. A
deviatoric shell viewed in the ambient 6D space rounds to roughly ``1e-16``
of the largest eigenvalue -- positive, so a test against zero passes, the
inverse is computed anyway and comes back with entries of order ``1e11``.
Comparing against the largest eigenvalue instead catches that, while
leaving several orders of margin for a genuinely anisotropic dataset, whose
spread across directions stays within a factor of a few.
"""

BANDWIDTH_MODES = ("polar", "global", "adapted")


@dataclass
class SliceEstimator:
    """Yield probability from the conditional density on the loading ray.

    Parameters
    ----------
    y_data : ndarray of shape (N, k)
        Yield points in subspace coordinates.
    basis : ndarray of shape (6, k)
        Basis that produced them.
    mode : str
        One of :data:`BANDWIDTH_MODES`.
    kappa : float
        Concentration of the von Mises-Fisher weight, ``polar`` mode.
        Dimensionless.
    tau_floor : float
        Lower bound on the radial width, MPa. A safeguard against a
        degenerate weighted variance, not a model parameter.
    sigma_r, sigma_t : float, optional
        Radial and transverse widths for ``adapted`` mode, MPa.
    sigma_t_mode : str
        ``"absolute"`` or ``"angular"``. Angular normalises by the point's
        own radius, so the angular resolution does not vary between
        directions of different radius.
    inverse_bandwidth : ndarray, optional
        ``A = H^-1`` in subspace coordinates, ``global`` mode.
    scale : float
        Multiplier on the fitted width.
    top_k : int, optional
        Dataset points retained per query point, by weight. ``None`` keeps
        all, which is slower and used to verify the approximation.
    min_eff_n : float
        Effective sample size below which the estimate is flagged.
    censor : bool
        Whether a low effective sample size also clears ``covered``.
        Coverage and statistical support are different questions, so this is
        off by default: a noisy estimate still exists.
    batch_size : int
        Query points per pass.
    """

    y_data: np.ndarray
    basis: np.ndarray
    mode: str = "polar"
    kappa: float = 20.0
    tau_floor: float = 2.0
    sigma_r: float | None = None
    sigma_t: float | None = None
    sigma_t_mode: str = "angular"
    inverse_bandwidth: np.ndarray | None = None
    scale: float = 1.0
    top_k: int | None = DEFAULT_TOP_K
    min_eff_n: float = 15.0
    censor: bool = False
    p_low: float = DEFAULT_P_LOW
    batch_size: int = 400

    name: str = "kde_slice"
    _quadratic_c: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the mode and cache per-point norms."""
        if self.mode not in BANDWIDTH_MODES:
            raise ValueError(
                f"bandwidth mode must be one of {BANDWIDTH_MODES}; got {self.mode!r}"
            )

        self.y_data = np.ascontiguousarray(np.asarray(self.y_data, dtype=np.float64))
        self.basis = np.asarray(self.basis, dtype=np.float64)

        self._norm2 = np.einsum("ij,ij->i", self.y_data, self.y_data)
        self._norm = np.sqrt(np.maximum(self._norm2, EPS))

        if self.mode == "polar":
            if self.kappa <= 0:
                raise ValueError("kappa must be positive")
        elif self.mode == "adapted":
            if not self.sigma_r or not self.sigma_t:
                raise ValueError("adapted mode requires sigma_r and sigma_t")
            if self.sigma_t_mode not in ("absolute", "angular"):
                raise ValueError("sigma_t_mode must be 'absolute' or 'angular'")
        else:
            if self.inverse_bandwidth is None:
                raise ValueError("global mode requires the inverse bandwidth matrix")
            matrix = np.asarray(self.inverse_bandwidth, dtype=np.float64) / (self.scale**2)
            self.inverse_bandwidth = 0.5 * (matrix + matrix.T)
            self._quadratic_c = np.einsum(
                "ij,jk,ik->i", self.y_data, self.inverse_bandwidth, self.y_data
            )

    # --- Construction --------------------------------------------------------

    @classmethod
    def fit(
        cls,
        context: EstimatorContext,
        mode: str = "polar",
        kappa: float = 20.0,
        tau_floor: float | None = None,
        sigma_r: float | None = None,
        sigma_t: float | None = None,
        sigma_t_mode: str = "angular",
        scale: float = 1.0,
        top_k: int | None = DEFAULT_TOP_K,
        min_eff_n: float = 15.0,
        censor: bool = False,
        p_low: float = DEFAULT_P_LOW,
        batch_size: int = 400,
    ) -> SliceEstimator:
        """Fit the estimator in the subspace the model occupies.

        Notes
        -----
        Scott's rule is applied with the intrinsic dimension ``k``, not the
        ambient six: ``h = N ** (-1/(k+4))``. In the subspace the covariance
        has full rank, so the inverse is a genuine Cholesky solve rather
        than a pseudo-inverse -- which matters here, because the slice needs
        the inverse and a pseudo-inverse would silently ignore the
        degenerate direction instead of penalising it.
        """
        basis = subspace_basis(context.transform.deviatoric, context.transform.convention)
        y = to_subspace(context.sigma_model, basis)
        n, dim = y.shape

        if n <= dim:
            raise ValueError(
                f"cannot estimate a {dim}x{dim} covariance from {n} points; "
                f"the dataset is too small for the slice method"
            )

        h = n ** (-1.0 / (dim + 4))
        covariance = np.cov(y, rowvar=False)
        bandwidth = (h**2) * covariance

        eigenvalues = np.linalg.eigvalsh(bandwidth)
        if eigenvalues.min() <= RANK_TOL * max(eigenvalues.max(), EPS):
            raise np.linalg.LinAlgError(
                f"the bandwidth matrix is not positive definite "
                f"(eigenvalues span {eigenvalues.min():.3e} to "
                f"{eigenvalues.max():.3e}); the dataset is degenerate, "
                f"occupying fewer than {dim} dimensions -- check direction "
                f"coverage"
            )

        inverse = np.linalg.inv(bandwidth)
        inverse = 0.5 * (inverse + inverse.T)

        if tau_floor is None:
            tau_floor = 0.25 * _shell_thickness(y)

        return cls(
            y_data=y,
            basis=basis,
            mode=mode,
            kappa=kappa,
            tau_floor=float(tau_floor),
            sigma_r=sigma_r,
            sigma_t=sigma_t,
            sigma_t_mode=sigma_t_mode,
            inverse_bandwidth=inverse,
            scale=scale,
            top_k=top_k,
            min_eff_n=min_eff_n,
            censor=censor,
            p_low=p_low,
            batch_size=batch_size,
        )

    @classmethod
    def from_params(cls, sigma_model: np.ndarray, params: dict) -> SliceEstimator:
        """Rebuild from stored bundle parameters."""
        basis = np.asarray(params["basis"], dtype=np.float64)
        inverse = params.get("inverse_bandwidth")
        return cls(
            y_data=to_subspace(sigma_model, basis),
            basis=basis,
            mode=str(params.get("mode", "polar")),
            kappa=float(params.get("kappa", 20.0)),
            tau_floor=float(params.get("tau_floor", 2.0)),
            sigma_r=params.get("sigma_r"),
            sigma_t=params.get("sigma_t"),
            sigma_t_mode=str(params.get("sigma_t_mode", "angular")),
            inverse_bandwidth=None if inverse is None else np.asarray(inverse),
            scale=float(params.get("scale", 1.0)),
            top_k=params.get("top_k", DEFAULT_TOP_K),
            min_eff_n=float(params.get("min_eff_n", 15.0)),
            censor=bool(params.get("censor", False)),
            p_low=float(params.get("p_low", DEFAULT_P_LOW)),
            batch_size=int(params.get("batch_size", 400)),
        )

    # --- Kernel geometry -----------------------------------------------------

    def angular_weight(self, angle_deg: np.ndarray) -> np.ndarray:
        """Weight given to a cloud point at this angle from the query.

        ``w = exp(kappa (cos theta - 1))``, the von Mises-Fisher kernel
        normalised to one at zero. Exposed so the console, the report and
        the diagnostic figure state one kernel rather than three copies.
        """
        cosine = np.cos(np.deg2rad(np.asarray(angle_deg, dtype=np.float64)))
        return np.exp(self.kappa * (cosine - 1.0))

    def half_weight_angle(self) -> float:
        """Angle at which a neighbour counts half as much, degrees.

        ``kappa`` is dimensionless and transfers between materials, which is
        why it is the parameter; it is also unreadable as a width. Solving
        ``exp(kappa (cos theta - 1)) = 1/2`` gives the cone the estimate
        borrows from.

        The rule of thumb ``1/sqrt(kappa)`` radians, which the legacy config
        quotes, runs one to three degrees narrow over the useful range --
        close enough to reason with, not close enough to report. This is
        the solved value.
        """
        if self.kappa <= 0.0:
            return 90.0
        cosine = 1.0 + np.log(0.5) / self.kappa
        if cosine <= -1.0:
            return 180.0
        return float(np.degrees(np.arccos(cosine)))

    def describe(self) -> str:
        """One line stating the kernel in physical terms."""
        dimension = self.basis.shape[1]
        if self.mode != "polar":
            return f"{self.mode} bandwidth in {dimension}D; min eff_n = {self.min_eff_n:g}"
        return (
            f"von Mises-Fisher kernel, kappa = {self.kappa:g} "
            f"(half weight at {self.half_weight_angle():.1f} deg, "
            f"rule of thumb {np.degrees(1.0 / np.sqrt(self.kappa)):.1f} deg); "
            f"{dimension}D subspace; tau floor = {self.tau_floor:.2f} MPa; "
            f"top {self.top_k} neighbours"
        )

    def to_params(self) -> dict:
        """Parameters to store in the model bundle."""
        params = {
            "basis": self.basis,
            "mode": self.mode,
            "kappa": self.kappa,
            "tau_floor": self.tau_floor,
            "sigma_t_mode": self.sigma_t_mode,
            "scale": self.scale,
            "top_k": self.top_k,
            "min_eff_n": self.min_eff_n,
            "censor": self.censor,
            "p_low": self.p_low,
            "batch_size": self.batch_size,
        }
        if self.sigma_r is not None:
            params["sigma_r"] = self.sigma_r
        if self.sigma_t is not None:
            params["sigma_t"] = self.sigma_t
        if self.inverse_bandwidth is not None:
            params["inverse_bandwidth"] = self.inverse_bandwidth * (self.scale**2)
        return params

    # --- Field evaluation ----------------------------------------------------

    def evaluate(self, sigma_model: np.ndarray) -> EstimatorOutput:
        """Evaluate a whole field, batched over query points."""
        sigma_model = np.asarray(sigma_model, dtype=np.float64)
        if sigma_model.ndim != 2 or sigma_model.shape[1] != 6:
            raise ValueError(f"expected a field of shape (M, 6), got {sigma_model.shape}")

        y_field = to_subspace(sigma_model, self.basis)
        n_points = len(y_field)
        radius = np.linalg.norm(y_field, axis=1)
        valid = radius > 1e-9

        probability = np.full(n_points, np.nan, dtype=np.float64)
        spread = np.full(n_points, np.nan, dtype=np.float64)
        effective_n = np.zeros(n_points, dtype=np.float64)
        covered = np.zeros(n_points, dtype=bool)

        for start in range(0, n_points, self.batch_size):
            end = min(start + self.batch_size, n_points)
            batch_valid = valid[start:end]
            if not np.any(batch_valid):
                continue

            batch_r = radius[start:end]
            safe_r = np.where(batch_r > 1e-9, batch_r, 1.0)
            directions = y_field[start:end] / safe_r[:, None]

            centres, tau, log_weights = self._ray_geometry_batch(directions)
            centres, log_weights = self._select_top_k(centres, log_weights)

            log_weights = log_weights - log_weights.max(axis=1, keepdims=True)
            weights = np.exp(log_weights)
            tau_column = tau[:, None]

            lower = ndtr(-centres / tau_column)
            upper = ndtr((batch_r[:, None] - centres) / tau_column)

            weight_sum = weights.sum(axis=1)
            batch_eff_n = weight_sum**2 / np.maximum((weights * weights).sum(axis=1), 1e-300)

            truncated = weights * (1.0 - lower)
            mass = truncated.sum(axis=1)
            numerator = (weights * (upper - lower)).sum(axis=1)
            safe_mass = np.maximum(mass, EPS)

            z = (batch_r[:, None] - centres) / tau_column
            density = (weights * (INV_SQRT_2PI * np.exp(-0.5 * z * z)) / tau_column).sum(
                axis=1
            )
            density /= safe_mass

            # Moments use the truncated weights: including the antipodal
            # branch, centred near -r, would inflate the variance by
            # hundreds of MPa. Residual error from truncation inside a
            # single component stays below 1e-3 MPa, because significant
            # points have mu/tau well above zero.
            first = (truncated * centres).sum(axis=1) / safe_mass
            second = (truncated * centres * centres).sum(axis=1) / safe_mass
            batch_spread = np.sqrt(tau**2 + np.maximum(second - first * first, 0.0))

            significant = (centres > 0.0) & (
                weights > weights.max(axis=1, keepdims=True) * 1e-3
            )
            centre_low = np.where(significant, centres, np.inf).min(axis=1)
            centre_high = np.where(significant, centres, -np.inf).max(axis=1)
            in_range = (batch_r >= centre_low) & (batch_r <= centre_high)

            batch_probability = np.clip(
                np.where(mass > EPS, numerator / safe_mass, np.nan), 0.0, 1.0
            )
            batch_covered = batch_valid & in_range & (density >= self.p_low)
            if self.censor:
                batch_covered = batch_covered & (batch_eff_n >= self.min_eff_n)

            probability[start:end] = np.where(batch_valid, batch_probability, np.nan)
            spread[start:end] = np.where(batch_valid, batch_spread, np.nan)
            effective_n[start:end] = np.where(batch_valid, batch_eff_n, 0.0)
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
        if radius < 1e-9:
            return 0.0

        centres, tau, log_weights = self.ray_geometry(direction)
        weights = np.exp(log_weights - _logsumexp(log_weights))

        lower = _norm.cdf(-centres / tau)
        mass = float(np.dot(weights, 1.0 - lower))
        if mass <= EPS:
            return float("nan")

        numerator = float(np.dot(weights, _norm.cdf((radius - centres) / tau) - lower))
        return float(np.clip(numerator / mass, 0.0, 1.0))

    def yield_radius(self, direction: np.ndarray, n_grid: int = 512) -> float:
        """Most probable yield radius along a direction, MPa."""
        centres, tau, log_weights = self.ray_geometry(direction)
        grid = self._radius_grid(centres, tau, log_weights, n_grid)
        density = self._mixture_density(grid, centres, tau, log_weights)
        return float(grid[int(np.argmax(density))])

    def effective_n(self, direction: np.ndarray) -> float:
        """Kish effective sample size along a direction.

        Approaching ``N`` means the bandwidth is too wide and angular
        resolution has been lost. A small value means the estimate is noisy
        -- a statement about confidence, not a reason to withhold a result:
        in polar mode the conditional density is defined everywhere, since
        weights decay smoothly and a nearest point always exists.
        """
        _, _, log_weights = self.ray_geometry(direction)
        weights = np.exp(log_weights - _logsumexp(log_weights))
        return float(1.0 / max(float(np.sum(weights**2)), EPS))

    def marginal_std(self, direction: np.ndarray) -> float:
        """Kernel width ``tau`` along a direction, MPa."""
        _, tau, _ = self.ray_geometry(direction)
        return float(tau)

    def ray_geometry(self, direction: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Return contribution centres, slice width and log-weights.

        Parameters
        ----------
        direction : ndarray
            Either an ambient 6D vector or subspace coordinates.

        Returns
        -------
        centres : ndarray of shape (N,)
            Where each dataset point places its contribution on the ray, MPa.
        tau : float
            Slice width, MPa.
        log_weights : ndarray of shape (N,)
            Unnormalised log-weights, shifted so the maximum is zero.
        """
        unit = subspace_direction(direction, self.basis)
        centres, tau, log_weights = self._ray_geometry_batch(unit[None, :])
        return centres[0], float(tau[0]), log_weights[0]

    def gradient(self, sigma_model: np.ndarray) -> np.ndarray | None:
        """Not available for a kernel mixture; returns ``None``."""
        return None

    # --- Internals -----------------------------------------------------------

    def _ray_geometry_batch(
        self, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centres, widths and log-weights for a batch of unit directions."""
        if self.mode == "polar":
            return self._geometry_polar(directions)
        if self.mode == "adapted":
            return self._geometry_adapted(directions)
        return self._geometry_global(directions)

    def _geometry_polar(self, directions: np.ndarray):
        """Contribution centred on each point's own radius.

        The key difference from ``adapted``: a point contributes at its own
        radius rather than at its projection onto the ray, so the curvature
        of the shell does not pull the estimate inward. Angular distance
        only decides whether the point is relevant to this direction.
        """
        cosine = (directions @ self.y_data.T) / np.maximum(self._norm[None, :], EPS)
        np.clip(cosine, -1.0, 1.0, out=cosine)

        centres = np.repeat(self._norm[None, :], directions.shape[0], axis=0)
        log_weights = -self.kappa * (1.0 - cosine)

        # tau is estimated from the data as the weighted scatter of radii,
        # the physical shell thickness rather than a free parameter.
        weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)

        mean = (weights * centres).sum(axis=1)
        variance = (weights * (centres - mean[:, None]) ** 2).sum(axis=1)
        local_eff_n = 1.0 / np.maximum((weights**2).sum(axis=1), EPS)
        correction = np.where(
            local_eff_n > 1.5, local_eff_n / np.maximum(local_eff_n - 1.0, EPS), 1.0
        )
        tau = np.maximum(
            np.sqrt(np.maximum(variance * correction, 0.0)) * self.scale, self.tau_floor
        )
        return centres, tau, log_weights

    def _geometry_adapted(self, directions: np.ndarray):
        """Separate radial and transverse widths."""
        centres = directions @ self.y_data.T
        tau = np.full(centres.shape[0], float(self.sigma_r) * self.scale)

        perpendicular = self._norm2[None, :] - centres * centres
        np.maximum(perpendicular, 0.0, out=perpendicular)

        log_weights = perpendicular
        if self.sigma_t_mode == "angular":
            # A fixed width in MPa gives different angular resolution in
            # different sectors: at 313 MPa the cone is half as wide as at
            # 180, so the effective sample size collapses precisely in the
            # tensile directions. Normalising by the point's own radius
            # removes that.
            log_weights = log_weights / np.maximum(self._norm2[None, :], EPS)
        log_weights = log_weights * (-0.5 / (float(self.sigma_t) ** 2))
        return centres, tau, log_weights

    def _geometry_global(self, directions: np.ndarray):
        """Textbook slice through the full kernel."""
        scaled = directions @ self.inverse_bandwidth
        a = np.maximum(np.einsum("bi,bi->b", scaled, directions), EPS)
        b = scaled @ self.y_data.T

        centres = b / a[:, None]
        tau = 1.0 / np.sqrt(a)

        log_weights = self._quadratic_c[None, :] - b * centres
        np.maximum(log_weights, 0.0, out=log_weights)
        log_weights *= -0.5
        return centres, tau, log_weights

    def _select_top_k(
        self, centres: np.ndarray, log_weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep the highest-weight dataset points for each query point."""
        n = log_weights.shape[1]
        if self.top_k is None or self.top_k >= n:
            return centres, log_weights

        index = np.argpartition(log_weights, n - self.top_k, axis=1)[:, n - self.top_k :]
        return (
            np.take_along_axis(centres, index, axis=1),
            np.take_along_axis(log_weights, index, axis=1),
        )

    @staticmethod
    def _mixture_density(
        r: np.ndarray, centres: np.ndarray, tau: float, log_weights: np.ndarray
    ) -> np.ndarray:
        """Untruncated mixture density, evaluated in log space."""
        weights = np.exp(log_weights - _logsumexp(log_weights))
        z = (np.asarray(r, dtype=np.float64)[:, None] - centres[None, :]) / tau
        return (weights[None, :] * INV_SQRT_2PI * np.exp(-0.5 * z * z) / tau).sum(axis=1)

    @staticmethod
    def _radius_grid(
        centres: np.ndarray, tau: float, log_weights: np.ndarray, n_grid: int
    ) -> np.ndarray:
        """Grid over the positive branch, around the significant centres."""
        weights = np.exp(log_weights - _logsumexp(log_weights))
        positive = centres > 0.0

        if positive.any() and weights[positive].sum() > EPS:
            w = weights[positive] / weights[positive].sum()
            mean = float(np.dot(w, centres[positive]))
            std = float(np.sqrt(max(np.dot(w, (centres[positive] - mean) ** 2), 0.0)))
        else:
            mean, std = float(np.abs(centres).max()), 0.0

        half = 5.0 * tau + 4.0 * std
        return np.linspace(max(0.0, mean - half), mean + half, n_grid)


def _logsumexp(values: np.ndarray) -> float:
    """Stable log-sum-exp of a one-dimensional array."""
    peak = float(np.max(values))
    return peak + float(np.log(np.sum(np.exp(values - peak))))


def _shell_thickness(
    y: np.ndarray, n_directions: int = 128, k_neighbours: int = 200, seed: int = 0
) -> float:
    """Estimate the local thickness of the yield shell, MPa.

    A global ``std(||y||)`` is not the thickness: it mixes microstructural
    scatter, which is what matters here, with the anisotropy of the surface
    itself. For a BCC polycrystal the radius runs from about 180 MPa in
    shear to 313 MPa in tension, and that relief would swamp the thickness.

    The measurement is therefore local: for each probe direction the ``k``
    angularly nearest points are taken and the spread of their radii
    computed. This is an upper bound -- a cone of ``k`` neighbours still has
    finite opening, so some anisotropy leaks in.
    """
    rng = np.random.default_rng(seed)
    radii = np.linalg.norm(y, axis=1)
    unit = y / np.maximum(radii[:, None], EPS)

    n_directions = min(n_directions, len(y))
    probes = unit[rng.choice(len(y), size=n_directions, replace=False)]

    k = min(k_neighbours, len(y))
    cosine = probes @ unit.T
    nearest = np.argpartition(-cosine, k - 1, axis=1)[:, :k]
    return float(np.median(np.std(radii[nearest], axis=1)))
