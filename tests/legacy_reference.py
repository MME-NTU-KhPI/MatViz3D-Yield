"""Verbatim extracts of the pre-refactor estimators, for parity testing.

Copied from ``solver/probability.py``, ``solver/probability_slice.py`` and
``material/kde_yield_surface.py`` with only two changes: the Ukrainian
comments are dropped and the ``config`` and ``tqdm`` imports are removed so
the functions stand alone.

The point is to have an independent oracle. The refactored estimators are
checked against *these* functions on synthetic clouds, so "the maths came
through the move unchanged" is a test result rather than an assurance. This
file is never imported by the package and must not be edited to make a test
pass -- if the two disagree, the port is wrong.

The recorded baseline from a real dataset run is still worth having, since
it also pins down the ingest stage. This covers the part that can be checked
without one.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm as _sp_norm

_EPS = 1e-12
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)
_S2 = np.sqrt(2.0)
_S6 = np.sqrt(6.0)

DEV_BASIS = np.array(
    [
        [1.0 / _S2, 1.0 / _S6, 0.0, 0.0, 0.0],
        [-1.0 / _S2, 1.0 / _S6, 0.0, 0.0, 0.0],
        [0.0, -2.0 / _S6, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def to_deviatoric(sigma_voigt):
    sigma_voigt = np.asarray(sigma_voigt, dtype=np.float64)
    s = sigma_voigt.copy()
    s_m = s[..., 0:3].mean(axis=-1, keepdims=True)
    s[..., 0:3] -= s_m
    return s


def to_5d(sigma_voigt):
    return np.asarray(sigma_voigt, dtype=np.float64) @ DEV_BASIS


def compute_kde_bundle(sigma_mpa):
    """From solver/build_kde.py."""
    sigma_mpa_raw = np.asarray(sigma_mpa, dtype=np.float64)
    sigma_mpa = to_deviatoric(sigma_mpa_raw)
    N, D = sigma_mpa.shape

    h = N ** (-1.0 / (D + 4))
    Cov = np.cov(sigma_mpa, rowvar=False)
    H = (h**2) * Cov

    rank = np.linalg.matrix_rank(H)
    H_inv = np.linalg.pinv(H) if rank < D else np.linalg.inv(H)

    return {
        "sigma_data": sigma_mpa,
        "n_points": int(N),
        "bandwidth_h": float(h),
        "cov_matrix": Cov,
        "bandwidth_H": H,
        "bandwidth_H_inv": H_inv,
    }


def upgrade_bundle(bundle):
    """From solver/kde_bundle_v2.py."""
    y = to_5d(bundle["sigma_data"])
    N = y.shape[0]
    DIM_DEV = 5

    h5 = float(N ** (-1.0 / (DIM_DEV + 4)))
    cov5 = np.cov(y, rowvar=False)
    H5 = (h5**2) * cov5

    H5_inv = np.linalg.inv(H5)
    H5_inv = 0.5 * (H5_inv + H5_inv.T)

    bundle.update(
        {
            "sigma_data_5d": y,
            "bandwidth_h5": h5,
            "cov_matrix_5d": cov5,
            "bandwidth_H5": H5,
            "bandwidth_H5_inv": H5_inv,
        }
    )
    return bundle


def evaluate_field(sigma_voigt_field, bundle, p_low=0.005, batch_size=1500):
    """From solver/probability.py."""
    sigma_field_raw = np.asarray(sigma_voigt_field, dtype=np.float64)
    M = len(sigma_field_raw)
    sigma_field = to_deviatoric(sigma_field_raw)

    sigma_data = bundle["sigma_data"]
    H = bundle["bandwidth_H"]

    r_star = np.linalg.norm(sigma_field, axis=1)
    valid = r_star > 1e-9

    prob = np.zeros(M, dtype=np.float64)
    sigma_d_out = np.zeros(M, dtype=np.float64)
    covered = np.zeros(M, dtype=bool)

    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        batch = sigma_field[start:end]
        batch_r = r_star[start:end]
        batch_val = valid[start:end]
        if not np.any(batch_val):
            continue

        d_batch = np.zeros_like(batch)
        safe_r = np.where(batch_r > 1e-9, batch_r, 1.0)
        d_batch[batch_val] = batch[batch_val] / safe_r[batch_val, np.newaxis]

        proj = d_batch @ sigma_data.T
        dH = d_batch @ H
        sigma_d_sq = np.einsum("bi,bi->b", dH, d_batch)
        sigma_d = np.sqrt(np.maximum(sigma_d_sq, 1e-15))

        z = (batch_r[:, np.newaxis] - proj) / sigma_d[:, np.newaxis]
        batch_prob = np.mean(_sp_norm.cdf(z), axis=1)
        batch_dens = np.mean(_sp_norm.pdf(z), axis=1) / sigma_d

        proj_min = proj.min(axis=1)
        proj_max = proj.max(axis=1)
        in_range = (batch_r >= proj_min) & (batch_r <= proj_max)
        density_ok = batch_dens >= p_low

        batch_prob[~batch_val] = 0.0
        prob[start:end] = batch_prob
        sigma_d_out[start:end] = sigma_d
        covered[start:end] = batch_val & in_range & density_ok

    return prob, sigma_d_out, covered


def evaluate_field_local(
    sigma_voigt_field, bundle, power=16.0, min_eff_n=15.0, batch_size=1500
):
    """From solver/probability.py."""
    sigma_field_raw = np.asarray(sigma_voigt_field, dtype=np.float64)
    M = len(sigma_field_raw)
    sigma_field = to_deviatoric(sigma_field_raw)
    sigma_data = bundle["sigma_data"]

    data_norms = np.linalg.norm(sigma_data, axis=1)
    safe_norms = np.where(data_norms > 1e-9, data_norms, 1.0)

    r_star = np.linalg.norm(sigma_field, axis=1)
    valid = r_star > 1e-9

    prob = np.full(M, np.nan, dtype=np.float64)
    sigma_d_out = np.full(M, np.nan, dtype=np.float64)
    covered = np.zeros(M, dtype=bool)

    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        batch = sigma_field[start:end]
        batch_r = r_star[start:end]
        batch_val = valid[start:end]
        if not np.any(batch_val):
            continue

        d_batch = np.zeros_like(batch)
        safe_r = np.where(batch_r > 1e-9, batch_r, 1.0)
        d_batch[batch_val] = batch[batch_val] / safe_r[batch_val, np.newaxis]

        proj = d_batch @ sigma_data.T
        cos_angle = proj / safe_norms[np.newaxis, :]
        w = np.clip(cos_angle, 0.0, None) ** power

        w_sum = w.sum(axis=1)
        eff_n = w_sum**2 / np.maximum((w**2).sum(axis=1), 1e-30)

        safe_wsum = np.where(w_sum > 1e-9, w_sum, 1.0)
        r_mean = (w @ data_norms) / safe_wsum
        r_var = (w @ (data_norms**2)) / safe_wsum - r_mean**2
        r_std = np.sqrt(np.maximum(r_var, 1e-6))

        z = (batch_r - r_mean) / r_std
        batch_prob = _sp_norm.cdf(z)
        batch_covered = batch_val & (w_sum > 1e-9) & (eff_n >= min_eff_n)

        prob[start:end] = np.where(batch_covered, batch_prob, np.nan)
        sigma_d_out[start:end] = r_std
        covered[start:end] = batch_covered

    return prob, sigma_d_out, covered


def evaluate_field_slice(
    sigma_voigt_field,
    bundle,
    mode="polar",
    kappa=20.0,
    tau_floor=2.0,
    sigma_r=None,
    sigma_t=None,
    sigma_t_mode="angular",
    p_low=0.005,
    min_eff_n=15.0,
    censor=False,
    batch_size=400,
    top_k=4096,
):
    """From solver/probability_slice.py."""
    sigma_field_raw = np.asarray(sigma_voigt_field, dtype=np.float64)
    M = sigma_field_raw.shape[0]

    y_field = to_5d(to_deviatoric(sigma_field_raw))
    r_star = np.linalg.norm(y_field, axis=1)
    valid = r_star > 1e-9

    y_data = bundle["sigma_data_5d"]
    N = y_data.shape[0]
    y_norm2 = np.einsum("ij,ij->i", y_data, y_data)
    y_norm = np.sqrt(np.maximum(y_norm2, _EPS))

    A_mat = bundle.get("bandwidth_H5_inv")
    c_vec = None
    if mode == "global":
        A_mat = 0.5 * (A_mat + A_mat.T)
        c_vec = np.einsum("ij,jk,ik->i", y_data, A_mat, y_data)

    prob = np.full(M, np.nan, dtype=np.float64)
    spread = np.full(M, np.nan, dtype=np.float64)
    eff_out = np.zeros(M, dtype=np.float64)
    covered = np.zeros(M, dtype=bool)

    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        b_valid = valid[start:end]
        if not np.any(b_valid):
            continue

        b_r = r_star[start:end]
        safe_r = np.where(b_r > 1e-9, b_r, 1.0)
        U = y_field[start:end] / safe_r[:, None]

        if mode == "adapted":
            mu = U @ y_data.T
            tau = np.full(mu.shape[0], sigma_r, dtype=np.float64)
            perp2 = y_norm2[None, :] - mu * mu
            np.maximum(perp2, 0.0, out=perp2)
            logw = perp2
            if sigma_t_mode == "angular":
                logw = logw / np.maximum(y_norm2[None, :], _EPS)
            logw = logw * (-0.5 / (sigma_t**2))
        elif mode == "polar":
            cos = (U @ y_data.T) / np.maximum(y_norm[None, :], _EPS)
            np.clip(cos, -1.0, 1.0, out=cos)
            mu = np.repeat(y_norm[None, :], U.shape[0], axis=0)
            logw = -kappa * (1.0 - cos)

            w = np.exp(logw - logw.max(axis=1, keepdims=True))
            w /= np.maximum(w.sum(axis=1, keepdims=True), _EPS)
            m_bar = (w * mu).sum(axis=1)
            var = (w * (mu - m_bar[:, None]) ** 2).sum(axis=1)
            n_eff_loc = 1.0 / np.maximum((w**2).sum(axis=1), _EPS)
            corr = np.where(
                n_eff_loc > 1.5, n_eff_loc / np.maximum(n_eff_loc - 1.0, _EPS), 1.0
            )
            tau = np.maximum(np.sqrt(np.maximum(var * corr, 0.0)), tau_floor)
        else:
            AU = U @ A_mat
            a = np.einsum("bi,bi->b", AU, U)
            a = np.maximum(a, _EPS)
            b_mat = AU @ y_data.T
            mu = b_mat / a[:, None]
            tau = 1.0 / np.sqrt(a)
            logw = c_vec[None, :] - b_mat * mu
            np.maximum(logw, 0.0, out=logw)
            logw = logw * -0.5

        if top_k is not None and top_k < N:
            idx = np.argpartition(logw, N - top_k, axis=1)[:, N - top_k :]
            mu_k = np.take_along_axis(mu, idx, axis=1)
            lw_k = np.take_along_axis(logw, idx, axis=1)
        else:
            mu_k, lw_k = mu, logw

        lw_k = lw_k - lw_k.max(axis=1, keepdims=True)
        w = np.exp(lw_k)

        tau_c = tau[:, None]
        zc = mu_k / tau_c

        lower = ndtr(-zc)
        upper = ndtr((b_r[:, None] - mu_k) / tau_c)

        w_sum = w.sum(axis=1)
        eff_n = w_sum**2 / np.maximum((w * w).sum(axis=1), 1e-300)

        w_t = w * (1.0 - lower)
        z_mass = w_t.sum(axis=1)
        num = (w * (upper - lower)).sum(axis=1)

        zz = (b_r[:, None] - mu_k) / tau_c
        dens = (w * (_INV_SQRT_2PI * np.exp(-0.5 * zz * zz)) / tau_c).sum(axis=1)
        dens /= np.maximum(z_mass, _EPS)

        safe_z = np.maximum(z_mass, _EPS)
        m1 = (w_t * mu_k).sum(axis=1) / safe_z
        m2 = (w_t * mu_k * mu_k).sum(axis=1) / safe_z
        b_spread = np.sqrt(tau**2 + np.maximum(m2 - m1 * m1, 0.0))

        signif = (mu_k > 0.0) & (w > w.max(axis=1, keepdims=True) * 1e-3)
        mu_lo = np.where(signif, mu_k, np.inf).min(axis=1)
        mu_hi = np.where(signif, mu_k, -np.inf).max(axis=1)
        in_range = (b_r >= mu_lo) & (b_r <= mu_hi)

        b_prob = np.where(z_mass > _EPS, num / safe_z, np.nan)
        b_prob = np.clip(b_prob, 0.0, 1.0)
        b_cov = b_valid & in_range & (dens >= p_low)
        if censor:
            b_cov = b_cov & (eff_n >= min_eff_n)

        prob[start:end] = np.where(b_valid, b_prob, np.nan)
        spread[start:end] = np.where(b_valid, b_spread, np.nan)
        eff_out[start:end] = np.where(b_valid, eff_n, 0.0)
        covered[start:end] = b_cov

    return prob, spread, covered, eff_out
