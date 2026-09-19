"""Split stress vectors into hydrostatic and deviatoric parts.

::

    sigma   = sigma_m * I + s
    sigma_m = (sigma_xx + sigma_yy + sigma_zz) / 3
    s       = sigma - sigma_m * I

The von Mises criterion depends only on the deviator ``s``: hydrostatic
pressure does not drive yielding. In 6D the yield surface is therefore not a
cloud around the origin but a cylinder, infinite along the hydrostatic axis
``(1, 1, 1, 0, 0, 0) / sqrt(3)``. Estimating density in the raw space lets
hydrostatic scatter distort both the ray direction and its length, which
makes genuine field points look uncovered by the dataset.

The transform is applied to the dataset and to every query, always through
:class:`mvyield.model.transform.SpaceTransform`, so the two spaces cannot
drift apart. This module holds only the arithmetic.

Slots 0-2 are the normal components under every convention this package
accepts (enforced by :class:`~mvyield.mechanics.voigt.VoigtConvention`), so
these functions are convention-agnostic: shear slots are never touched.

The functions are ported unchanged from ``solver/deviator.py``.
"""

from __future__ import annotations

import numpy as np

EPS_NORM = 1e-9


def hydrostatic_mean(sigma6: np.ndarray) -> np.ndarray:
    """Mean normal stress ``sigma_m = (XX + YY + ZZ) / 3``.

    Parameters
    ----------
    sigma6 : ndarray of shape (..., 6)
        Stress vectors, MPa.

    Returns
    -------
    ndarray of shape (...,)
        Scalar for a single vector, an ``(N,)`` array for a field.
    """
    sigma6 = np.asarray(sigma6, dtype=np.float64)
    return sigma6[..., 0:3].mean(axis=-1)


def to_deviatoric(sigma6: np.ndarray) -> np.ndarray:
    """Remove the hydrostatic part, leaving a trace-free normal block.

    Parameters
    ----------
    sigma6 : ndarray of shape (..., 6)
        Stress vectors, MPa.

    Returns
    -------
    ndarray
        Same shape as the input; ``result[..., :3].sum(axis=-1)`` is ~0.
    """
    sigma6 = np.asarray(sigma6, dtype=np.float64)
    s = sigma6.copy()
    s_m = s[..., 0:3].mean(axis=-1, keepdims=True)
    s[..., 0:3] -= s_m
    return s


def deviatoric_direction(direction: np.ndarray) -> np.ndarray:
    """Project a direction onto the deviatoric subspace and renormalise.

    Used where the input is a loading direction rather than a stress state,
    for instance when evaluating a ray-marginal density along a given ray.
    """
    direction = np.asarray(direction, dtype=np.float64)
    projected = to_deviatoric(direction)
    magnitude = float(np.linalg.norm(projected))
    return projected / max(magnitude, EPS_NORM)


def decompose(sigma6: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose stress vectors into deviatoric direction, radius and pressure.

    This is the single entry point for anything that needs the polar view of
    a stress state, so that direction and radius are always computed from the
    same deviatoric projection.

    Parameters
    ----------
    sigma6 : ndarray of shape (..., 6)
        Raw stress vectors, MPa.

    Returns
    -------
    direction : ndarray of shape (..., 6)
        Unit vectors inside the deviatoric subspace; zero where the deviator
        vanishes (purely hydrostatic states).
    radius : ndarray of shape (...,)
        Deviator norm ``||s||``, MPa.
    sigma_m : ndarray of shape (...,)
        Removed hydrostatic mean, MPa. Kept for diagnostics: a large spread
        here is what motivated the deviatoric transform in the first place.
    """
    sigma6 = np.asarray(sigma6, dtype=np.float64)
    single = sigma6.ndim == 1
    if single:
        sigma6 = sigma6[np.newaxis, :]

    sigma_m = hydrostatic_mean(sigma6)
    s = to_deviatoric(sigma6)
    radius = np.linalg.norm(s, axis=-1)

    valid = radius > EPS_NORM
    direction = np.zeros_like(s)
    direction[valid] = s[valid] / radius[valid, np.newaxis]

    if single:
        return direction[0], radius[0], sigma_m[0]
    return direction, radius, sigma_m
