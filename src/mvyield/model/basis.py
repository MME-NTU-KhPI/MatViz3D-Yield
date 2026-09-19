"""Orthonormal basis of the subspace the model actually occupies.

Why a basis is needed at all
----------------------------
Ray marginalisation uses the bandwidth matrix directly, through
``sigma_d = sqrt(d^T H d)``, and does not care that ``H`` is singular. After
the deviatoric transform every point satisfies ``XX + YY + ZZ = 0``, so the
variance along the hydrostatic axis is identically zero and ``rank(H) = 5``.
That is why the legacy build had to fall back on ``pinv``.

The slice method works with the *inverse*, and there ``pinv`` is not a
harmless fallback: it returns zero along the degenerate direction, silently
ignoring the hydrostatic component instead of penalising it, which corrupts
the exponents. The fix is to drop the degenerate direction entirely and work
in coordinates of the subspace itself:

    y = B.T @ sigma,   B in R^(6 x k),   B.T @ B = I_k

There the covariance has full rank, the Cholesky factorisation is honest,
and Scott's rule gets the correct dimension.

Because the columns are orthonormal in the Mandel metric -- which coincides
with the ordinary Euclidean product, this being the point of Mandel notation
-- the map is an isometry: norms and angles are preserved, so radii in
subspace coordinates are the same numbers in MPa as in the ambient space.

Every deviatoric column is trace-free, hence ``B.T @ (sigma_m * I) = 0``:
projecting already removes the hydrostatic part, and applying
:func:`~mvyield.model.deviator.to_deviatoric` first is redundant though
harmless.

Choice of basis
---------------
The first two deviatoric columns are the octahedral coordinates, so ``y[0]``
and ``y[1]`` can be plotted directly in the pi-plane::

    b1 = ( 1, -1,  0, 0, 0, 0) / sqrt(2)     pi-plane, sigma_1 - sigma_2
    b2 = ( 1,  1, -2, 0, 0, 0) / sqrt(6)     pi-plane, 2*sigma_3 - sigma_1 - sigma_2
    b3, b4, b5                               the three shear slots

Pressure-sensitive materials
----------------------------
When the space transform keeps the hydrostatic axis (``deviatoric=False``),
the model occupies the full 6D space and the basis is the identity. Both
cases go through :func:`subspace_basis`, so the slice estimator needs no
special case for pressure-sensitive materials -- it just gets ``k = 6``.
"""

from __future__ import annotations

import numpy as np

from mvyield.mechanics.voigt import VOIGT_DIM, VoigtConvention

DIM_DEVIATORIC = 5
"""Dimension of the trace-free subspace."""

_S2 = np.sqrt(2.0)
_S6 = np.sqrt(6.0)

DEVIATORIC_BASIS: np.ndarray = np.array(
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
"""Columns spanning the trace-free subspace, for slot order (XX, YY, ZZ, *)."""

HYDROSTATIC_UNIT: np.ndarray = np.array(
    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
) / np.sqrt(3.0)
"""Unit vector along the hydrostatic axis: the orthogonal complement."""

DEVIATORIC_LABELS: tuple[str, ...] = (
    r"$y_1 = (\sigma_{xx}-\sigma_{yy})/\sqrt{2}$",
    r"$y_2 = (\sigma_{xx}+\sigma_{yy}-2\sigma_{zz})/\sqrt{6}$",
    r"$y_3 = \sqrt{2}\,\sigma_{yz}$",
    r"$y_4 = \sqrt{2}\,\sigma_{xz}$",
    r"$y_5 = \sqrt{2}\,\sigma_{xy}$",
)


def subspace_basis(deviatoric: bool, convention: VoigtConvention) -> np.ndarray:
    """Return the basis of the subspace a model occupies.

    Parameters
    ----------
    deviatoric : bool
        Whether the space transform removed the hydrostatic part.
    convention : VoigtConvention
        Slot order of the model space. Normal components always occupy slots
        0-2, so the deviatoric block is unaffected by the shear ordering and
        the same basis serves every accepted convention.

    Returns
    -------
    ndarray of shape (6, k)
        ``k = 5`` for a deviatoric space, ``k = 6`` otherwise.
    """
    if not deviatoric:
        return np.eye(VOIGT_DIM, dtype=np.float64)
    return DEVIATORIC_BASIS.copy()


def to_subspace(sigma6: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project stress vectors onto subspace coordinates.

    Parameters
    ----------
    sigma6 : ndarray of shape (..., 6)
        Vectors in the ambient model space, MPa.
    basis : ndarray of shape (6, k)
        Orthonormal columns from :func:`subspace_basis`.

    Returns
    -------
    ndarray of shape (..., k)
        Subspace coordinates. Norms are preserved for any component already
        lying in the subspace.
    """
    return np.asarray(sigma6, dtype=np.float64) @ basis


def from_subspace(y: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Lift subspace coordinates back into the ambient 6D space."""
    return np.asarray(y, dtype=np.float64) @ basis.T


def subspace_direction(direction: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project a direction into the subspace and renormalise.

    Accepts either an ambient 6D vector or coordinates already in the
    subspace, so callers holding one or the other need no branch.

    Raises
    ------
    ValueError
        If the direction lies entirely along the discarded axis, which for a
        deviatoric space means a purely hydrostatic state -- a direction the
        model cannot represent.
    """
    direction = np.asarray(direction, dtype=np.float64)
    k = basis.shape[1]
    y = direction if direction.shape[-1] == k else to_subspace(direction, basis)

    magnitude = float(np.linalg.norm(y))
    if magnitude < 1e-12:
        raise ValueError(
            "direction has no component inside the model subspace; for a "
            "deviatoric model this means a purely hydrostatic state, which "
            "carries no yield information"
        )
    return y / magnitude
