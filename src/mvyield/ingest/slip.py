"""Slip systems and grain orientations.

A slip system is a plane normal and a slip direction lying in that plane.
Yielding by the Schmid criterion happens when the shear stress resolved onto
any one of them reaches the critical resolved shear stress, so this table
plus an orientation is the whole of what makes one grain yield before
another under the same load.

Normalisation happens once, here, at import. The legacy code renormalised
all 48 systems inside every call of ``rss`` -- for 40 geometries and 120
steps that is 4800 repetitions of the same arithmetic, and worse, it left
open the possibility of a caller passing unnormalised vectors and getting a
silently rescaled tau.

Orientations
------------
``local_cs`` holds Bunge Euler angles ``(phi1, Phi, phi2)`` in degrees, one
triple per grain. :func:`euler_to_rotation` builds the matrices exactly as
``math_utils.euler_to_matrix_bunge_local_to_global`` did, because the
generated datasets computed their reference yield points with that same
matrix: a different but equally defensible Euler convention would put this
package's answers next to a reference that no longer describes them.
"""

from __future__ import annotations

import numpy as np

from mvyield.settings import ConfigError

# --- Slip system families ----------------------------------------------------
#
# Each entry is (plane normal, slip direction) in crystal axes, as integer
# Miller indices. Kept as integers so that a typo is visible; normalisation
# is applied below.

BCC_12: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
    # {110} <111>
    ((1, 1, 0), (-1, 1, 1)), ((1, 1, 0), (1, -1, 1)),
    ((1, 0, 1), (-1, 1, 1)), ((1, 0, 1), (1, 1, -1)),
    ((0, 1, 1), (1, -1, 1)), ((0, 1, 1), (1, 1, -1)),
    ((1, -1, 0), (1, 1, 1)), ((1, -1, 0), (1, 1, -1)),
    ((1, 0, -1), (1, 1, 1)), ((1, 0, -1), (1, -1, 1)),
    ((0, 1, -1), (1, 1, 1)), ((0, 1, -1), (-1, 1, 1)),
)
"""BCC with the {110}<111> family only."""

BCC_48: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
    # {110} <111> -- 24 systems
    ((0, 1, 1), (-1, -1, 1)), ((0, 1, 1), (-1, 1, -1)),
    ((0, 1, 1), (1, -1, 1)), ((0, 1, 1), (1, 1, -1)),
    ((0, 1, -1), (-1, -1, -1)), ((0, 1, -1), (-1, 1, 1)),
    ((0, 1, -1), (1, -1, -1)), ((0, 1, -1), (1, 1, 1)),
    ((1, 0, 1), (-1, -1, 1)), ((1, 0, 1), (-1, 1, 1)),
    ((1, 0, 1), (1, -1, -1)), ((1, 0, 1), (1, 1, -1)),
    ((1, 0, -1), (-1, -1, -1)), ((1, 0, -1), (-1, 1, -1)),
    ((1, 0, -1), (1, -1, 1)), ((1, 0, -1), (1, 1, 1)),
    ((1, 1, 0), (-1, 1, -1)), ((1, 1, 0), (-1, 1, 1)),
    ((1, 1, 0), (1, -1, -1)), ((1, 1, 0), (1, -1, 1)),
    ((1, -1, 0), (-1, -1, -1)), ((1, -1, 0), (-1, -1, 1)),
    ((1, -1, 0), (1, 1, -1)), ((1, -1, 0), (1, 1, 1)),
    # {112} <111> -- 24 systems
    ((1, 1, 2), (-1, -1, 1)), ((1, 1, 2), (1, 1, -1)),
    ((1, 1, -2), (-1, -1, -1)), ((1, 1, -2), (1, 1, 1)),
    ((1, -1, 2), (-1, 1, 1)), ((1, -1, 2), (1, -1, -1)),
    ((-1, 1, 2), (-1, 1, -1)), ((-1, 1, 2), (1, -1, 1)),
    ((1, 2, 1), (-1, 1, -1)), ((1, 2, 1), (1, -1, 1)),
    ((1, 2, -1), (-1, 1, 1)), ((1, 2, -1), (1, -1, -1)),
    ((1, -2, 1), (-1, -1, -1)), ((1, -2, 1), (1, 1, 1)),
    ((-1, 2, 1), (-1, -1, 1)), ((-1, 2, 1), (1, 1, -1)),
    ((2, 1, 1), (-1, 1, 1)), ((2, 1, 1), (1, -1, -1)),
    ((2, 1, -1), (-1, 1, -1)), ((2, 1, -1), (1, -1, 1)),
    ((2, -1, 1), (-1, -1, 1)), ((2, -1, 1), (1, 1, -1)),
    ((-2, 1, 1), (-1, -1, -1)), ((-2, 1, 1), (1, 1, 1)),
)
"""BCC with both the {110} and {112} families.

What the legacy analysis and the dataset generator both use, so it is the
default for this project's datasets.
"""

FCC_12: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
    # {111} <110>
    ((1, 1, 1), (0, -1, 1)), ((1, 1, 1), (1, 0, -1)), ((1, 1, 1), (-1, 1, 0)),
    ((-1, 1, 1), (0, -1, 1)), ((-1, 1, 1), (1, 0, 1)), ((-1, 1, 1), (1, 1, 0)),
    ((1, -1, 1), (0, 1, 1)), ((1, -1, 1), (1, 0, -1)), ((1, -1, 1), (1, 1, 0)),
    ((1, 1, -1), (0, -1, -1)), ((1, 1, -1), (1, 0, 1)), ((1, 1, -1), (-1, 1, 0)),
)
"""FCC: aluminium, copper, nickel, austenitic steels."""

FAMILIES = {"BCC": BCC_12, "BCC_48": BCC_48, "FCC": FCC_12}
"""Structure name to its slip systems."""


def slip_systems(structure: str) -> tuple[np.ndarray, np.ndarray]:
    """Unit plane normals and slip directions for one crystal structure.

    Parameters
    ----------
    structure : str
        Key of :data:`FAMILIES`.

    Returns
    -------
    planes : ndarray of shape (S, 3)
        Unit normals.
    directions : ndarray of shape (S, 3)
        Unit slip directions.
    """
    try:
        systems = FAMILIES[structure.upper()]
    except KeyError:
        raise ConfigError(
            f"no slip systems for structure {structure!r}; "
            f"available: {sorted(FAMILIES)}"
        ) from None
    return _normalised(systems)


def euler_to_rotation(euler_deg: np.ndarray) -> np.ndarray:
    """Bunge Euler angles in degrees to rotation matrices.

    Parameters
    ----------
    euler_deg : ndarray of shape (N, 3) or (3,)
        ``(phi1, Phi, phi2)`` per grain, degrees.

    Returns
    -------
    ndarray of shape (N, 3, 3)
        One matrix per grain.
    """
    angles = np.radians(np.atleast_2d(np.asarray(euler_deg, dtype=np.float64)))
    if angles.shape[1] != 3:
        raise ValueError(f"expected Euler triples, got shape {angles.shape}")

    c1, s1 = np.cos(angles[:, 0]), np.sin(angles[:, 0])
    c, s = np.cos(angles[:, 1]), np.sin(angles[:, 1])
    c2, s2 = np.cos(angles[:, 2]), np.sin(angles[:, 2])

    rotation = np.empty((angles.shape[0], 3, 3))
    rotation[:, 0, 0] = c1 * c2 - s1 * s2 * c
    rotation[:, 0, 1] = -c1 * s2 - s1 * c2 * c
    rotation[:, 0, 2] = s1 * s
    rotation[:, 1, 0] = s1 * c2 + c1 * s2 * c
    rotation[:, 1, 1] = -s1 * s2 + c1 * c2 * c
    rotation[:, 1, 2] = -c1 * s
    rotation[:, 2, 0] = s2 * s
    rotation[:, 2, 1] = c2 * s
    rotation[:, 2, 2] = c
    return rotation


def resolved_shear(
    tensors: np.ndarray,
    planes: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Shear stress resolved onto every slip system.

    Parameters
    ----------
    tensors : ndarray of shape (N, 3, 3)
        Stress tensors already expressed in the grain frame.
    planes, directions : ndarray
        Unit normals and slip directions from :func:`slip_systems`.

    Returns
    -------
    ndarray of shape (N, S)
        ``|n . sigma . m|`` for each point and system. The absolute value is
        taken because slip runs either way along a direction.
    """
    tractions = np.einsum("nij,mj->nmi", tensors, planes)
    return np.abs(np.einsum("nmi,mi->nm", tractions, directions))


def schmid_tensors(
    planes: np.ndarray,
    directions: np.ndarray,
    rotations: np.ndarray,
) -> np.ndarray:
    """Slip systems as Mandel-form Schmid tensors in the global frame.

    The resolved shear on system ``m`` of grain ``g`` is

    .. math::

        \\tau = n^T (R \\sigma R^T) m = (R^T n)^T \\sigma (R^T m)
              = \\sigma : \\mathrm{sym}(n' \\otimes m')

    so rotating the *systems* once per grain gives the same numbers as
    rotating the *stress* once per material point -- and there are two
    orders of magnitude fewer grains than points. In Mandel components the
    double contraction is an ordinary dot product, which turns the whole
    per-step computation into one matrix product.

    Parameters
    ----------
    planes, directions : ndarray of shape (S, 3)
        Unit vectors from :func:`slip_systems`.
    rotations : ndarray of shape (G, 3, 3)
        Grain orientations from :func:`euler_to_rotation`.

    Returns
    -------
    ndarray of shape (G, S, 6)
        Schmid tensors in ``MANDEL_XY_LAST`` order, so that
        ``sigma_mandel @ tensor`` is the resolved shear.

    Notes
    -----
    The rotation is applied as ``R sigma R^T``, pairing the Bunge matrix of
    :func:`euler_to_rotation` with this contraction. That pairing is not
    arbitrary: the generated datasets computed their reference yield points
    with exactly it, so the opposite one -- defensible on its own -- would
    put this package a few per cent away from the reference with nothing to
    explain the gap.
    """
    n = np.einsum("gji,mj->gmi", rotations, planes)
    m = np.einsum("gji,mj->gmi", rotations, directions)

    inverse_sqrt2 = 1.0 / np.sqrt(2.0)
    tensors = np.empty(n.shape[:-1] + (6,))
    tensors[..., 0] = n[..., 0] * m[..., 0]
    tensors[..., 1] = n[..., 1] * m[..., 1]
    tensors[..., 2] = n[..., 2] * m[..., 2]
    tensors[..., 3] = (n[..., 1] * m[..., 2] + n[..., 2] * m[..., 1]) * inverse_sqrt2
    tensors[..., 4] = (n[..., 0] * m[..., 2] + n[..., 2] * m[..., 0]) * inverse_sqrt2
    tensors[..., 5] = (n[..., 0] * m[..., 1] + n[..., 1] * m[..., 0]) * inverse_sqrt2
    return tensors


def independent_systems(planes: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Indices of the slip systems that are distinct under the criterion.

    A system ``(n, b)`` and its reverse ``(n, -b)`` resolve the same
    ``|tau|`` -- slip runs either way along a direction -- and their Schmid
    tensors differ only in sign. The BCC_48 table lists both members of
    every pair, so it holds 48 entries but only 24 distinct systems.

    Counting table entries therefore doubles any "how many systems are
    active" answer, which is how the legacy analysis came to report that
    every yield point sits on an edge: the minimum it could ever return was
    two. Use these indices when counting.

    Returns
    -------
    ndarray
        Indices into ``planes`` / ``directions``, one per distinct system.
    """
    tensors = schmid_tensors(planes, directions, np.eye(3)[None, :, :])[0]

    seen: dict[tuple, int] = {}
    keep: list[int] = []
    for index, tensor in enumerate(tensors):
        leading = np.flatnonzero(np.abs(tensor) > 1e-12)
        canonical = tensor if not len(leading) else tensor * np.sign(tensor[leading[0]])
        key = tuple(np.round(canonical, 9) + 0.0)  # +0.0 normalises signed zero
        if key not in seen:
            seen[key] = index
            keep.append(index)
    return np.array(keep, dtype=np.int64)


def selfcheck(tolerance: float = 1e-12) -> dict[str, float]:
    """Verify that every slip direction lies in its slip plane.

    ``n . m = 0`` is what makes a system physical: slip happens *within* a
    plane. A mistyped Miller index usually breaks exactly this, and nothing
    downstream would notice -- the resolved shear would simply be wrong.
    """
    worst: dict[str, float] = {}
    for name in FAMILIES:
        planes, directions = slip_systems(name)
        residual = float(np.abs(np.einsum("mi,mi->m", planes, directions)).max())
        worst[name] = residual
        if residual > tolerance:
            raise AssertionError(
                f"{name}: a slip direction is not in its plane (max |n.m| = {residual:.2e})"
            )
    return worst


def _normalised(systems) -> tuple[np.ndarray, np.ndarray]:
    """Turn integer Miller indices into unit vectors."""
    planes = np.array([p for p, _ in systems], dtype=np.float64)
    directions = np.array([d for _, d in systems], dtype=np.float64)
    planes /= np.linalg.norm(planes, axis=1, keepdims=True)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return planes, directions
