"""Voigt/Mandel conventions and stress-vector algebra.

A 6-component stress vector is ambiguous unless two things are pinned down:

1. **Component order** -- which slot holds which shear component.
2. **Shear scaling** -- whether shear entries carry the Mandel factor
   ``sqrt(2)``. With the factor, the Euclidean dot product of two vectors
   equals the tensor double contraction ``sigma : eps``, so norms and
   distances in 6D are physically meaningful. Without it, they are not.

Both must match between the dataset and every query, or the model is
silently evaluated in the wrong coordinates.

Known inconsistency in the legacy code
--------------------------------------
The pre-refactor code mixed two orders and two scalings:

===============================  =========  =========  ==============
Producer                         XY slot    Mandel     Reference
===============================  =========  =========  ==============
``dataset analysis script``      3          no         dataset points
``solver/build_kde.VOIGT_COLS``  3          --         column labels
``solver/build_kde.AXIS_NAMES``  5          --         axis labels
``solver/voigt_stress`` doc      3          --         docstring
``solver/voigt_stress`` code     5          yes        Kirsch query
``solver/deviator`` doc          5          yes        docstring
===============================  =========  =========  ==============

``VOIGT_COLS`` and ``AXIS_NAMES`` are indexed by the same ``i`` inside a
single module, so at least one of them is wrong wherever it is used.

More importantly, dataset points put engineering ``SXY`` in slot 3 while
Kirsch queries put ``sqrt(2) * sxy`` in slot 5. For any stress state with
non-zero in-plane shear -- which is everywhere off the symmetry axes in the
Kirsch field -- the query and the dataset are not in the same coordinates.

This module makes the convention an explicit object rather than a comment.
:data:`LEGACY_DATASET` and :data:`LEGACY_QUERY` reproduce the old behaviour
bit-for-bit so regression baselines still validate; :data:`MANDEL_XY_LAST`
is the convention new work should use. Use :func:`convention_mismatch_report`
to quantify what the legacy pairing costs before changing anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VOIGT_DIM = 6
"""Number of independent components of a symmetric second-order tensor."""

EPS_NORM = 1e-9
"""Threshold below which a stress vector is treated as null."""

SQRT2 = float(np.sqrt(2.0))


@dataclass(frozen=True)
class VoigtConvention:
    """Component order and shear scaling of a 6D stress vector.

    Parameters
    ----------
    name : str
        Identifier stored in artefacts so a run can be traced back.
    order : tuple of str
        Component labels in slot order. Must be a permutation of
        ``("XX", "YY", "ZZ", "XY", "YZ", "XZ")``.
    mandel : bool
        Whether shear slots carry the ``sqrt(2)`` factor.
    """

    name: str
    order: tuple[str, str, str, str, str, str]
    mandel: bool

    CANONICAL: tuple[str, ...] = ("XX", "YY", "ZZ", "XY", "YZ", "XZ")

    def __post_init__(self) -> None:
        """Validate that the order is a permutation of the canonical labels."""
        if sorted(self.order) != sorted(self.CANONICAL):
            raise ValueError(
                f"convention {self.name!r}: order {self.order} is not a "
                f"permutation of {self.CANONICAL}"
            )
        if self.order[:3] != ("XX", "YY", "ZZ"):
            raise ValueError(
                f"convention {self.name!r}: normal components must occupy "
                f"slots 0-2, got {self.order[:3]}"
            )

    @property
    def shear_slots(self) -> tuple[int, int, int]:
        """Slot indices holding shear components, always ``(3, 4, 5)``."""
        return (3, 4, 5)

    def slot_of(self, label: str) -> int:
        """Return the slot index holding ``label`` (e.g. ``"XY"``)."""
        return self.order.index(label)

    def to_dict(self) -> dict:
        """Serialise for storage in a model bundle."""
        return {"name": self.name, "order": list(self.order), "mandel": self.mandel}

    @classmethod
    def from_dict(cls, data: dict) -> VoigtConvention:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            name=data["name"],
            order=tuple(data["order"]),  # type: ignore[arg-type]
            mandel=bool(data["mandel"]),
        )


MANDEL_XY_LAST = VoigtConvention(
    name="mandel_xy_last",
    order=("XX", "YY", "ZZ", "YZ", "XZ", "XY"),
    mandel=True,
)
"""Recommended convention: Mandel-scaled, in-plane shear in the last slot."""

MANDEL_XY_FIRST = VoigtConvention(
    name="mandel_xy_first",
    order=("XX", "YY", "ZZ", "XY", "YZ", "XZ"),
    mandel=True,
)
"""Mandel-scaled with the ANSYS/engineering shear order."""

LEGACY_DATASET = VoigtConvention(
    name="legacy_dataset",
    order=("XX", "YY", "ZZ", "XY", "YZ", "XZ"),
    mandel=False,
)
"""What the original dataset analysis script produced: XY in slot 3, no sqrt(2)."""

LEGACY_QUERY = VoigtConvention(
    name="legacy_query",
    order=("XX", "YY", "ZZ", "YZ", "XZ", "XY"),
    mandel=True,
)
"""What ``solver.voigt_stress.to_voigt_6d`` produced: XY in slot 5, with sqrt(2)."""

CONVENTIONS: dict[str, VoigtConvention] = {
    c.name: c for c in (MANDEL_XY_LAST, MANDEL_XY_FIRST, LEGACY_DATASET, LEGACY_QUERY)
}


def get_convention(name: str) -> VoigtConvention:
    """Look up a registered convention by name.

    Raises
    ------
    KeyError
        If the name is unknown, with the available names in the message.
    """
    try:
        return CONVENTIONS[name]
    except KeyError:
        raise KeyError(
            f"unknown Voigt convention {name!r}; available: {sorted(CONVENTIONS)}"
        ) from None


# --- Assembly from tensor components ----------------------------------------


def from_components(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    xy: np.ndarray,
    yz: np.ndarray,
    xz: np.ndarray,
    convention: VoigtConvention,
) -> np.ndarray:
    """Assemble 6D stress vectors from tensor components.

    Parameters
    ----------
    xx, yy, zz, xy, yz, xz : ndarray
        Tensor components in consistent units (MPa throughout this package).
        Shear inputs are *engineering* components, i.e. the plain tensor
        entries with no ``sqrt(2)``; the factor is applied here when the
        convention requires it.
    convention : VoigtConvention
        Target order and scaling.

    Returns
    -------
    ndarray of shape (N, 6)
        Stress vectors in the requested convention.
    """
    parts = {
        "XX": np.asarray(xx, dtype=np.float64).ravel(),
        "YY": np.asarray(yy, dtype=np.float64).ravel(),
        "ZZ": np.asarray(zz, dtype=np.float64).ravel(),
        "XY": np.asarray(xy, dtype=np.float64).ravel(),
        "YZ": np.asarray(yz, dtype=np.float64).ravel(),
        "XZ": np.asarray(xz, dtype=np.float64).ravel(),
    }
    sizes = {v.size for v in parts.values()}
    if len(sizes) != 1:
        raise ValueError(f"component arrays have mismatched sizes: {sizes}")

    n = sizes.pop()
    out = np.zeros((n, VOIGT_DIM), dtype=np.float64)
    scale = SQRT2 if convention.mandel else 1.0
    for slot, label in enumerate(convention.order):
        is_shear = label not in ("XX", "YY", "ZZ")
        out[:, slot] = parts[label] * (scale if is_shear else 1.0)
    return out


def plane_stress(
    sxx: np.ndarray,
    syy: np.ndarray,
    sxy: np.ndarray,
    convention: VoigtConvention,
) -> np.ndarray:
    """Assemble plane-stress states, with ``zz``, ``yz`` and ``xz`` set to zero."""
    sxx = np.asarray(sxx, dtype=np.float64).ravel()
    zeros = np.zeros_like(sxx)
    return from_components(sxx, syy, zeros, sxy, zeros, zeros, convention)


def convert(
    sigma6: np.ndarray,
    source: VoigtConvention,
    target: VoigtConvention,
) -> np.ndarray:
    """Re-express stress vectors from one convention in another.

    Parameters
    ----------
    sigma6 : ndarray of shape (..., 6)
        Stress vectors in ``source``.
    source, target : VoigtConvention
        Input and output conventions.

    Returns
    -------
    ndarray
        Same shape as the input, expressed in ``target``.
    """
    sigma6 = _as_vectors(sigma6)
    if source == target:
        return sigma6.copy()

    out = np.empty_like(sigma6)
    for target_slot, label in enumerate(target.order):
        src_slot = source.slot_of(label)
        column = sigma6[..., src_slot]
        if label not in ("XX", "YY", "ZZ"):
            if source.mandel and not target.mandel:
                column = column / SQRT2
            elif target.mandel and not source.mandel:
                column = column * SQRT2
        out[..., target_slot] = column
    return out


# --- Invariants --------------------------------------------------------------


def von_mises(sigma6: np.ndarray, convention: VoigtConvention) -> np.ndarray:
    """Von Mises equivalent stress.

    Computed from tensor components recovered through ``convention``, so the
    result is independent of slot order and shear scaling. Any disagreement
    between two conventions therefore shows up here as a number, which is
    what makes this function useful as a cross-check on data ingestion.

    Parameters
    ----------
    sigma6 : ndarray of shape (..., 6)
        Stress vectors, MPa.
    convention : VoigtConvention
        Convention the vectors are expressed in.

    Returns
    -------
    ndarray
        Equivalent stress, MPa.
    """
    sigma6 = _as_vectors(sigma6)
    canonical = convert(sigma6, convention, MANDEL_XY_FIRST)
    xx, yy, zz = canonical[..., 0], canonical[..., 1], canonical[..., 2]
    xy = canonical[..., 3] / SQRT2
    yz = canonical[..., 4] / SQRT2
    xz = canonical[..., 5] / SQRT2

    j2_times_6 = (
        (xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2 + 6.0 * (xy**2 + yz**2 + xz**2)
    )
    return np.sqrt(np.maximum(0.5 * j2_times_6, 0.0))


def norm(sigma6: np.ndarray) -> np.ndarray:
    """Euclidean norm of each stress vector.

    Only equals the tensor Frobenius norm under a Mandel convention.
    """
    return np.linalg.norm(_as_vectors(sigma6), axis=-1)


def unit_direction(sigma6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split stress vectors into unit directions and magnitudes.

    Null vectors (norm below :data:`EPS_NORM`) yield a zero direction and a
    zero magnitude rather than a division warning.

    Returns
    -------
    direction : ndarray of shape (..., 6)
    magnitude : ndarray of shape (...,)
    """
    sigma6 = _as_vectors(sigma6)
    magnitude = np.linalg.norm(sigma6, axis=-1)
    valid = magnitude > EPS_NORM
    direction = np.zeros_like(sigma6)
    direction[valid] = sigma6[valid] / magnitude[valid, np.newaxis]
    return direction, magnitude


# --- Diagnostics -------------------------------------------------------------


def convention_mismatch_report(
    sigma6: np.ndarray,
    assumed: VoigtConvention,
    actual: VoigtConvention,
) -> dict[str, float]:
    """Quantify the cost of interpreting data in the wrong convention.

    Reinterpreting the same numbers under a different convention changes the
    physical state they describe. This reports how much, so the decision to
    keep or fix a legacy pairing rests on a number rather than on intuition.

    Parameters
    ----------
    sigma6 : ndarray of shape (N, 6)
        Raw stress vectors, MPa.
    assumed : VoigtConvention
        Convention the consuming code believes the data is in.
    actual : VoigtConvention
        Convention the data was actually produced in.

    Returns
    -------
    dict
        ``vm_rel_error_max`` / ``vm_rel_error_mean``: relative error in
        equivalent stress. ``norm_ratio_mean``: ratio of 6D norms, which
        drives radial position on the yield surface. ``shear_fraction``:
        share of points where shear carries more than 1% of the norm --
        if this is near zero the mismatch is harmless for this dataset.
    """
    sigma6 = _as_vectors(sigma6)
    vm_assumed = von_mises(sigma6, assumed)
    vm_actual = von_mises(sigma6, actual)

    denominator = np.maximum(np.abs(vm_actual), EPS_NORM)
    rel_error = np.abs(vm_assumed - vm_actual) / denominator

    shear_energy = np.sum(sigma6[:, 3:] ** 2, axis=1)
    total_energy = np.maximum(np.sum(sigma6**2, axis=1), EPS_NORM)
    shear_fraction = float(np.mean(shear_energy / total_energy > 0.01))

    return {
        "vm_rel_error_max": float(np.max(rel_error)),
        "vm_rel_error_mean": float(np.mean(rel_error)),
        "norm_ratio_mean": float(
            np.mean(
                norm(convert(sigma6, actual, assumed)) / np.maximum(norm(sigma6), EPS_NORM)
            )
        ),
        "shear_fraction": shear_fraction,
    }


def _as_vectors(sigma6: np.ndarray) -> np.ndarray:
    """Validate and normalise input to a float64 array with a trailing 6-axis."""
    arr = np.asarray(sigma6, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.shape[-1] != VOIGT_DIM:
        raise ValueError(f"expected trailing axis of size {VOIGT_DIM}, got {arr.shape}")
    return arr
