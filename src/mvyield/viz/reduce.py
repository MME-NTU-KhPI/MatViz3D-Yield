"""Reducing a 3D field to something that can be drawn on paper.

A wire is a volume. Field figures are two-dimensional. Rather than teach
every figure about three dimensions, the field is reduced first and the
existing figures are reused unchanged -- the same ones that served the plate.

Reductions come from ``view.reductions`` in the case config, so choosing what
to look at is a configuration decision rather than a code change. A 2D
problem declares none and behaves exactly as before.

``projection`` with ``reduce: max`` is the most useful for a wire and the
least obvious: it collapses the axial coordinate by taking the worst value
along it, so one picture answers "where in the cross-section is the danger,
accounting for the whole length". A slice answers a narrower question and
can miss the critical station entirely.

For genuine three-dimensional inspection, export to ParaView instead --
see :func:`mvyield.viz.io.export_vtu`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from mvyield.mechanics.field import FieldMeta, StressField, Topology

log = logging.getLogger(__name__)

AXES = {"x": 0, "y": 1, "z": 2}


@dataclass
class ReducedField:
    """A 2D view of a field, with the scalar data carried along.

    Parameters
    ----------
    name : str
        Label used as a filename suffix and in figure titles.
    field : StressField
        The reduced field, always a 2D topology.
    scalars : dict of str to ndarray
        Reduced scalar fields, keyed as ``"<method>/<quantity>"``.
    description : str
        One line describing what the reduction did, for the figure caption.
    """

    name: str
    field: StressField
    scalars: dict[str, np.ndarray]
    description: str = ""


def apply_reductions(
    field: StressField,
    scalars: dict[str, np.ndarray],
    reductions: tuple[dict, ...],
) -> list[ReducedField]:
    """Produce every 2D view requested for a field.

    Parameters
    ----------
    field : StressField
        The field to reduce.
    scalars : dict of str to ndarray
        Per-point scalar fields to carry through the reduction.
    reductions : tuple of dict
        Entries from ``view.reductions``.

    Returns
    -------
    list of ReducedField
        For a 2D field with no reductions configured, a single passthrough
        view, so callers need no special case.
    """
    if not field.topology.is_3d:
        if reductions:
            log.info(
                "field is already 2D; ignoring %d configured reduction(s)", len(reductions)
            )
        return [ReducedField(name="", field=field, scalars=scalars)]

    if not reductions:
        raise ValueError(
            "this field is three-dimensional but no view.reductions are "
            "configured, so field maps cannot be drawn. Add at least one "
            "reduction (slice, unroll or projection), or set export.vtk to "
            "true and inspect the field in ParaView."
        )

    views: list[ReducedField] = []
    for index, spec in enumerate(reductions):
        kind = spec.get("kind")
        name = spec.get("name") or f"{kind}_{index}"

        if kind == "slice":
            view = _slice(field, scalars, spec, name)
        elif kind == "projection":
            view = _projection(field, scalars, spec, name)
        elif kind == "unroll":
            view = _unroll(field, scalars, spec, name)
        else:
            raise ValueError(f"unknown reduction kind {kind!r}")

        if view.field.n_points == 0:
            log.warning("[skip] reduction %r selected no points", name)
            continue
        views.append(view)

    return views


def _slice(field, scalars, spec, name) -> ReducedField:
    """Take points lying near a plane of constant coordinate."""
    axis = _axis_index(spec.get("plane", "z"))
    value = float(spec.get("value", 0.0))
    tolerance = float(spec.get("tol", 1e-4))

    mask = np.abs(field.points[:, axis] - value) <= tolerance
    if not mask.any():
        coords = field.points[:, axis]
        raise ValueError(
            f"reduction {name!r} selected no points: no coordinate within "
            f"{tolerance:g} of {value:g} along {spec.get('plane', 'z')} "
            f"(range {coords.min():.4g} to {coords.max():.4g}). Widen 'tol' "
            f"or move 'value' onto a node plane."
        )

    kept = _remaining_axes(axis)
    points = np.zeros((int(mask.sum()), 3))
    points[:, 0] = field.points[mask, kept[0]]
    points[:, 1] = field.points[mask, kept[1]]

    return ReducedField(
        name=name,
        field=_rebuild(field, points, field.sigma6[mask], mask),
        scalars={key: values[mask] for key, values in scalars.items()},
        description=(
            f"slice at {spec.get('plane', 'z')} = {value:g} (tolerance {tolerance:g})"
        ),
    )


def _projection(field, scalars, spec, name) -> ReducedField:
    """Collapse one axis by reducing over it.

    Points are binned by their remaining two coordinates and the requested
    statistic is taken within each bin. With ``reduce: max`` this gives the
    envelope: the worst value anywhere along the collapsed axis.

    The stress vector reported for a bin is the one belonging to the
    extremum of the first scalar field, so the stress state shown is a state
    that actually occurred rather than an average of several.
    """
    axis = _axis_index(spec.get("axis", "z"))
    how = str(spec.get("reduce", "max"))
    decimals = int(spec.get("decimals", 6))

    if how not in ("max", "min", "mean"):
        raise ValueError(f"projection reduce must be max, min or mean; got {how!r}")

    kept = _remaining_axes(axis)
    keys = np.round(field.points[:, kept], decimals)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    n_bins = len(counts)

    driver = _driver_field(scalars)
    representative = np.zeros(n_bins, dtype=int)

    for bin_index in range(n_bins):
        members = np.flatnonzero(inverse == bin_index)
        if driver is None:
            representative[bin_index] = members[0]
        else:
            values = driver[members]
            finite = np.isfinite(values)
            if not finite.any():
                representative[bin_index] = members[0]
            elif how == "min":
                representative[bin_index] = members[finite][np.argmin(values[finite])]
            else:
                representative[bin_index] = members[finite][np.argmax(values[finite])]

    points = np.zeros((n_bins, 3))
    points[:, :2] = np.unique(keys, axis=0)

    reduced: dict[str, np.ndarray] = {}
    for key, values in scalars.items():
        reduced[key] = _reduce_by_bin(values, inverse, n_bins, how)

    mask = np.zeros(field.n_points, dtype=bool)
    mask[representative] = True

    return ReducedField(
        name=name,
        field=_rebuild(field, points, field.sigma6[representative], mask),
        scalars=reduced,
        description=f"{how} projected along {spec.get('axis', 'z')}",
    )


def _unroll(field, scalars, spec, name) -> ReducedField:
    """Map an axisymmetric body onto radius and axial position.

    For a wire this turns the cross-section into a single ``(r, z)`` plane,
    which is a fair view precisely when the body is axisymmetric. It is not
    a general reduction and will misrepresent a body that is not.
    """
    axis = _axis_index(spec.get("axis", "z"))
    kept = _remaining_axes(axis)

    radius = np.hypot(field.points[:, kept[0]], field.points[:, kept[1]])
    points = np.zeros((field.n_points, 3))
    points[:, 0] = radius
    points[:, 1] = field.points[:, axis]

    return ReducedField(
        name=name,
        field=_rebuild(field, points, field.sigma6, np.ones(field.n_points, dtype=bool)),
        scalars=dict(scalars),
        description=f"unrolled to (r, {spec.get('axis', 'z')})",
    )


def _rebuild(field: StressField, points, sigma6, mask) -> StressField:
    """Assemble a 2D field from reduced coordinates."""
    valid = None if field.valid is None else field.valid[mask]
    return StressField(
        points=points,
        sigma6=sigma6,
        topology=Topology.CLOUD2D,
        meta=FieldMeta(
            source=field.meta.source,
            convention=field.meta.convention,
            stress_unit=field.meta.stress_unit,
            length_unit=field.meta.length_unit,
            frame=field.meta.frame,
            extra=dict(field.meta.extra),
        ),
        valid=valid,
        hints=field.hints,
    )


def _reduce_by_bin(values, inverse, n_bins, how) -> np.ndarray:
    """Reduce a scalar field within bins, tolerating NaN and empty bins."""
    if values.dtype == bool:
        # A boolean field such as coverage: a bin counts as covered when any
        # member is. Taking a mean here would silently produce a fraction.
        out = np.zeros(n_bins, dtype=bool)
        np.logical_or.at(out, inverse, values)
        return out

    out = np.full(n_bins, np.nan, dtype=np.float64)
    finite = np.isfinite(values)

    if how == "mean":
        totals = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        np.add.at(totals, inverse[finite], values[finite])
        np.add.at(counts, inverse[finite], 1.0)
        nonempty = counts > 0
        out[nonempty] = totals[nonempty] / counts[nonempty]
        return out

    fill = -np.inf if how == "max" else np.inf
    working = np.full(n_bins, fill, dtype=np.float64)
    operation = np.maximum if how == "max" else np.minimum
    operation.at(working, inverse[finite], values[finite])

    touched = np.isfinite(working)
    out[touched] = working[touched]
    return out


def _driver_field(scalars: dict[str, np.ndarray]) -> np.ndarray | None:
    """Pick the scalar field that decides which point represents a bin."""
    for key in scalars:
        if key.endswith("/probability"):
            return scalars[key]
    return next(iter(scalars.values()), None)


def _axis_index(name: str) -> int:
    """Translate an axis name to a column index."""
    try:
        return AXES[str(name).lower()]
    except KeyError:
        raise ValueError(f"axis must be x, y or z; got {name!r}") from None


def _remaining_axes(axis: int) -> tuple[int, int]:
    """Return the two axes left after collapsing one."""
    return tuple(i for i in range(3) if i != axis)  # type: ignore[return-value]
