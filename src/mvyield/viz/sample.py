"""Sampling fields along paths and at points.

Profiles in the pre-refactor code were fixed in place: the probe sat at
``theta = 90``, ``r = a``, the axis was ``x/a``, and the sweep ran around a
hole. All of that is Kirsch. A wire needs profiles along a radius, along the
axis, around the circumference -- none of which exist in that code.

Paths are therefore declared in the config and interpolated here, so one
profile figure serves every problem. Supported kinds:

``line``
    Straight segment between two points.
``arc``
    Circular arc in a coordinate plane.
``ray``
    A line in polar form, with the radius optionally in multiples of the
    geometry's reference length. This is the Kirsch case expressed as
    configuration rather than as code.

Points may also be located by rule rather than by coordinates. ``argmax``
finds the extremum of a named field, which is what the hard-coded
``theta = 90, r = a`` probe was really trying to be: on a wire the critical
location is not known in advance and must be found.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from mvyield.mechanics.field import StressField

log = logging.getLogger(__name__)


def build_path(spec: dict, length_scale: float | None = None) -> np.ndarray:
    """Turn a path specification into sample coordinates.

    Parameters
    ----------
    spec : dict
        Entry from ``probes.paths``.
    length_scale : float, optional
        Reference length, used when a ray declares ``units: hole_radii``.

    Returns
    -------
    ndarray of shape (n, 3)
        Sample points.
    """
    kind = spec.get("kind")
    count = int(spec.get("n", 200))

    if kind == "line":
        start = np.asarray(spec["from"], dtype=np.float64)
        end = np.asarray(spec["to"], dtype=np.float64)
        parameter = np.linspace(0.0, 1.0, count)[:, None]
        return _to_3d(start[None, :] + parameter * (end - start)[None, :])

    if kind == "arc":
        centre = np.asarray(spec.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
        radius = float(spec["radius"])
        plane = str(spec.get("plane", "xy")).lower()
        start = np.deg2rad(float(spec.get("from_deg", 0.0)))
        end = np.deg2rad(float(spec.get("to_deg", 360.0)))

        angle = np.linspace(start, end, count)
        points = np.tile(_to_3d(centre[None, :]), (count, 1))
        first, second = _plane_axes(plane)
        points[:, first] = centre[first] + radius * np.cos(angle)
        points[:, second] = centre[second] + radius * np.sin(angle)
        return points

    if kind == "ray":
        angle = np.deg2rad(float(spec.get("angle_deg", 0.0)))
        r_from = float(spec.get("r_from", 0.0))
        r_to = float(spec["r_to"])

        if str(spec.get("units", "")) == "hole_radii":
            if length_scale is None:
                raise ValueError(
                    f"path {spec.get('name')!r} uses units 'hole_radii', but the "
                    f"case supplies no reference length. Give coordinates in "
                    f"physical units, or have the case set a length scale."
                )
            r_from, r_to = r_from * length_scale, r_to * length_scale

        radius = np.linspace(r_from, r_to, count)
        points = np.zeros((count, 3))
        points[:, 0] = radius * np.cos(angle)
        points[:, 1] = radius * np.sin(angle)
        return points

    raise ValueError(f"unknown path kind {kind!r}; expected line, arc or ray")


def sample_along(
    field: StressField,
    values: np.ndarray,
    path: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate a scalar field onto path coordinates.

    Parameters
    ----------
    field : StressField
        Field the values belong to.
    values : ndarray of shape (M,)
        Scalar field to sample.
    path : ndarray of shape (n, 3)
        Sample coordinates.
    method : str
        ``"linear"`` interpolates and returns ``NaN`` outside the convex
        hull; ``"nearest"`` always returns a value.

    Returns
    -------
    ndarray of shape (n,)
        Sampled values, ``NaN`` where the path leaves the field.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) != field.n_points:
        raise ValueError(
            f"values has {len(values)} entries for a field of {field.n_points} points"
        )

    usable = field.valid_mask() & np.isfinite(values)
    if not usable.any():
        return np.full(len(path), np.nan)

    dimensions = 2 if not field.topology.is_3d else 3
    source = field.points[usable, :dimensions]
    target = np.asarray(path, dtype=np.float64)[:, :dimensions]

    sampled = griddata(source, values[usable], target, method=method)

    outside = np.isnan(sampled)
    if outside.any() and method == "linear":
        log.debug("%d of %d path samples fall outside the field", outside.sum(), len(path))
    return sampled


def sample_stress_along(
    field: StressField, path: np.ndarray, method: str = "nearest"
) -> np.ndarray:
    """Sample the stress vector along a path.

    Defaults to nearest-neighbour: interpolating between stress states
    produces a state that never occurred anywhere in the body, which is
    misleading when the sample is then fed back to a yield model.
    """
    dimensions = 2 if not field.topology.is_3d else 3
    usable = field.valid_mask()

    source = field.points[usable, :dimensions]
    target = np.asarray(path, dtype=np.float64)[:, :dimensions]

    if method == "nearest":
        _, index = cKDTree(source).query(target)
        return field.sigma6[usable][index]

    return np.column_stack(
        [
            griddata(source, field.sigma6[usable, component], target, method=method)
            for component in range(6)
        ]
    )


def locate_point(spec: dict, field: StressField, scalars: dict[str, np.ndarray]) -> int:
    """Find a probe point by rule and return its index.

    Parameters
    ----------
    spec : dict
        Entry from ``probes.points``.
    field : StressField
        The field to search.
    scalars : dict of str to ndarray
        Named scalar fields available for ``argmax`` and ``argmin``.

    Returns
    -------
    int
        Index into the field.
    """
    kind = spec.get("kind", "argmax")

    if kind in ("argmax", "argmin"):
        name = spec.get("of", "P_primary")
        try:
            values = np.asarray(scalars[name], dtype=np.float64)
        except KeyError:
            raise KeyError(
                f"probe {spec.get('name')!r} refers to field {name!r}, which was "
                f"not computed; available: {sorted(scalars)}"
            ) from None

        candidates = np.where(field.valid_mask() & np.isfinite(values), values, np.nan)
        if not np.isfinite(candidates).any():
            raise ValueError(
                f"probe {spec.get('name')!r}: field {name!r} has no finite values "
                f"at any valid point"
            )
        return int(np.nanargmax(candidates) if kind == "argmax" else np.nanargmin(candidates))

    if kind == "coordinate":
        target = _to_3d(np.asarray(spec["at"], dtype=np.float64)[None, :])
        dimensions = 2 if not field.topology.is_3d else 3
        distance = np.linalg.norm(
            field.points[:, :dimensions] - target[:, :dimensions], axis=1
        )
        distance[~field.valid_mask()] = np.inf
        return int(np.argmin(distance))

    raise ValueError(f"unknown probe kind {kind!r}; expected argmax, argmin or coordinate")


def path_abscissa(path: np.ndarray, length_scale: float | None = None) -> np.ndarray:
    """Arc length along a path, for use as the profile x-axis.

    Normalised by the reference length when the case provides one, so the
    axis reads ``s/a``; otherwise it stays in physical units.
    """
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(steps)])
    return distance / length_scale if length_scale else distance


def _to_3d(points: np.ndarray) -> np.ndarray:
    """Pad 2D coordinates with a zero third column."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if points.shape[1] == 3:
        return points
    if points.shape[1] == 2:
        return np.column_stack([points, np.zeros(len(points))])
    raise ValueError(f"coordinates must have 2 or 3 components, got {points.shape[1]}")


def _plane_axes(plane: str) -> tuple[int, int]:
    """Column indices of the two axes spanning a coordinate plane."""
    planes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    try:
        return planes[plane]
    except KeyError:
        raise ValueError(f"plane must be xy, xz or yz; got {plane!r}") from None
