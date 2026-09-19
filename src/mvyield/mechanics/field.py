"""Stress field container shared by every case.

An analytical solution and an ANSYS export differ only in how they are
produced. Once loaded they are the same thing: points in space, a stress
vector at each point, and enough metadata to plot them. Expressing that as
one type is what lets a new problem be a config file rather than new code.

Topology is carried explicitly because it, not the name of the case, decides
how a field can be rendered: a regular grid takes ``imshow``, a scattered
plane takes triangulation, a 3D cloud needs a slice or a projection first.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum

import numpy as np

from .voigt import VoigtConvention


class Topology(StrEnum):
    """How field points are arranged in space."""

    GRID2D = "grid2d"
    """Regular X-Y grid; ``grid_shape`` is set."""

    CLOUD2D = "cloud2d"
    """Scattered points confined to a plane."""

    CLOUD3D = "cloud3d"
    """Scattered points in 3D, e.g. nodal FE results without connectivity."""

    MESH3D = "mesh3d"
    """3D points plus element connectivity."""

    @property
    def is_3d(self) -> bool:
        """Whether the topology needs a reduction before 2D rendering."""
        return self in (Topology.CLOUD3D, Topology.MESH3D)


@dataclass(frozen=True)
class Outline:
    """A curve drawn over a field plot, such as a hole boundary.

    Parameters
    ----------
    kind : str
        ``"circle"`` or ``"polyline"``.
    points : ndarray of shape (N, 2)
        Vertices for a polyline; the centre as a single row for a circle.
    radius : float, optional
        Circle radius; ignored for polylines.
    """

    kind: str
    points: np.ndarray
    radius: float | None = None


@dataclass
class GeometryHints:
    """Optional presentation hints a case may attach to its field.

    All fields are optional by design: a case that supplies nothing still
    plots correctly, just with physical axis units and no overlay. This is
    what removes ``hole_radius`` and ``hole_center`` from the signature of
    every plotting function.

    Parameters
    ----------
    length_scale : float, optional
        Reference length used to normalise axes (hole radius for Kirsch,
        wire radius for the wire case). ``None`` keeps physical units.
    length_symbol : str
        Symbol for the reference length in axis labels, e.g. ``"a"`` gives
        ``x/a``.
    length_unit : str
        Unit shown when no normalisation applies.
    outlines : list of Outline
        Curves drawn over field plots.
    """

    length_scale: float | None = None
    length_symbol: str = "a"
    length_unit: str = "mm"
    outlines: list[Outline] = dataclass_field(default_factory=list)


@dataclass
class FieldMeta:
    """Provenance of a stress field."""

    source: str
    """``"analytic"``, ``"ansys"``, ``"pinn"``, ..."""

    convention: VoigtConvention
    """Convention the stress vectors are expressed in."""

    stress_unit: str = "MPa"
    length_unit: str = "mm"
    frame: str = "global"
    """Coordinate frame; only ``"global"`` may be compared against a model."""

    grid_shape: tuple[int, int] | None = None
    """Row/column shape for :attr:`Topology.GRID2D`, else ``None``."""

    extra: dict = dataclass_field(default_factory=dict)


@dataclass
class StressField:
    """Points, stresses and metadata for one problem.

    Parameters
    ----------
    points : ndarray of shape (M, 3)
        Coordinates. 2D problems carry a zero third column so that a single
        code path handles both.
    sigma6 : ndarray of shape (M, 6)
        Raw stress vectors in ``meta.convention``, MPa.
    topology : Topology
        Arrangement of the points.
    meta : FieldMeta
        Provenance and units.
    valid : ndarray of shape (M,), optional
        Boolean mask; ``False`` marks excluded points such as the interior of
        a hole. ``None`` means every point is valid.
    hints : GeometryHints
        Optional presentation hints.
    """

    points: np.ndarray
    sigma6: np.ndarray
    topology: Topology
    meta: FieldMeta
    valid: np.ndarray | None = None
    hints: GeometryHints = dataclass_field(default_factory=GeometryHints)

    def __post_init__(self) -> None:
        """Validate shapes and coerce dtypes."""
        self.points = np.asarray(self.points, dtype=np.float64)
        self.sigma6 = np.asarray(self.sigma6, dtype=np.float64)

        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must have shape (M, 3), got {self.points.shape}")
        if self.sigma6.ndim != 2 or self.sigma6.shape[1] != 6:
            raise ValueError(f"sigma6 must have shape (M, 6), got {self.sigma6.shape}")
        if len(self.points) != len(self.sigma6):
            raise ValueError(
                f"points and sigma6 disagree on count: "
                f"{len(self.points)} vs {len(self.sigma6)}"
            )
        if self.valid is not None:
            self.valid = np.asarray(self.valid, dtype=bool)
            if self.valid.shape != (len(self.points),):
                raise ValueError(
                    f"valid must have shape ({len(self.points)},), got {self.valid.shape}"
                )
        if self.topology is Topology.GRID2D and self.meta.grid_shape is None:
            raise ValueError("grid2d topology requires meta.grid_shape")

    @property
    def n_points(self) -> int:
        """Total number of points, including masked ones."""
        return len(self.points)

    @property
    def n_valid(self) -> int:
        """Number of points that participate in evaluation."""
        return self.n_points if self.valid is None else int(self.valid.sum())

    @property
    def x(self) -> np.ndarray:
        """First coordinate column."""
        return self.points[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Second coordinate column."""
        return self.points[:, 1]

    @property
    def z(self) -> np.ndarray:
        """Third coordinate column; all zeros for 2D problems."""
        return self.points[:, 2]

    def valid_mask(self) -> np.ndarray:
        """Boolean mask of valid points, materialised even when unset."""
        if self.valid is None:
            return np.ones(self.n_points, dtype=bool)
        return self.valid

    def summary(self) -> str:
        """One-line description for logs."""
        return (
            f"{self.meta.source} field, {self.topology.value}, "
            f"{self.n_valid}/{self.n_points} valid points, "
            f"|sigma| median {np.median(np.linalg.norm(self.sigma6, axis=1)):.1f} "
            f"{self.meta.stress_unit}"
        )
