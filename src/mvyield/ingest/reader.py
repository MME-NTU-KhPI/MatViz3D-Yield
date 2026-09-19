"""Reading a MatViz3D dataset: geometries, load steps and what they carry.

Layout
------
::

    /<G>/                     one RVE, group name is a number
        voxels        (n,n,n) grain id per voxel
        local_cs      (g,3)   Bunge Euler angles per grain, degrees
        cubeSize      scalar  voxels along an edge
        P_matrix      (6,6)   normalised elastic coefficients
        ls_<L>/               one load step
            results         (N,19)
            results_avg     (19,)
            results_max     (19,)
            results_min     (19,)
            eps_as_loading  (6,)
    /ground_truth/            present in generated datasets only
    /last_set                 scalar bookkeeping marker

The ``results`` columns are fixed by the C++ writer:

===========  ==========================================
0            node id
1, 2, 3      x, y, z voxel indices
4, 5, 6      reserved -- zero in generated files,
             displacements in exported ones
7 .. 12      SX, SY, SZ, SXY, SYZ, SXZ, Pa
13 .. 18     EX, EY, EZ, GXY, GYZ, GXZ, dimensionless
===========  ==========================================

Two things this module does that the legacy reader did not
----------------------------------------------------------
It reads **lazily**. ``HDF5DataProcessor`` stacked every geometry, every
step and every dataset into one object array on first access; for the
880 MB noise datasets that is the whole file in memory before any analysis
starts. Here a step is read when it is asked for.

It identifies geometries **structurally** -- a root group holding at least
one ``ls_*`` subgroup -- rather than by excluding names known to be special.
``last_set`` is a scalar dataset and ``ground_truth`` holds no load steps,
so both fall out without a list to maintain, and a dataset that invents a
new sibling group will not be mistaken for an RVE.

Stress arrives in Pa and leaves in MPa, because every other module in this
package states its stresses in MPa and a unit that changes at a boundary is
a unit that eventually does not.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mvyield.mechanics import voigt
from mvyield.mechanics.voigt import LEGACY_DATASET, VoigtConvention

try:
    import h5py
except ImportError as exc:  # pragma: no cover - exercised by the doctor
    raise ImportError(
        "reading a dataset needs h5py, which is not installed. "
        "Install it with: pip install -e \".[ingest]\""
    ) from exc


N_COLUMNS = 19
"""Width of a ``results`` block."""

COL_NODE_ID = 0
COL_COORDS = slice(1, 4)
COL_STRESS = slice(7, 13)
COL_STRAIN = slice(13, 19)

DATASET_CONVENTION: VoigtConvention = LEGACY_DATASET
"""Order the file stores stress in: XX, YY, ZZ, XY, YZ, XZ, no sqrt(2).

Confirmed by the ``component_order`` attribute that generated datasets
carry, and by the column layout of the C++ writer for exported ones.
"""

PA_TO_MPA = 1e-6

STEP_PREFIX = "ls_"


class DatasetError(RuntimeError):
    """Raised when a file does not have the structure this reader expects."""


@dataclass(frozen=True)
class Geometry:
    """One RVE: its microstructure and the frame its grains are given in.

    Parameters
    ----------
    id : str
        Group name, kept as text because it identifies the RVE in reports.
    index : int
        Position in the reader's ordering, counting from zero regardless of
        whether the file numbers its groups from zero or one.
    n_steps : int
        Load steps recorded for this geometry.
    cube_size : int or None
        Voxels along an edge.
    voxels : ndarray or None
        Grain id per voxel, shape ``(n, n, n)``.
    euler_deg : ndarray or None
        Bunge Euler angles per grain, degrees, shape ``(n_grains, 3)``.
    p_matrix : ndarray or None
        Normalised elastic coefficients, shape ``(6, 6)``.
    """

    id: str
    index: int
    n_steps: int
    cube_size: int | None = None
    voxels: np.ndarray | None = None
    euler_deg: np.ndarray | None = None
    p_matrix: np.ndarray | None = None

    @property
    def n_grains(self) -> int:
        """Number of grains, from the orientation table."""
        return 0 if self.euler_deg is None else len(self.euler_deg)


@dataclass(frozen=True)
class Step:
    """One load step of one geometry, in MPa.

    Parameters
    ----------
    geometry : str
        Owning geometry id.
    index : int
        Step number, as written in the ``ls_<L>`` group name.
    node_id : ndarray of shape (N,)
        Node identifiers as recorded.
    coords : ndarray of shape (N, 3)
        Voxel indices.
    stress : ndarray of shape (N, 6)
        Per-point stress, MPa, in :data:`DATASET_CONVENTION`.
    strain : ndarray of shape (N, 6)
        Per-point strain, engineering shear, dimensionless.
    stress_avg : ndarray of shape (6,)
        Volume average from ``results_avg``, MPa.
    strain_avg : ndarray of shape (6,)
        Volume average of the strain columns.
    eps_as_loading : ndarray of shape (6,)
        Applied strain that drove this step.
    """

    geometry: str
    index: int
    node_id: np.ndarray
    coords: np.ndarray
    stress: np.ndarray
    strain: np.ndarray
    stress_avg: np.ndarray
    strain_avg: np.ndarray
    eps_as_loading: np.ndarray

    @property
    def n_points(self) -> int:
        """Number of material points."""
        return len(self.stress)

    def stress_in(self, convention: VoigtConvention) -> np.ndarray:
        """Per-point stress re-expressed in another convention, MPa."""
        return voigt.convert(self.stress, DATASET_CONVENTION, convention)

    def stress_avg_in(self, convention: VoigtConvention) -> np.ndarray:
        """Volume-averaged stress re-expressed in another convention, MPa."""
        return voigt.convert(self.stress_avg, DATASET_CONVENTION, convention)


class DatasetReader:
    """Lazy reader over one dataset file.

    The file handle stays open for the reader's lifetime, so use it as a
    context manager::

        with DatasetReader(path) as reader:
            for step in reader.steps("0"):
                ...

    Parameters
    ----------
    path : Path or str
        Dataset file.
    stress_scale : float, optional
        Factor converting stored stress to MPa. Defaults to
        :data:`PA_TO_MPA`, matching every dataset this project has produced;
        pass ``1.0`` for a file already written in MPa.
    """

    def __init__(self, path: Path | str, stress_scale: float = PA_TO_MPA) -> None:
        self.path = Path(path)
        self.stress_scale = float(stress_scale)
        self._file: h5py.File | None = None
        self._geometry_ids: tuple[str, ...] = ()

    # --- Lifetime ------------------------------------------------------------

    def open(self) -> DatasetReader:
        """Open the file and index its geometries."""
        if self._file is not None:
            return self

        if not self.path.exists():
            raise FileNotFoundError(f"dataset not found: {self.path}")

        self._file = h5py.File(self.path, "r")
        self._geometry_ids = _find_geometries(self._file)

        if not self._geometry_ids:
            self.close()
            raise DatasetError(
                f"{self.path.name} holds no geometry groups. A geometry is a "
                f"group containing at least one '{STEP_PREFIX}*' subgroup; "
                f"this file's root has none."
            )
        return self

    def close(self) -> None:
        """Close the file. Safe to call more than once."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> DatasetReader:
        """Open on entry."""
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        """Close on exit, including on an exception."""
        self.close()

    # --- Structure -----------------------------------------------------------

    @property
    def attrs(self) -> dict:
        """Root attributes, with JSON-valued ones already decoded."""
        return _decode_attrs(self._handle().attrs)

    @property
    def geometry_ids(self) -> tuple[str, ...]:
        """Geometry group names, ordered numerically."""
        self._handle()
        return self._geometry_ids

    @property
    def n_geometries(self) -> int:
        """Number of RVEs in the file."""
        return len(self.geometry_ids)

    def n_steps(self, geometry: str) -> int:
        """Number of load steps recorded for one geometry."""
        return len(self._step_names(geometry))

    def geometry(self, geometry: str) -> Geometry:
        """Read one geometry's microstructure."""
        group = self._group(geometry)

        return Geometry(
            id=geometry,
            index=self.geometry_ids.index(geometry),
            n_steps=self.n_steps(geometry),
            cube_size=_scalar(group, "cubeSize"),
            voxels=_array(group, "voxels"),
            euler_deg=_array(group, "local_cs"),
            p_matrix=_array(group, "P_matrix"),
        )

    def geometries(self) -> Iterator[Geometry]:
        """Iterate over every geometry."""
        for geometry_id in self.geometry_ids:
            yield self.geometry(geometry_id)

    # --- Load steps ----------------------------------------------------------

    def step(self, geometry: str, index: int) -> Step:
        """Read one load step.

        Parameters
        ----------
        geometry : str
            Geometry group name.
        index : int
            Position in the step ordering, not necessarily the ``L`` in
            ``ls_<L>`` -- the two coincide for every dataset seen so far,
            but the ordering is what iteration promises.
        """
        names = self._step_names(geometry)
        try:
            name = names[index]
        except IndexError:
            raise IndexError(
                f"geometry {geometry!r} has {len(names)} steps; asked for {index}"
            ) from None
        return self._read_step(geometry, name)

    def steps(self, geometry: str) -> Iterator[Step]:
        """Iterate over a geometry's load steps in order."""
        for name in self._step_names(geometry):
            yield self._read_step(geometry, name)

    def raw(self, geometry: str, step_name: str, dataset: str) -> np.ndarray:
        """Read one dataset verbatim, without unit conversion or reshaping.

        An escape hatch for the parts of the file this reader does not
        model -- ``results_max`` and ``results_min``, say -- so that using
        them does not require changing this class first.
        """
        group = self._group(geometry)
        if step_name not in group or dataset not in group[step_name]:
            raise KeyError(f"{geometry}/{step_name}/{dataset} is not in {self.path.name}")
        return np.asarray(group[step_name][dataset][()])

    def coordinate_extent(self, geometry: str | None = None) -> float:
        """Width of the coordinate box, as the analytical case uses it.

        The smaller of the x and y spans, matching what the legacy loader
        computed. It sizes the plotting window of a scale-free analytical
        solution -- see :mod:`mvyield.cases.kirsch` -- so it is a drawing
        decision rather than a physical one, and the RVE's voxel grid having
        no relation to a plate is not the problem it appears to be.
        """
        geometry = geometry or self.geometry_ids[0]
        coords = self.step(geometry, 0).coords
        spans = np.ptp(coords[:, :2], axis=0)
        return float(spans.min())

    # --- Reference values ----------------------------------------------------

    @property
    def has_ground_truth(self) -> bool:
        """Whether the file carries reference values from a generator."""
        return "ground_truth" in self._handle()

    def ground_truth(self, geometry: str) -> dict[str, np.ndarray]:
        """Reference values for one geometry, when the file has them.

        Present only in generated datasets. The pipeline never reads these
        -- they exist to score it.
        """
        handle = self._handle()
        if "ground_truth" not in handle or geometry not in handle["ground_truth"]:
            raise KeyError(
                f"{self.path.name} carries no ground truth for geometry "
                f"{geometry!r}; only generated datasets do"
            )
        group = handle["ground_truth"][geometry]
        return {name: np.asarray(group[name][()]) for name in group}

    # --- Reporting -----------------------------------------------------------

    def describe(self) -> str:
        """Summary for logs and the run report."""
        first = self.geometry(self.geometry_ids[0])
        steps = self.n_steps(self.geometry_ids[0])

        lines = [
            f"file        : {self.path.name}",
            f"geometries  : {self.n_geometries} ({self.geometry_ids[0]}..{self.geometry_ids[-1]})",
            f"steps       : {steps} per geometry",
            f"cube        : {first.cube_size} voxels, {first.n_grains} grains",
            f"ground truth: {'yes' if self.has_ground_truth else 'no'}",
        ]
        if "noise_level" in self.attrs:
            lines.append(f"noise       : {self.attrs['noise_level']}")
        return "\n".join(lines)

    # --- Internals -----------------------------------------------------------

    def _handle(self) -> h5py.File:
        """Return the open file, or say why there is not one."""
        if self._file is None:
            raise DatasetError(
                f"the reader for {self.path.name} is closed; use it as a "
                f"context manager, or call open() first"
            )
        return self._file

    def _group(self, geometry: str) -> h5py.Group:
        """Look up a geometry group by name."""
        handle = self._handle()
        if geometry not in self._geometry_ids:
            raise KeyError(
                f"geometry {geometry!r} is not in {self.path.name}; "
                f"available: {list(self._geometry_ids)}"
            )
        return handle[geometry]

    def _step_names(self, geometry: str) -> tuple[str, ...]:
        """Ordered ``ls_*`` group names of one geometry."""
        return _sorted_steps(self._group(geometry))

    def _read_step(self, geometry: str, name: str) -> Step:
        """Read and unpack one ``ls_*`` group."""
        group = self._group(geometry)[name]
        results = _require(group, "results", f"{geometry}/{name}")

        if results.ndim != 2 or results.shape[1] < N_COLUMNS:
            raise DatasetError(
                f"{geometry}/{name}/results has shape {results.shape}; "
                f"expected (N, {N_COLUMNS}) -- this file does not use the "
                f"MatViz3D column layout"
            )

        averages = _row(group, "results_avg")

        return Step(
            geometry=geometry,
            index=_step_number(name),
            node_id=results[:, COL_NODE_ID].astype(np.int64),
            coords=results[:, COL_COORDS].astype(np.float64),
            stress=results[:, COL_STRESS].astype(np.float64) * self.stress_scale,
            strain=results[:, COL_STRAIN].astype(np.float64),
            stress_avg=averages[COL_STRESS] * self.stress_scale,
            strain_avg=averages[COL_STRAIN],
            eps_as_loading=_row(group, "eps_as_loading", width=6),
        )


# --- Module helpers ----------------------------------------------------------


def _find_geometries(handle: h5py.File) -> tuple[str, ...]:
    """Root groups that hold load steps, ordered numerically.

    Numeric names sort by value so that ``10`` follows ``9``; anything else
    sorts after them, alphabetically, so an unconventional name is kept
    rather than silently dropped.
    """
    names = [
        name
        for name, node in handle.items()
        if isinstance(node, h5py.Group) and any(k.startswith(STEP_PREFIX) for k in node)
    ]
    return tuple(sorted(names, key=lambda n: (0, int(n), "") if n.isdigit() else (1, 0, n)))


def _sorted_steps(group: h5py.Group) -> tuple[str, ...]:
    """``ls_*`` subgroup names, ordered by their step number."""
    names = [k for k in group if k.startswith(STEP_PREFIX)]
    return tuple(sorted(names, key=_step_number))


def _step_number(name: str) -> int:
    """Step number from an ``ls_<L>`` group name, or -1 when it has none."""
    tail = name[len(STEP_PREFIX) :]
    return int(tail) if tail.isdigit() else -1


def _require(group: h5py.Group, name: str, where: str) -> np.ndarray:
    """Read a dataset that must be present."""
    if name not in group:
        raise DatasetError(f"{where} has no {name!r} dataset")
    return np.asarray(group[name][()])


def _array(group: h5py.Group, name: str) -> np.ndarray | None:
    """Read an optional dataset as an array."""
    if name not in group or not isinstance(group[name], h5py.Dataset):
        return None
    return np.asarray(group[name][()])


def _scalar(group: h5py.Group, name: str) -> int | None:
    """Read an optional scalar dataset as a plain int."""
    value = _array(group, name)
    if value is None:
        return None
    return int(value.reshape(-1)[0]) if value.size else None


def _row(group: h5py.Group, name: str, width: int = N_COLUMNS) -> np.ndarray:
    """Read a 1D dataset, returning zeros of the right width when absent.

    ``results_avg`` and ``eps_as_loading`` are conveniences rather than
    load-bearing data -- the per-point block carries the same information --
    so a file missing one is worth continuing with.
    """
    value = _array(group, name)
    if value is None:
        return np.zeros(width, dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _decode_attrs(attrs: h5py.AttributeManager) -> dict:
    """Copy HDF5 attributes into a plain dict, decoding JSON strings.

    ``gen_config`` and ``noise_spec`` are stored as JSON text. Decoding here
    means callers read ``attrs["gen_config"]["crss"]`` instead of parsing
    the same string in three places.
    """
    out: dict = {}
    for key, value in attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, str) and value.startswith(("{", "[")):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        out[key] = value
    return out
