"""Finding the yield point of each load step.

The dataset records elastic states: a strain was applied, and the stress it
produced was written down. None of those states is at yield. What makes a
yield point is the observation that the response is linear, so the whole
field can be scaled by a single factor ``k`` until the first material point
reaches the criterion. That scaled state is the yield point, and the cloud
of them over every geometry and every load direction is the yield surface.

Two criteria
------------
``SCHMID``
    Slip begins when the shear resolved onto any slip system of any grain
    reaches the critical resolved shear stress. Orientation-dependent, so
    identical loads yield at different magnitudes in differently oriented
    grains -- which is precisely the scatter the probabilistic model exists
    to describe.

``VON_MISES``
    Yielding when the equivalent stress reaches a scalar strength. Ignores
    orientation; kept because it is the textbook comparison.

The seven-point orientation search
----------------------------------
For each material point the Schmid path tries its own grain's orientation
and those of its six face neighbours, and keeps the largest resolved shear.
This models the ambiguity of assigning a finite-element node to a grain at a
boundary. It is not a detail to tidy away: the generated datasets computed
their reference yield points the same way, so dropping it would move this
package's answers away from the reference by a few per cent with nothing to
show why.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from mvyield.ingest.artefacts import IngestResult, StepTable
from mvyield.ingest.material import MaterialProperties
from mvyield.ingest.reader import DATASET_CONVENTION, DatasetReader, Geometry, Step
from mvyield.ingest.slip import (
    euler_to_rotation,
    independent_systems,
    schmid_tensors,
    slip_systems,
)
from mvyield.mechanics import voigt
from mvyield.mechanics.voigt import MANDEL_XY_LAST, VoigtConvention
from mvyield.model.bundle import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    YieldPointSet,
)
from mvyield.settings import ConfigError

log = logging.getLogger(__name__)

NEIGHBOUR_OFFSETS = (
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)
"""The point's own voxel and its six face neighbours."""

ACTIVE_TOLERANCE = 0.995
"""A slip system counts as active within half a percent of the maximum."""

EPS_TAU = 1e-9
"""Below this the criterion value is treated as zero and the step dropped."""

POINT_CHUNK = 4096
"""Material points per matrix product, bounding the intermediate's size."""


@dataclass(frozen=True)
class StepResult:
    """The yield point of one load step, with its diagnostics."""

    geometry: str
    geometry_index: int
    step: int
    sigma6: np.ndarray
    k_factor: float
    tau_max: float
    von_mises: float
    principal_stress: np.ndarray
    principal_strain: np.ndarray
    strain6: np.ndarray
    active_systems: int


class EigenTracker:
    """Keep principal directions consistently ordered along a load path.

    ``eigh`` returns eigenvalues ascending and eigenvectors up to a sign, so
    between neighbouring steps a principal axis can appear to swap places
    with another or flip direction. Plotting the raw output produces
    discontinuities that look like physics and are not.

    Each step's eigenvectors are matched to the previous step's by largest
    overlap, and signs aligned. When the matching is not a permutation --
    two axes claiming the same predecessor, which happens where eigenvalues
    genuinely cross -- it falls back to ordering by magnitude, because a
    wrong pairing is worse than an honest reset.
    """

    def __init__(self) -> None:
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the history. Call between geometries."""
        self._previous = None

    def track(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return eigenvalues and eigenvectors, ordered against history."""
        values, vectors = np.linalg.eigh(tensor)

        if self._previous is None:
            order = np.argsort(values)[::-1]
            values, vectors = values[order], vectors[:, order]
        else:
            overlap = np.abs(self._previous.T @ vectors)
            order = np.argmax(overlap, axis=1)

            if len(np.unique(order)) < 3:
                order = np.argsort(values)[::-1]
                values, vectors = values[order], vectors[:, order]
            else:
                values = values[order]
                vectors = vectors[:, order]
                vectors = vectors * np.sign(np.diag(self._previous.T @ vectors))

        self._previous = vectors
        return values, vectors


# --- Per-step criteria -------------------------------------------------------


def schmid_step(
    step: Step,
    voxels: np.ndarray,
    rotations: np.ndarray,
    planes: np.ndarray,
    directions: np.ndarray,
    crss: float,
    tensors: np.ndarray | None = None,
    distinct: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, int]:
    """Scale a load step until the first slip system reaches the CRSS.

    Parameters
    ----------
    step : Step
        The load step to scale.
    voxels : ndarray
        Grain id per voxel.
    rotations : ndarray of shape (G, 3, 3)
        Grain orientations.
    planes, directions : ndarray
        Unit slip systems.
    crss : float
        Critical resolved shear stress, MPa.
    tensors : ndarray of shape (G, S, 6), optional
        Precomputed Schmid tensors from
        :func:`~mvyield.ingest.slip.schmid_tensors`. Built here when
        omitted; pass them in to build them once per geometry rather than
        once per step.
    distinct : ndarray, optional
        Indices of the independent slip systems, used for the active-system
        count only. See
        :func:`~mvyield.ingest.slip.independent_systems`.

    Returns
    -------
    sigma6 : ndarray of shape (6,)
        The yield point, MPa, in :data:`DATASET_CONVENTION`.
    k_factor : float
        Multiplier applied to the step.
    tau_max : float
        Largest resolved shear before scaling, MPa.
    active_systems : int
        Slip systems within :data:`ACTIVE_TOLERANCE` of the maximum at the
        winning point, after scaling.
    """
    if tensors is None:
        tensors = schmid_tensors(planes, directions, rotations)
    if distinct is None:
        distinct = independent_systems(planes, directions)

    sigma_mandel = voigt.convert(step.stress, DATASET_CONVENTION, MANDEL_XY_LAST)
    indices = _voxel_indices(step.coords, voxels.shape)
    n_points, n_grains = len(sigma_mandel), len(tensors)

    # Every point against every grain, once. The seven neighbourhood passes
    # then only look up a number rather than each redoing the arithmetic.
    by_grain = _resolved_by_grain(sigma_mandel, tensors)

    rows = np.arange(n_points)
    best_tau = np.zeros(n_points)
    best_grain = np.zeros(n_points, dtype=np.int64)

    for offset in NEIGHBOUR_OFFSETS:
        grains = _grain_at(voxels, indices, offset)
        usable = (grains >= 0) & (grains < n_grains)
        candidate = np.where(usable, by_grain[rows, np.clip(grains, 0, n_grains - 1)], 0.0)

        improved = candidate > best_tau
        best_grain = np.where(improved, grains, best_grain)
        best_tau = np.where(improved, candidate, best_tau)

    winner = int(np.argmax(best_tau))
    tau_max = float(best_tau[winner])
    if tau_max <= EPS_TAU:
        raise _NoYield("resolved shear is zero")

    k_factor = crss / tau_max
    sigma6 = step.stress[winner] * k_factor

    active = _count_active(sigma6, tensors[best_grain[winner]][distinct])
    return sigma6, k_factor, tau_max, active


def von_mises_step(step: Step, sigma_y: float) -> tuple[np.ndarray, float, float, int]:
    """Scale a load step until the equivalent stress reaches the strength."""
    equivalent = voigt.von_mises(step.stress, DATASET_CONVENTION)
    winner = int(np.argmax(equivalent))
    peak = float(equivalent[winner])

    if peak <= EPS_TAU:
        raise _NoYield("equivalent stress is zero")

    k_factor = sigma_y / peak
    return step.stress[winner] * k_factor, k_factor, peak, 0


# --- Step selection ----------------------------------------------------------


def _rule_max_macro_sz(results: list[StepResult], steps: list[Step]) -> int:
    """Step whose volume-averaged ZZ stress is largest in magnitude."""
    zz = DATASET_CONVENTION.slot_of("ZZ")
    return int(np.argmax([abs(s.stress_avg[zz]) for s in steps]))


def _rule_max_von_mises(results: list[StepResult], steps: list[Step]) -> int:
    """Step whose yield point has the largest equivalent stress."""
    return int(np.argmax([r.von_mises for r in results]))


def _rule_last(results: list[StepResult], steps: list[Step]) -> int:
    """The final step of the path."""
    return len(results) - 1


SELECTION_RULES: dict[str, Callable[[list[StepResult], list[Step]], int]] = {
    "max_macro_sz": _rule_max_macro_sz,
    "max_von_mises": _rule_max_von_mises,
    "last": _rule_last,
}
"""``dataset.select_step`` values, choosing one representative step per RVE.

The representative is used only by figures that show one state per geometry.
Every step contributes a point to the cloud regardless.
"""


def get_selection_rule(name: str) -> Callable[[list[StepResult], list[Step]], int]:
    """Look up a step-selection rule by config name."""
    try:
        return SELECTION_RULES[name]
    except KeyError:
        raise ConfigError(
            f"unknown dataset.select_step {name!r}; "
            f"available: {sorted(SELECTION_RULES)}"
        ) from None


# --- Whole-dataset analysis --------------------------------------------------


def analyse_dataset(
    reader: DatasetReader,
    material: MaterialProperties,
    analysis: str = "SCHMID",
    select_step: str = "max_macro_sz",
    convention: VoigtConvention = MANDEL_XY_LAST,
    geometries: Sequence[str] | None = None,
    seed: int = 0,
    progress: Callable[[str, int, int], None] | None = None,
) -> IngestResult:
    """Extract a yield point from every load step of every geometry.

    Parameters
    ----------
    reader : DatasetReader
        An open reader.
    material : MaterialProperties
        Supplies the CRSS or the scalar strength, depending on ``analysis``.
    analysis : str
        ``"SCHMID"`` or ``"VON_MISES"``.
    select_step : str
        Rule naming one representative step per geometry.
    convention : VoigtConvention
        Convention the cloud is written in. Defaults to the package's
        recommended one, which is also the default model space, so the two
        agree without a conversion nobody remembers to make.
    seed : int
        Reproducible geometry shuffling for the held-out split.
    geometries : sequence of str, optional
        Analyse only these RVEs. A full pass over a large dataset takes
        minutes, and looking at three geometries first is how one finds a
        misconfigured material before paying for all forty.
    progress : callable, optional
        Called as ``(geometry_id, done, total)`` after each geometry.

    Returns
    -------
    IngestResult
    """
    analysis = analysis.upper()
    rule = get_selection_rule(select_step)
    collected: list[StepResult] = []
    representative_rows: list[int] = []
    skipped: dict[str, int] = {}

    selected = _selected_geometries(reader, geometries)

    for position, geometry_id in enumerate(selected, start=1):
        geometry = reader.geometry(geometry_id)
        criterion = _make_criterion(analysis, geometry, material)

        tracker = EigenTracker()
        results: list[StepResult] = []
        steps: list[Step] = []

        for step in reader.steps(geometry_id):
            outcome = _analyse_step(step, geometry, criterion, tracker, skipped)
            if outcome is not None:
                results.append(outcome)
                steps.append(step)

        if results:
            representative_rows.append(len(collected) + rule(results, steps))
            collected.extend(results)

        if progress is not None:
            progress(geometry_id, position, len(selected))

    if not collected:
        raise ConfigError(
            f"no load step of {reader.path.name} produced a yield point"
            + (f" ({_reasons(skipped)})" if skipped else "")
        )

    return _assemble(collected, representative_rows, material, reader,
                     analysis, convention, skipped, seed)


# --- Internals ---------------------------------------------------------------


class _NoYield(RuntimeError):
    """A load step that carries no yield information."""


def _selected_geometries(
    reader: DatasetReader,
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve the geometries to analyse, in the reader's own order."""
    if requested is None:
        return reader.geometry_ids

    wanted = [str(name) for name in requested]
    missing = [name for name in wanted if name not in reader.geometry_ids]
    if missing:
        raise ConfigError(
            f"{reader.path.name} has no geometries {missing}; "
            f"available: {list(reader.geometry_ids)}"
        )
    return tuple(name for name in reader.geometry_ids if name in set(wanted))


def _make_criterion(
    analysis: str,
    geometry: Geometry,
    material: MaterialProperties,
) -> Callable[[Step], tuple[np.ndarray, float, float, int]]:
    """Bind the per-geometry inputs a criterion needs, once per geometry."""
    if analysis == "VON_MISES":
        strength = material.require_sigma_y()
        return lambda step: von_mises_step(step, strength)

    if analysis != "SCHMID":
        raise ConfigError(
            f"dataset.analysis must be SCHMID or VON_MISES; got {analysis!r}"
        )

    crss = material.require_crss()
    if geometry.voxels is None or geometry.euler_deg is None:
        raise ConfigError(
            f"the Schmid analysis needs grain orientations, and geometry "
            f"{geometry.id!r} has "
            f"{'no voxels' if geometry.voxels is None else 'no local_cs'}. "
            f"Use dataset.analysis: VON_MISES for a dataset without "
            f"microstructure."
        )

    planes, directions = slip_systems(material.structure)
    rotations = euler_to_rotation(geometry.euler_deg)
    voxels = geometry.voxels

    # Built once per geometry: the orientations do not change along the
    # load path, so rebuilding them per step was pure repetition.
    tensors = schmid_tensors(planes, directions, rotations)
    distinct = independent_systems(planes, directions)

    return lambda step: schmid_step(
        step, voxels, rotations, planes, directions, crss, tensors, distinct
    )


def _analyse_step(
    step: Step,
    geometry: Geometry,
    criterion: Callable[[Step], tuple[np.ndarray, float, float, int]],
    tracker: EigenTracker,
    skipped: dict[str, int],
) -> StepResult | None:
    """Analyse one step, or record why it produced nothing.

    A step is dropped rather than contributed as a zero point. The legacy
    code set ``k = 0`` and carried on, which put a stress state of all
    zeros into the cloud -- a point at the origin, pulling every density
    estimate towards it.
    """
    if not np.isfinite(step.stress).all():
        skipped["non-finite stress"] = skipped.get("non-finite stress", 0) + 1
        return None

    try:
        sigma6, k_factor, tau_max, active = criterion(step)
    except _NoYield as exc:
        reason = str(exc)
        skipped[reason] = skipped.get(reason, 0) + 1
        return None

    principal_stress, _ = tracker.track(_to_tensors(sigma6[None, :])[0])
    strain6 = np.asarray(step.eps_as_loading, dtype=np.float64).ravel() * k_factor
    principal_strain = _principal_strain(step.eps_as_loading, k_factor)

    return StepResult(
        geometry=geometry.id,
        geometry_index=geometry.index,
        step=step.index,
        sigma6=sigma6,
        k_factor=k_factor,
        tau_max=tau_max,
        von_mises=float(np.ravel(voigt.von_mises(sigma6, DATASET_CONVENTION))[0]),
        principal_stress=principal_stress,
        principal_strain=principal_strain,
        strain6=strain6,
        active_systems=active,
    )


def _assemble(
    results: list[StepResult],
    representative_rows: list[int],
    material: MaterialProperties,
    reader: DatasetReader,
    analysis: str,
    convention: VoigtConvention,
    skipped: dict[str, int],
    seed: int = 0,
) -> IngestResult:
    """Turn per-step results into the two artefacts."""
    raw = np.array([r.sigma6 for r in results], dtype=np.float64)
    representative = np.zeros(len(results), dtype=bool)
    representative[representative_rows] = True

    points = YieldPointSet(
        sigma6=voigt.convert(raw, DATASET_CONVENTION, convention),
        group_id=np.array([r.geometry_index for r in results], dtype=np.int32),
        meta={
            "convention": convention.name,
            "source": str(reader.path),
            "analysis": analysis,
            "material": material.to_dict(),
            "skipped_steps": dict(skipped),
        },
    )
    points = _with_split(points, seed)

    table = StepTable(
        geometry_id=np.array([r.geometry for r in results], dtype=np.str_),
        geometry_index=np.array([r.geometry_index for r in results], dtype=np.int32),
        step=np.array([r.step for r in results], dtype=np.int32),
        k_factor=np.array([r.k_factor for r in results], dtype=np.float64),
        tau_max=np.array([r.tau_max for r in results], dtype=np.float64),
        von_mises=np.array([r.von_mises for r in results], dtype=np.float64),
        principal_stress=np.array([r.principal_stress for r in results], dtype=np.float64),
        principal_strain=np.array([r.principal_strain for r in results], dtype=np.float64),
        strain6=np.array([r.strain6 for r in results], dtype=np.float64),
        active_systems=np.array([r.active_systems for r in results], dtype=np.int32),
        representative=representative,
    )

    return IngestResult(points=points, steps=table, material=material, skipped=skipped)


def _with_split(points: YieldPointSet, seed: int) -> YieldPointSet:
    """Assign the held-out split the artefact carries from here on.

    Calibration needs held-out geometries and cannot invent them later: a
    split made at fit time would differ between methods and between runs,
    so the comparison between methods would include a difference in what
    they were scored on. Made once, here, it travels with the cloud.

    Fewer than three geometries cannot be split three ways. That is a
    legitimate dataset -- ``--geometries 0`` produces one -- so it is a
    warning rather than a failure, and calibration is what stops later.
    """
    if points.n_groups < 3:
        log.warning(
            "%d geometr%s cannot be split into train, validation and test, so "
            "the cloud carries no split. Automatic hyper-parameters need one; "
            "set them manually, or analyse at least three RVE.",
            points.n_groups,
            "y" if points.n_groups == 1 else "ies",
        )
        return points

    split = points.with_split(seed=seed)
    log.info(
        "split by geometry: %d train, %d validation, %d test points",
        int((split.split == SPLIT_TRAIN).sum()),
        int((split.split == SPLIT_VAL).sum()),
        int((split.split == SPLIT_TEST).sum()),
    )
    return split


def _to_tensors(sigma6: np.ndarray) -> np.ndarray:
    """Stress vectors in :data:`DATASET_CONVENTION` to 3x3 tensors."""
    sigma6 = np.atleast_2d(np.asarray(sigma6, dtype=np.float64))
    slot = DATASET_CONVENTION.slot_of

    tensors = np.zeros((len(sigma6), 3, 3))
    tensors[:, 0, 0] = sigma6[:, slot("XX")]
    tensors[:, 1, 1] = sigma6[:, slot("YY")]
    tensors[:, 2, 2] = sigma6[:, slot("ZZ")]
    tensors[:, 0, 1] = tensors[:, 1, 0] = sigma6[:, slot("XY")]
    tensors[:, 1, 2] = tensors[:, 2, 1] = sigma6[:, slot("YZ")]
    tensors[:, 0, 2] = tensors[:, 2, 0] = sigma6[:, slot("XZ")]
    return tensors


def _resolved_by_grain(
    sigma_mandel: np.ndarray,
    tensors: np.ndarray,
    chunk: int = POINT_CHUNK,
) -> np.ndarray:
    """Largest resolved shear for every point-grain pair.

    One matrix product covers all points, all grains and all slip systems,
    which is what the Mandel form buys: a double contraction becomes a dot
    product, and BLAS does the rest. Chunking over points bounds the
    intermediate at a few tens of megabytes, so a larger RVE costs more time
    rather than more memory.

    Returns
    -------
    ndarray of shape (N, G)
    """
    n_grains, n_systems, _ = tensors.shape
    flat = tensors.reshape(n_grains * n_systems, 6)

    resolved = np.empty((len(sigma_mandel), n_grains))
    for start in range(0, len(sigma_mandel), chunk):
        block = np.abs(sigma_mandel[start : start + chunk] @ flat.T)
        resolved[start : start + chunk] = block.reshape(-1, n_grains, n_systems).max(axis=2)
    return resolved


def _voxel_indices(coords: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Integer voxel indices, clipped into the grid."""
    indices = np.rint(coords[:, :3]).astype(np.int64)
    for axis in range(3):
        np.clip(indices[:, axis], 0, shape[axis] - 1, out=indices[:, axis])
    return indices


def _grain_at(voxels: np.ndarray, indices: np.ndarray, offset: tuple[int, int, int]) -> np.ndarray:
    """Grain id of the voxel at ``indices + offset``, clipped at the border."""
    shifted = np.empty_like(indices)
    for axis, delta in enumerate(offset):
        np.clip(indices[:, axis] + delta, 0, voxels.shape[axis] - 1, out=shifted[:, axis])
    return voxels[shifted[:, 0], shifted[:, 1], shifted[:, 2]].astype(np.int64)


def _count_active(sigma6: np.ndarray, tensors: np.ndarray) -> int:
    """Slip systems carrying nearly the maximum resolved shear.

    More than one means the yield point sits on an edge of the surface,
    where two systems activate together -- the states the surface is least
    smooth at, and the ones a density estimate has most trouble with.

    The legacy code computed this with the opposite rotation to the one it
    used for the yield point itself, so it counted the active systems of a
    differently oriented tensor. Here both go through the same Schmid
    tensors.
    """
    sigma_mandel = voigt.convert(sigma6[None, :], DATASET_CONVENTION, MANDEL_XY_LAST)
    taus = np.abs(sigma_mandel[0] @ tensors.T)

    peak = float(taus.max())
    if peak <= EPS_TAU:
        return 0
    return int(np.count_nonzero(taus >= peak * ACTIVE_TOLERANCE))


def _principal_strain(eps_as_loading: np.ndarray, k_factor: float) -> np.ndarray:
    """Principal values of the applied strain, scaled to the yield point.

    ``eps_as_loading`` holds engineering shear, so the tensor entries are
    half of the stored values.
    """
    eps = np.asarray(eps_as_loading, dtype=np.float64).ravel()
    tensor = np.zeros((3, 3))
    tensor[0, 0], tensor[1, 1], tensor[2, 2] = eps[0], eps[1], eps[2]
    tensor[0, 1] = tensor[1, 0] = 0.5 * eps[3]
    tensor[1, 2] = tensor[2, 1] = 0.5 * eps[4]
    tensor[0, 2] = tensor[2, 0] = 0.5 * eps[5]
    return np.sort(np.linalg.eigvalsh(tensor * k_factor))[::-1]


def _reasons(skipped: dict[str, int]) -> str:
    """Render skip counts for an error message."""
    return ", ".join(f"{count} with {reason}" for reason, count in sorted(skipped.items()))
