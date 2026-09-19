"""Model artefacts: point cloud plus per-method parameters, on disk.

Replaces the pickle bundle of the pre-refactor code. Pickle ties a file to
the exact class layout that wrote it, so renaming a class or moving a module
breaks every stored model, and the format is neither inspectable nor safe to
load from an untrusted source. Here arrays go into a ``.npz`` and metadata
into a sibling ``.json`` that a human can read and diff.

Two artefact types share this module:

``*.yieldpoints.npz``
    Output of dataset analysis: the yield-point cloud plus the train /
    validation / test split. This is the contract between the ingest stage
    and everything downstream, and it is also the training set for a neural
    surrogate later.

``*.model.npz``
    A fitted model: the same cloud, the :class:`SpaceTransform` it lives in,
    and resolved hyper-parameters with their provenance.

The cloud is stored **once** per model, with methods as parameter blocks on
top of it, because ray marginalisation and the conditional slice share the
data and the space but nothing else. In the old code the slice was built
*from* the ray model, which made it impossible to run one without the other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .transform import SpaceTransform

SCHEMA_VERSION = 2
"""Bump when a stored field changes meaning. Readers refuse newer versions."""

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2


@dataclass
class YieldPointSet:
    """A cloud of yield points with provenance and a reusable split.

    Parameters
    ----------
    sigma6 : ndarray of shape (N, 6)
        Raw yield-surface points, MPa, in ``convention``.
    group_id : ndarray of shape (N,)
        Source RVE or geometry index for each point.
    weight : ndarray of shape (N,), optional
        Per-point weights; ones when omitted.
    split : ndarray of shape (N,), optional
        Train / validation / test assignment. Built by :meth:`with_split`.
    meta : dict
        Material and provenance metadata, including the convention.
    """

    sigma6: np.ndarray
    group_id: np.ndarray
    weight: np.ndarray | None = None
    split: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shapes and fill defaults."""
        self.sigma6 = np.asarray(self.sigma6, dtype=np.float64)
        if self.sigma6.ndim != 2 or self.sigma6.shape[1] != 6:
            raise ValueError(f"sigma6 must have shape (N, 6), got {self.sigma6.shape}")

        self.group_id = np.asarray(self.group_id, dtype=np.int32).ravel()
        if len(self.group_id) != len(self.sigma6):
            raise ValueError(
                f"group_id has {len(self.group_id)} entries for {len(self.sigma6)} points"
            )
        if self.weight is None:
            self.weight = np.ones(len(self.sigma6), dtype=np.float64)
        else:
            self.weight = np.asarray(self.weight, dtype=np.float64).ravel()
            if len(self.weight) != len(self.sigma6):
                raise ValueError("weight length does not match the number of points")
        if self.split is not None:
            self.split = np.asarray(self.split, dtype=np.uint8).ravel()
            if len(self.split) != len(self.sigma6):
                raise ValueError("split length does not match the number of points")

    @property
    def n_points(self) -> int:
        """Number of yield points."""
        return len(self.sigma6)

    @property
    def n_groups(self) -> int:
        """Number of distinct source geometries."""
        return len(np.unique(self.group_id))

    def with_split(
        self,
        val_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 0,
    ) -> YieldPointSet:
        """Assign a train / validation / test split **by group**, not by point.

        Points from one RVE are strongly correlated: neighbouring voxels see
        almost the same stress state. Splitting at random would place near
        duplicates on both sides, producing a flattering validation score and
        a model that does not transfer to an unseen RVE. Splitting whole
        geometries is the only honest option here, and doing it once at
        ingest time means every method is later scored on the same held-out
        data.

        Parameters
        ----------
        val_fraction, test_fraction : float
            Share of *groups*, not points, held out.
        seed : int
            Reproducible group shuffling.

        Returns
        -------
        YieldPointSet
            New instance; the original is unchanged.
        """
        if not 0.0 <= val_fraction + test_fraction < 1.0:
            raise ValueError("validation and test fractions must sum to less than 1")

        groups = np.unique(self.group_id)
        if len(groups) < 3:
            raise ValueError(
                f"cannot split {len(groups)} group(s) three ways; a grouped split "
                f"needs at least 3 geometries. Analyse more RVE, or evaluate "
                f"without a held-out set."
            )

        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(groups)
        n_val = max(1, round(val_fraction * len(groups)))
        n_test = max(1, round(test_fraction * len(groups)))

        assignment = {g: SPLIT_TRAIN for g in shuffled}
        for g in shuffled[:n_val]:
            assignment[g] = SPLIT_VAL
        for g in shuffled[n_val : n_val + n_test]:
            assignment[g] = SPLIT_TEST

        split = np.array([assignment[g] for g in self.group_id], dtype=np.uint8)
        return YieldPointSet(
            sigma6=self.sigma6,
            group_id=self.group_id,
            weight=self.weight,
            split=split,
            meta={**self.meta, "split_seed": seed, "split_by": "group_id"},
        )

    def subset(self, which: int) -> YieldPointSet:
        """Return the points belonging to one split partition."""
        if self.split is None:
            raise ValueError("no split assigned; call with_split() first")
        mask = self.split == which
        return YieldPointSet(
            sigma6=self.sigma6[mask],
            group_id=self.group_id[mask],
            weight=self.weight[mask] if self.weight is not None else None,
            meta=self.meta,
        )

    def content_hash(self) -> str:
        """Stable digest of the point cloud, used for caching.

        Depends only on the numbers, so an unchanged dataset never triggers a
        rebuild, and any change to the points does.
        """
        digest = sha256()
        digest.update(np.ascontiguousarray(self.sigma6).tobytes())
        digest.update(np.ascontiguousarray(self.group_id).tobytes())
        return digest.hexdigest()[:16]

    def save(self, path: Path) -> Path:
        """Write the artefact to ``path`` plus a sibling ``.json``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {
            "sigma6": self.sigma6,
            "group_id": self.group_id,
            "weight": self.weight,
        }
        if self.split is not None:
            arrays["split"] = self.split
        np.savez_compressed(path, **arrays)

        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "kind": "yieldpoints",
            "n_points": self.n_points,
            "n_groups": self.n_groups,
            "content_hash": self.content_hash(),
            "created_at": _timestamp(),
            "meta": _json_safe(self.meta),
        }
        _write_json(path.with_suffix(".json"), sidecar)
        return path

    @classmethod
    def load(cls, path: Path) -> YieldPointSet:
        """Read an artefact written by :meth:`save`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"yield-point artefact not found: {path}\n"
                f"Run 'mvy ingest --case <case.yaml>' to produce it."
            )
        sidecar = _read_json(path.with_suffix(".json"))
        _check_schema(sidecar, path)

        with np.load(path) as data:
            return cls(
                sigma6=data["sigma6"],
                group_id=data["group_id"],
                weight=data["weight"] if "weight" in data else None,
                split=data["split"] if "split" in data else None,
                meta=sidecar.get("meta", {}),
            )


@dataclass
class ModelBundle:
    """A fitted yield model: cloud, space and per-method parameters.

    Parameters
    ----------
    points : YieldPointSet
        The cloud every estimator works on.
    transform : SpaceTransform
        Mapping applied to the cloud and to every query. Stored here so the
        two can never be prepared differently.
    methods : tuple of str
        Methods this bundle carries parameters for.
    primary : str
        Method used for headline decisions and reported first.
    hyperparams : dict
        Resolved values with provenance, keyed by parameter name. Each entry
        holds at least ``value`` and ``source``; see
        :meth:`record_hyperparam`.
    params : dict
        Free-form per-method parameters, keyed by method name.
    diagnostics : dict
        Summary statistics of the fitted cloud.
    """

    points: YieldPointSet
    transform: SpaceTransform
    methods: tuple[str, ...]
    primary: str
    hyperparams: dict[str, dict] = field(default_factory=dict)
    params: dict[str, dict] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the method list and the primary choice."""
        self.methods = tuple(self.methods)
        if not self.methods:
            raise ValueError("a model bundle must carry at least one method")
        if self.primary not in self.methods:
            raise ValueError(
                f"primary method {self.primary!r} is not among {list(self.methods)}"
            )

    def record_hyperparam(
        self,
        name: str,
        value: float,
        source: str,
        objective: float | None = None,
        grid: list[float] | None = None,
        curve: dict[float, float] | None = None,
    ) -> None:
        """Store a resolved hyper-parameter together with where it came from.

        Provenance is not decoration. A reader of a figure needs to know
        whether ``power = 8`` was cross-validated or picked by eye from a
        sweep table, and that distinction is lost the moment the value is
        copied into a config file by hand.

        Parameters
        ----------
        name : str
            Parameter name, e.g. ``"power"`` or ``"kappa"``.
        value : float
            Resolved value.
        source : str
            ``"manual"``, ``"cv_loglik"``, ``"locus_mae"``, ``"frozen"``.
        objective : float, optional
            Score at the selected value, when a criterion was optimised.
        grid : list of float, optional
            Candidates that were searched.
        curve : dict of float to float, optional
            Objective at every candidate. Stored as two parallel lists
            rather than as a mapping: JSON keys are strings, so a float-keyed
            dict does not survive a round trip through the sidecar, and the
            diagnostic plot would read back ``"20.0"`` where it wants ``20``.
        """
        self.hyperparams[name] = {
            "value": float(value),
            "source": source,
            "objective": objective,
            "grid_searched": list(grid) if grid else None,
            "calibrated_at": _timestamp() if source not in ("manual", "frozen") else None,
        }
        if curve:
            ordered = sorted(curve.items())
            self.hyperparams[name]["curve"] = {
                "grid": [float(x) for x, _ in ordered],
                "objective": [float(y) for _, y in ordered],
            }

    def hyperparam(self, name: str) -> float:
        """Look up a resolved hyper-parameter value.

        Raises
        ------
        KeyError
            With a message naming the calibration command, since a missing
            hyper-parameter almost always means a step was skipped.
        """
        try:
            return float(self.hyperparams[name]["value"])
        except KeyError:
            raise KeyError(
                f"hyper-parameter {name!r} is not recorded in this bundle; "
                f"run 'mvy calibrate' or set it manually in the case config"
            ) from None

    def inputs_hash(self, config_fingerprint: str) -> str:
        """Digest of everything the fit depends on, for build caching.

        Combines the point cloud with a fingerprint of the model section of
        the config, so a rebuild is triggered by a changed dataset *or* by a
        changed setting, and by nothing else. File timestamps are deliberately
        not involved: copying a dataset should not invalidate a model.
        """
        digest = sha256()
        digest.update(self.points.content_hash().encode())
        digest.update(config_fingerprint.encode())
        return digest.hexdigest()[:16]

    def save(self, path: Path, inputs_hash: str | None = None) -> Path:
        """Write the bundle to ``path`` plus a readable ``.json`` sidecar."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {
            "sigma6": self.points.sigma6,
            "group_id": self.points.group_id,
            "weight": self.points.weight,
        }
        if self.points.split is not None:
            arrays["split"] = self.points.split
        for method, block in self.params.items():
            for key, value in block.items():
                if isinstance(value, np.ndarray):
                    arrays[f"params/{method}/{key}"] = value
        np.savez_compressed(path, **arrays)

        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "kind": "model",
            "methods": list(self.methods),
            "primary": self.primary,
            "space": self.transform.to_dict(),
            "hyperparams": self.hyperparams,
            "params": {
                method: {
                    key: _json_safe(value)
                    for key, value in block.items()
                    if not isinstance(value, np.ndarray)
                }
                for method, block in self.params.items()
            },
            "diagnostics": _json_safe(self.diagnostics),
            "points": {
                "n_points": self.points.n_points,
                "n_groups": self.points.n_groups,
                "content_hash": self.points.content_hash(),
                "meta": _json_safe(self.points.meta),
            },
            "inputs_hash": inputs_hash,
            "created_at": _timestamp(),
        }
        _write_json(path.with_suffix(".json"), sidecar)
        return path

    @classmethod
    def load(cls, path: Path) -> ModelBundle:
        """Read a bundle written by :meth:`save`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"model bundle not found: {path}\n"
                f"Run 'mvy build-model --case <case.yaml>' to build it, or point "
                f"model.path at an existing bundle."
            )
        sidecar = _read_json(path.with_suffix(".json"))
        _check_schema(sidecar, path)

        with np.load(path) as data:
            points = YieldPointSet(
                sigma6=data["sigma6"],
                group_id=data["group_id"],
                weight=data["weight"] if "weight" in data else None,
                split=data["split"] if "split" in data else None,
                meta=sidecar.get("points", {}).get("meta", {}),
            )
            params: dict[str, dict] = {
                method: dict(block) for method, block in sidecar.get("params", {}).items()
            }
            for key in data.files:
                if key.startswith("params/"):
                    _, method, name = key.split("/", 2)
                    params.setdefault(method, {})[name] = data[key]

        return cls(
            points=points,
            transform=SpaceTransform.from_dict(sidecar["space"]),
            methods=tuple(sidecar["methods"]),
            primary=sidecar["primary"],
            hyperparams=sidecar.get("hyperparams", {}),
            params=params,
            diagnostics=sidecar.get("diagnostics", {}),
        )

    @staticmethod
    def peek(path: Path) -> dict:
        """Read only the sidecar, without loading arrays.

        Used by cache checks and by ``mvy doctor``, where opening a
        multi-megabyte archive to read one hash would be wasteful.
        """
        return _read_json(Path(path).with_suffix(".json"))

    def describe(self) -> str:
        """Multi-line summary for logs and reports."""
        lines = [
            f"points   : {self.points.n_points} from {self.points.n_groups} geometries",
            f"space    : {self.transform.describe()}",
            f"methods  : {', '.join(self.methods)} (primary: {self.primary})",
        ]
        for name, record in self.hyperparams.items():
            source = record.get("source", "?")
            objective = record.get("objective")
            tail = f", objective {objective:.4g}" if objective is not None else ""
            lines.append(f"  {name:<8s}: {record['value']:g}  [{source}{tail}]")
        return "\n".join(lines)


# --- Helpers -----------------------------------------------------------------


def _timestamp() -> str:
    """Return the current UTC time in ISO-8601, to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _check_schema(sidecar: dict, path: Path) -> None:
    """Refuse artefacts written by a newer version of the package."""
    version = int(sidecar.get("schema_version", 0))
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"{path.name} uses schema version {version}, but this build "
            f"understands up to {SCHEMA_VERSION}. Upgrade mvyield."
        )


def _write_json(path: Path, payload: dict) -> None:
    """Write a sidecar with stable formatting so diffs stay readable."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def _read_json(path: Path) -> dict:
    """Read a sidecar, with a clear message when it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"metadata sidecar not found: {path}\n"
            f"An artefact is the .npz and the .json together; the .json is "
            f"missing, so this file cannot be interpreted."
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars and arrays into JSON-serialisable values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
