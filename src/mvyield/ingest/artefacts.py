"""What dataset analysis produces.

Two artefacts, deliberately separate.

:class:`~mvyield.model.bundle.YieldPointSet` is the cloud every estimator is
fitted against: stresses, their source geometry, nothing else. Its content
hash decides whether a model needs rebuilding, so anything that does not
change the points must not be in it -- otherwise adding a diagnostic column
invalidates every cached model.

:class:`StepTable` is one row per analysed load step, in the same order as
the cloud, holding what the dataset figures need and the model does not:
principal values, the Schmid multiplier, how many slip systems were active.
Keeping it out of the model artefact is what lets ``mvy plot`` redraw the
dataset figures without loading a model, and lets the model be small.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mvyield.ingest.material import MaterialProperties
from mvyield.model.bundle import YieldPointSet

SCHEMA_VERSION = 2
"""Bump when a stored column changes meaning, or when one is added."""


@dataclass
class StepTable:
    """Per-step results of dataset analysis, one row per yield point.

    Parameters
    ----------
    geometry_id : ndarray of shape (N,)
        Source geometry name for each row, as text.
    geometry_index : ndarray of shape (N,)
        Position of that geometry in the reader's ordering.
    step : ndarray of shape (N,)
        Load step number within the geometry.
    k_factor : ndarray of shape (N,)
        Multiplier that brought the step to the yield criterion.
    tau_max : ndarray of shape (N,)
        Criterion value before scaling: resolved shear for Schmid, von Mises
        equivalent stress for the other mode. MPa.
    von_mises : ndarray of shape (N,)
        Equivalent stress of the yield point itself, MPa. What
        ``datasets_synth/info.txt`` reports.
    principal_stress : ndarray of shape (N, 3)
        Principal stresses of the yield point, ordered for continuity along
        the load path rather than by magnitude. MPa.
    principal_strain : ndarray of shape (N, 3)
        Principal strains of the applied loading, scaled by ``k_factor``.
    strain6 : ndarray of shape (N, 6)
        Applied strain at yield, engineering shear, in the dataset's own
        component order. Records which loading directions the dataset
        actually sampled, which is what a gap in the yield surface has to be
        read against.
    active_systems : ndarray of shape (N,)
        Slip systems within half a percent of the most stressed one. A count
        above one means the yield point sits on an edge of the surface.
    representative : ndarray of shape (N,), bool
        Whether this row is its geometry's representative step, chosen by
        ``dataset.select_step``. The dataset figures that show one state per
        RVE use these.
    """

    geometry_id: np.ndarray
    geometry_index: np.ndarray
    step: np.ndarray
    k_factor: np.ndarray
    tau_max: np.ndarray
    von_mises: np.ndarray
    principal_stress: np.ndarray
    principal_strain: np.ndarray
    strain6: np.ndarray
    active_systems: np.ndarray
    representative: np.ndarray

    COLUMNS = (
        "geometry_id",
        "geometry_index",
        "step",
        "k_factor",
        "tau_max",
        "von_mises",
        "principal_stress",
        "principal_strain",
        "strain6",
        "active_systems",
        "representative",
    )

    def __post_init__(self) -> None:
        """Check that every column agrees on the number of rows."""
        sizes = {len(getattr(self, name)) for name in self.COLUMNS}
        if len(sizes) > 1:
            raise ValueError(f"step table columns disagree on length: {sizes}")

    def __len__(self) -> int:
        """Number of rows."""
        return len(self.step)

    @property
    def n_rows(self) -> int:
        """Number of rows."""
        return len(self)

    def representatives(self) -> StepTable:
        """The one row per geometry chosen by the step-selection rule."""
        return self.select(np.asarray(self.representative, dtype=bool))

    def select(self, mask: np.ndarray) -> StepTable:
        """Return the rows a boolean mask keeps."""
        mask = np.asarray(mask, dtype=bool)
        return StepTable(**{name: getattr(self, name)[mask] for name in self.COLUMNS})

    def save(self, path: Path) -> Path:
        """Write the table plus a sidecar describing it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(path, **{name: getattr(self, name) for name in self.COLUMNS})

        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "kind": "steps",
            "n_rows": self.n_rows,
            "columns": list(self.COLUMNS),
        }
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> StepTable:
        """Read a table written by :meth:`save`.

        A table written by an older version is refused with instructions
        rather than half-read: a missing column would otherwise surface as a
        ``KeyError`` from inside a plotting function.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"step table not found: {path}\n"
                f"Run 'mvy ingest --case <case.yaml>' to produce it."
            )

        sidecar_path = path.with_suffix(".json")
        version = 0
        if sidecar_path.exists():
            version = json.loads(sidecar_path.read_text(encoding="utf-8")).get(
                "schema_version", 0
            )
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{path.name} was written by schema version {version}, and this "
                f"package reads version {SCHEMA_VERSION}. Re-run "
                f"'mvy ingest --case <case.yaml> --force' to rebuild it."
            )

        with np.load(path, allow_pickle=False) as data:
            return cls(**{name: data[name] for name in cls.COLUMNS})


@dataclass
class IngestResult:
    """Everything one pass over a dataset produced.

    Parameters
    ----------
    points : YieldPointSet
        The cloud, in the model convention.
    steps : StepTable
        Per-step results, row-aligned with the cloud.
    material : MaterialProperties
        What the dataset says the material is.
    skipped : dict
        Reason to how many load steps it cost. Empty when everything was
        usable. Reported rather than silently absorbed: a dataset where a
        third of the steps dropped out is a different dataset.
    """

    points: YieldPointSet
    steps: StepTable
    material: MaterialProperties
    skipped: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check that the cloud and the table describe the same rows."""
        if self.points.n_points != self.steps.n_rows:
            raise ValueError(
                f"the cloud has {self.points.n_points} points but the step "
                f"table has {self.steps.n_rows} rows; they must be aligned"
            )

    @property
    def n_skipped(self) -> int:
        """Total load steps that produced no yield point."""
        return sum(self.skipped.values())

    def save(self, directory: Path, stem: str | None = None) -> dict[str, Path]:
        """Write both artefacts, named by the cloud's content hash.

        Naming by content means an unchanged dataset analysed twice lands on
        the same files, and a changed one cannot overwrite the old answer.
        """
        directory = Path(directory)
        stem = stem or self.points.content_hash()

        return {
            "points": self.points.save(directory / f"{stem}.yieldpoints.npz"),
            "steps": self.steps.save(directory / f"{stem}.steps.npz"),
        }

    def describe(self) -> str:
        """Summary for the console and the run report."""
        equivalent = self.steps.von_mises
        mean = float(np.mean(equivalent))
        std = float(np.std(equivalent))

        lines = [
            f"yield points : {self.points.n_points} from {self.points.n_groups} geometries",
            f"sigma_vm     : {mean:.2f} +- {std:.2f} MPa"
            f"  (CV {100.0 * std / mean:.2f} %)" if mean > 0 else "sigma_vm     : undefined",
            f"k factor     : {self.steps.k_factor.min():.3g} .. "
            f"{self.steps.k_factor.max():.3g}",
        ]
        if self.skipped:
            for reason, count in sorted(self.skipped.items()):
                lines.append(f"skipped      : {count} steps ({reason})")
        return "\n".join(lines)
