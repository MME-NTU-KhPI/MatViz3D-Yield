"""Mode A as a step: dataset in, yield-point artefact out.

Analysing forty RVE takes minutes, and the answer depends only on the file
and on three settings. So the result is cached under a key built from both,
and a second run of the same case reads it back instead of recomputing.

The key deliberately excludes the file's modification time. A dataset that
was copied, restored from a backup or re-downloaded byte-for-byte is the
same dataset, and invalidating the cache on a timestamp would spend those
minutes proving it.
"""

from __future__ import annotations

import json
import logging
from hashlib import sha256
from pathlib import Path

from mvyield.model.bundle import YieldPointSet
from mvyield.paths import RunPaths, resolve_input
from mvyield.settings import CaseConfig, ConfigError, DatasetConfig

log = logging.getLogger(__name__)


def dataset_key(source: Path, config: DatasetConfig) -> str:
    """Stable identifier for one dataset analysed one way.

    Built from the resolved path, the file's size and the settings that
    change the answer. Two different datasets cannot collide; the same
    dataset analysed with a different criterion does not reuse the first
    answer.
    """
    digest = sha256()
    digest.update(str(Path(source).resolve()).encode())
    digest.update(str(Path(source).stat().st_size).encode())
    digest.update(
        json.dumps(
            {
                "analysis": config.analysis,
                "select_step": config.select_step,
                "material_override": config.material_override,
            },
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()[:16]


def resolve_dataset(cfg: CaseConfig, paths: RunPaths) -> Path:
    """Locate the dataset a case names, or explain that it is missing."""
    if cfg.dataset.source is None:
        raise ConfigError(
            "this step needs dataset.source, but the case config does not set "
            "it. Add the dataset path, or use 'mvy run' to evaluate an "
            "existing model without opening a dataset."
        )

    resolved = resolve_input(cfg.dataset.source, paths.root)
    if not resolved.exists():
        raise FileNotFoundError(
            f"dataset not found: {resolved}\n"
            f"The case names {cfg.dataset.source}. Put the file there, point "
            f"MVY_DATA_DIR at the directory holding it, or override the path "
            f"with --set dataset.source=<path>."
        )
    return resolved


def run_ingest(
    cfg: CaseConfig,
    paths: RunPaths,
    force: bool = False,
    geometries: list[str] | None = None,
):
    """Analyse the dataset, or load an equivalent earlier analysis.

    Parameters
    ----------
    cfg : CaseConfig
        The resolved case.
    paths : RunPaths
        Run layout; artefacts land in its shared model store.
    force : bool
        Re-analyse even when a cached artefact matches.
    geometries : list of str, optional
        Analyse only these RVE. Never cached, because a partial answer must
        not be mistaken for the whole one.

    Returns
    -------
    IngestResult
    """
    # Imported here: this is the only path that needs h5py, and 'mvy run'
    # must not pay for it.
    from mvyield.ingest import DatasetReader, analyse_dataset, read_material
    from mvyield.ingest.artefacts import IngestResult, StepTable
    from mvyield.ingest.material import MaterialProperties

    source = resolve_dataset(cfg, paths)
    key = dataset_key(source, cfg.dataset)
    points_path = paths.models_dir / f"{key}.yieldpoints.npz"
    steps_path = paths.models_dir / f"{key}.steps.npz"

    if not force and geometries is None and points_path.exists() and steps_path.exists():
        points = YieldPointSet.load(points_path)
        log.info(
            "reusing dataset analysis: %d points from %d geometries (%s)",
            points.n_points,
            points.n_groups,
            points_path.name,
        )
        return IngestResult(
            points=points,
            steps=StepTable.load(steps_path),
            material=MaterialProperties.from_dict(points.meta.get("material", {})),
            skipped=dict(points.meta.get("skipped_steps", {})),
        )

    log.info("analysing %s (%s)", source.name, cfg.dataset.analysis)

    with DatasetReader(source) as reader:
        material = read_material(reader, cfg.dataset.material_override)
        log.info("material:\n%s", material.describe())

        result = analyse_dataset(
            reader,
            material,
            analysis=cfg.dataset.analysis,
            select_step=cfg.dataset.select_step,
            convention=cfg.model.voigt_convention,
            geometries=geometries,
            seed=cfg.seed,
            progress=_progress,
        )

    if geometries is None:
        written = result.save(paths.models_dir, stem=key)
        log.info("wrote %s", written["points"].name)
    else:
        log.warning("partial analysis of %d geometries; not cached", len(geometries))

    return result


def _progress(geometry: str, done: int, total: int) -> None:
    """Log progress at a rate that stays readable for forty geometries."""
    if done == total or done % 5 == 0 or total <= 10:
        log.info("  geometry %s (%d/%d)", geometry, done, total)
