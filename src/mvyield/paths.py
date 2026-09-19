"""Filesystem layout: repository root, artefact store, per-run directories.

Two rules this module exists to enforce:

* **No absolute paths in configs.** Everything resolves against a repository
  root found by walking up from the config file, or against ``MVY_DATA_DIR``.
  A config that works on one machine works on another.
* **No side effects on import.** Directories are created by
  :meth:`RunPaths.prepare`, never as a side effect of reading a module. The
  old ``config.py`` called ``mkdir`` at import time, which meant importing it
  to inspect a value created folders.

Each run gets its own timestamped directory, so a previous result is never
silently overwritten and can always be compared against.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT_MARKERS = ("pyproject.toml", ".git")


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the project root by walking upwards from ``start``.

    Parameters
    ----------
    start : Path, optional
        Directory to start from. Defaults to the installed package location,
        which works for an editable install and falls back to the working
        directory otherwise.

    Returns
    -------
    Path
        First ancestor containing a marker from :data:`ROOT_MARKERS`, or the
        current working directory if none is found.
    """
    if start is None:
        start = Path(__file__).resolve().parent
    start = Path(start).resolve()

    for candidate in (start, *start.parents):
        if any((candidate / marker).exists() for marker in ROOT_MARKERS):
            return candidate
    return Path.cwd()


def data_root(repo_root: Path | None = None) -> Path:
    """Return the data directory, honouring the ``MVY_DATA_DIR`` override.

    Large datasets often live outside the repository. Setting the
    environment variable moves them without editing any config.
    """
    override = os.environ.get("MVY_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (repo_root or find_repo_root()) / "data"


def resolve_input(path: Path | str, repo_root: Path | None = None) -> Path:
    """Resolve a config-supplied path against the repository root.

    Absolute paths are returned unchanged. Relative paths are tried against
    the repository root first, then against the data root, so both
    ``data/datasets/x.hdf5`` and ``datasets/x.hdf5`` work.
    """
    path = Path(path).expanduser()
    if path.is_absolute():
        return path

    root = repo_root or find_repo_root()
    direct = (root / path).resolve()
    if direct.exists():
        return direct

    via_data = (data_root(root) / path).resolve()
    if via_data.exists():
        return via_data
    return direct


@dataclass
class RunPaths:
    """Directory layout of a single run.

    Parameters
    ----------
    root : Path
        Repository root.
    run_dir : Path
        Directory holding this run's outputs.
    """

    root: Path
    run_dir: Path

    @classmethod
    def for_run(
        cls,
        name: str,
        kind: str = "run",
        repo_root: Path | None = None,
        stamp: datetime | None = None,
    ) -> RunPaths:
        """Build a timestamped run directory descriptor.

        Parameters
        ----------
        name : str
            Case name, used in the directory name.
        kind : str
            ``"run"``, ``"ingest"`` or ``"plot"``; appears in the directory
            name so the purpose of a folder is visible from a listing.
        repo_root : Path, optional
            Overrides root discovery, mainly for tests.
        stamp : datetime, optional
            Fixed timestamp, mainly for tests.

        Notes
        -----
        No directory is created here. Call :meth:`prepare` for that.
        """
        root = repo_root or find_repo_root()
        stamp = stamp or datetime.now()
        label = f"{stamp:%Y-%m-%d_%H%M}_{kind}_{_slug(name)}"
        return cls(root=root, run_dir=root / "runs" / label)

    @property
    def models_dir(self) -> Path:
        """Shared artefact store; not per-run, since models are reused."""
        return self.root / "models"

    @property
    def plots_dir(self) -> Path:
        """Figures for this run."""
        return self.run_dir / "plots"

    @property
    def data_dir(self) -> Path:
        """Numerical outputs: ``results.npz``, exported meshes."""
        return self.run_dir / "data"

    @property
    def log_file(self) -> Path:
        """Full log of this run."""
        return self.run_dir / "run.log"

    @property
    def resolved_config(self) -> Path:
        """Copy of the merged configuration that produced this run."""
        return self.run_dir / "config.resolved.yaml"

    @property
    def summary_file(self) -> Path:
        """Human-readable report."""
        return self.run_dir / "summary.md"

    def prepare(self) -> RunPaths:
        """Create the run directories. The only place that touches the disk."""
        for directory in (self.run_dir, self.plots_dir, self.data_dir, self.models_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def plot_path(self, name: str, suffix: str = "", extension: str = "png") -> Path:
        """Build a figure path.

        Parameters
        ----------
        name : str
            Registered plot name.
        suffix : str
            Reduction label for 3D fields, e.g. ``"midspan"``. Appended
            after a double underscore so the base name stays greppable.
        extension : str
            File extension without the dot.
        """
        stem = f"{name}__{_slug(suffix)}" if suffix else name
        return self.plots_dir / f"{stem}.{extension}"


def _slug(text: str) -> str:
    """Reduce a label to a filename-safe ASCII slug.

    Non-ASCII characters are dropped rather than transliterated: output
    filenames stay portable, and every plot name in this package is already
    English by convention.
    """
    cleaned = [
        char if (char.isalnum() and char.isascii()) else "_" for char in text.strip().lower()
    ]
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unnamed"
