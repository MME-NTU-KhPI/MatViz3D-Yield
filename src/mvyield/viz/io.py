"""Writing figures and fields to disk.

One function saves every figure. Previously each plotting function called
``savefig`` itself, always at 150 dpi and always PNG only, so raising the
resolution or adding a vector format meant editing forty call sites, and any
that were missed stayed at the old setting without anyone noticing.

Two rules are enforced rather than suggested:

* **PNG at 300 dpi or better.** The config schema rejects anything lower, so
  the check has already happened by the time a figure is drawn.
* **SVG alongside the raster, by default.** Vector output is what makes a
  figure editable at proof stage. Producing it costs nothing at draw time
  and cannot be recovered afterwards from a raster.

Filenames are ASCII slugs. Reduction suffixes are separated by a double
underscore so ``kde_probability__midspan.svg`` still matches a search for
the plot name.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from mvyield.paths import RunPaths
from mvyield.settings import PlotConfig

log = logging.getLogger(__name__)


def save_figure(
    fig,
    name: str,
    paths: RunPaths,
    config: PlotConfig,
    suffix: str = "",
    close: bool = True,
) -> list[Path]:
    """Write one figure in every configured format.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to write.
    name : str
        Registered plot name, used as the filename stem.
    paths : RunPaths
        Run layout supplying the plots directory.
    config : PlotConfig
        Formats and resolution, already validated.
    suffix : str
        Reduction label for a 3D field, such as ``"midspan"``.
    close : bool
        Close the figure afterwards. Leaving figures open exhausts memory
        over a run that draws dozens.

    Returns
    -------
    list of Path
        Files written, in the order the formats were configured.
    """
    import matplotlib.pyplot as plt

    written: list[Path] = []
    for extension in config.formats:
        target = paths.plot_path(name, suffix=suffix, extension=extension)
        target.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            target,
            dpi=config.dpi if extension == "png" else None,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            transparent=False,
        )
        written.append(target)

    log.debug("wrote %s (%s)", name, ", ".join(config.formats))

    if close:
        plt.close(fig)
    return written


def export_vtu(
    points: np.ndarray,
    fields: dict[str, np.ndarray],
    path: Path,
) -> Path:
    """Write a point cloud with scalar fields for ParaView.

    A wire is a three-dimensional body, and matplotlib is the wrong tool for
    inspecting one: it has no depth buffer worth the name, and building an
    isosurface viewer would take longer than the analysis. Exporting the
    field lets ParaView do what it is good at -- isosurfaces of ``P(G)``,
    arbitrary cuts, rotation -- while the figures here stay two-dimensional
    and publication-shaped.

    Parameters
    ----------
    points : ndarray of shape (M, 3)
        Coordinates.
    fields : dict of str to ndarray
        Scalar fields of length ``M``. Non-finite values are preserved;
        ParaView handles them.
    path : Path
        Output path, conventionally ``.vtu``.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    ImportError
        If ``meshio`` is missing, with the install command.
    """
    try:
        import meshio
    except ImportError:
        raise ImportError(
            "VTK export needs meshio, which is not installed.\n"
            'Install it with: pip install -e ".[vtk]"\n'
            "Or set export.vtk to false in the case config."
        ) from None

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (M, 3), got {points.shape}")

    bad = {k: len(v) for k, v in fields.items() if len(v) != len(points)}
    if bad:
        raise ValueError(f"field lengths do not match {len(points)} points: {bad}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Vertex cells keep this a valid unstructured grid without needing the
    # mesh connectivity, which nodal exports do not always carry.
    cells = [("vertex", np.arange(len(points), dtype=np.int64)[:, None])]
    point_data = {
        name: np.asarray(values, dtype=np.float64) for name, values in fields.items()
    }

    meshio.write_points_cells(str(path), points, cells, point_data=point_data)
    log.info("wrote %s (%d points, %d fields)", path.name, len(points), len(fields))
    return path
