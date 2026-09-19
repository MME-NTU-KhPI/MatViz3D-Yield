"""Importing a stress field exported from ANSYS.

What arrives is a text table of nodal results. The work is not the parsing
but the three things that silently go wrong around it:

**Order and scaling.** ANSYS writes engineering shear components -- plain
tensor entries, no ``sqrt(2)`` -- in the order SX, SY, SZ, SXY, SYZ, SXZ.
The model space is Mandel-scaled with a different shear order. The factor is
applied exactly once, here, by
:func:`~mvyield.mechanics.voigt.from_components`; nothing downstream
rescales, and nothing downstream needs to know where the numbers came from.

**Units.** Exports are usually in Pa and metres, the package works in MPa
and millimetres. Converting on import means a field is either right at the
boundary or wrong immediately, rather than plausible-looking and off by a
million.

**Frame.** A result exported in a local or cylindrical coordinate system
cannot be compared against a model built in the global frame, and the
comparison would still produce a picture. The frame is carried and checked
rather than assumed.

No pandas
---------
Reading a table is not worth a dependency that only the dataset extra
installs: evaluating a model on an exported field must work with the core
install alone.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from mvyield.mechanics import voigt
from mvyield.mechanics.field import FieldMeta, GeometryHints, StressField, Topology
from mvyield.settings import ConfigError, FieldConfig

log = logging.getLogger(__name__)

KNOWN_FORMATS = ("prnsol_csv", "mechanical_txt")
"""Text layouts this adapter reads."""

STRESS_TO_MPA = {"Pa": 1.0e-6, "kPa": 1.0e-3, "MPa": 1.0, "GPa": 1.0e3, "psi": 6.894757e-3}
LENGTH_TO_MM = {"m": 1.0e3, "cm": 10.0, "mm": 1.0, "um": 1.0e-3, "in": 25.4}

DEFAULT_COLUMNS = {
    "node": "NODE",
    "xyz": ["X", "Y", "Z"],
    "stress": ["SX", "SY", "SZ", "SXY", "SYZ", "SXZ"],
}
"""Column names PRNSOL writes, used when the config names none."""

PLANE_TOLERANCE = 1e-9
"""Spread in z below which a field counts as confined to a plane."""


def build_field(config: FieldConfig, extent: float | None = None) -> StressField:
    """Read an exported stress field.

    Parameters
    ----------
    config : FieldConfig
        The ``field`` section, including ``path``, ``columns`` and ``units``.
    extent : float, optional
        Ignored; an imported field carries its own geometry.

    Returns
    -------
    StressField
    """
    if config.path is None:
        raise ConfigError("field.source 'ansys' requires field.path")
    if config.format not in KNOWN_FORMATS:
        raise ConfigError(
            f"field.format {config.format!r} is not supported; "
            f"available: {list(KNOWN_FORMATS)}"
        )

    columns = _resolve_columns(config.columns)
    wanted = [*columns["xyz"], *columns["stress"]]
    table = read_table(Path(config.path), wanted)

    stress_scale = _unit_factor(
        (config.units or {}).get("stress", "MPa"), STRESS_TO_MPA, "stress"
    )
    length_scale = _unit_factor(
        (config.units or {}).get("length", "mm"), LENGTH_TO_MM, "length"
    )

    points = np.column_stack([table[name] for name in columns["xyz"]]) * length_scale
    components = [table[name] * stress_scale for name in columns["stress"]]

    # The export's own order is SX, SY, SZ, SXY, SYZ, SXZ, with engineering
    # shear; from_components applies the convention's factor once.
    sigma6 = voigt.from_components(*components, convention=config.voigt_convention)

    topology = _infer_topology(points)
    log.info(
        "ANSYS field: %d nodes from %s, %s, stresses %s -> MPa",
        len(points),
        Path(config.path).name,
        topology.value,
        (config.units or {}).get("stress", "MPa"),
    )

    return StressField(
        points=points,
        sigma6=sigma6,
        topology=topology,
        meta=FieldMeta(
            source="ansys",
            convention=config.voigt_convention,
            stress_unit="MPa",
            length_unit="mm",
            frame=config.frame,
            extra={
                "case": "ansys",
                "path": str(config.path),
                "format": config.format,
                "source_stress_unit": (config.units or {}).get("stress", "MPa"),
                "source_length_unit": (config.units or {}).get("length", "mm"),
            },
        ),
        hints=GeometryHints(length_unit="mm"),
    )


def read_table(path: Path, wanted: list[str]) -> dict[str, np.ndarray]:
    """Read a whitespace- or comma-separated table by column name.

    PRNSOL paginates: it repeats the header, and interleaves blank lines and
    banner text between blocks of numbers. So rather than assuming a layout,
    this finds the header naming the wanted columns and then keeps every
    fully numeric row with the right width, whatever sits between them.

    Parameters
    ----------
    path : Path
        Text file.
    wanted : list of str
        Column names that must be present.

    Returns
    -------
    dict
        Column name to values.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"exported field not found: {path}\n"
            f"Export it from ANSYS, or correct field.path in the case config."
        )

    header: list[str] | None = None
    rows: list[list[float]] = []

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens = _split(line)
            if not tokens:
                continue

            if header is None or not _is_numeric(tokens):
                if set(wanted).issubset(tokens):
                    header = tokens
                continue

            if len(tokens) == len(header):
                rows.append([float(token) for token in tokens])

    if header is None:
        raise ConfigError(
            f"{path.name} has no header naming {wanted}. Check field.columns "
            f"against the file, or re-export with PRNSOL,S,COMP."
        )
    if not rows:
        raise ConfigError(f"{path.name} has a header but no numeric rows")

    values = np.asarray(rows, dtype=np.float64)
    return {name: values[:, header.index(name)] for name in wanted}


# --- Helpers -----------------------------------------------------------------


def _resolve_columns(configured: dict | None) -> dict:
    """Merge the configured column names over the PRNSOL defaults."""
    columns = {key: list(value) if isinstance(value, list) else value
               for key, value in DEFAULT_COLUMNS.items()}
    columns.update(configured or {})

    for key in ("xyz", "stress"):
        expected = len(DEFAULT_COLUMNS[key])
        if len(columns[key]) != expected:
            raise ConfigError(
                f"field.columns.{key} must name {expected} columns, "
                f"got {columns[key]}"
            )
    return columns


def _unit_factor(unit: str, table: dict[str, float], what: str) -> float:
    """Convert a declared unit into the package's own."""
    try:
        return table[str(unit)]
    except KeyError:
        raise ConfigError(
            f"field.units.{what} is {unit!r}; known units: {sorted(table)}"
        ) from None


def _infer_topology(points: np.ndarray) -> Topology:
    """Decide whether an imported cloud is planar or genuinely 3D."""
    spread = float(np.ptp(points[:, 2])) if len(points) else 0.0
    return Topology.CLOUD2D if spread <= PLANE_TOLERANCE else Topology.CLOUD3D


def _split(line: str) -> list[str]:
    """Split a line on commas or whitespace, whichever it uses."""
    stripped = line.strip()
    if not stripped:
        return []
    parts = stripped.split(",") if "," in stripped else stripped.split()
    return [part.strip() for part in parts if part.strip()]


def _is_numeric(tokens: list[str]) -> bool:
    """Whether every token parses as a number."""
    for token in tokens:
        try:
            float(token)
        except ValueError:
            return False
    return True
