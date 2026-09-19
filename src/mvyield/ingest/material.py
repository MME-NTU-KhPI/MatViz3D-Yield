"""The material, read from the dataset that defines it.

A dataset is not just measurements: the crystal structure, the critical
resolved shear stress and the elastic constants were fixed when it was
generated, and the file records them. Restating them in a config would
create a second source of truth that eventually disagrees with the first --
which is why there is no material config file, only
``dataset.material_override`` for what a particular file does not carry.

Exported datasets carry nothing. They are written by the C++ pipeline,
which stores no attributes at all, so every property for those comes from
the override and this module's job is to say so clearly rather than to
invent a plausible number.

Provenance
----------
Every property records where it came from. A reader of a figure needs to
know whether ``crss = 150 MPa`` was read from the file or typed into a case
config, and that distinction disappears the moment the value is copied.

Units
-----
Datasets store stress in Pa; this package works in MPa, and so does
``material_override``. A value that looks like Pa is rejected rather than
used, because a CRSS wrong by six orders of magnitude yields a probability
field of zeros -- a failure that survives all the way to a figure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from mvyield.ingest.reader import DatasetReader
from mvyield.settings import ConfigError

log = logging.getLogger(__name__)

KNOWN_STRUCTURES = ("BCC", "BCC_48", "FCC")
"""Crystal structures the slip-system library covers.

HCP is absent on purpose. Its slip systems depend on the c/a ratio, so it
needs a material parameter rather than a table, and offering a default one
would quietly analyse a titanium dataset as if it were magnesium.
"""

DEFAULT_STRUCTURE = "BCC_48"
"""Structure assumed when nothing says otherwise.

Both the legacy analysis and the dataset generator use the 48-system BCC
family, so this default reproduces them. It is recorded as a default in the
provenance, never presented as something the file said.
"""

PA_TO_MPA = 1e-6

CRSS_MIN_MPA = 0.1
CRSS_MAX_MPA = 1.0e4
"""Plausibility window for a shear strength in MPa.

Wide enough for anything from soft aluminium to a hard steel, narrow enough
that a value left in Pa (``1.5e8``) cannot pass.
"""

OVERRIDE_KEYS = ("crss", "structure", "sigma_y", "c11", "c12", "c44")
"""Accepted keys of ``dataset.material_override``."""


@dataclass(frozen=True)
class MaterialProperties:
    """What the dataset says the material is.

    Parameters
    ----------
    crss : float or None
        Critical resolved shear stress, MPa. Required by the Schmid
        analysis; ``None`` when neither the file nor the override supplies
        one.
    structure : str
        Crystal structure, selecting the slip-system family.
    sigma_y : float or None
        Scalar yield strength, MPa. Used only by the von Mises analysis.
    elastic : dict
        Cubic stiffness constants ``c11``, ``c12``, ``c44`` in GPa, where
        the dataset records them.
    metadata : dict
        Non-material facts worth carrying into the report: which generator
        wrote the file, at which noise level, in which units.
    provenance : dict
        Property name to where its value came from.
    """

    crss: float | None = None
    structure: str = DEFAULT_STRUCTURE
    sigma_y: float | None = None
    elastic: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    def require_crss(self) -> float:
        """Return the CRSS, or explain what to do about its absence."""
        if self.crss is None:
            raise ConfigError(
                "the Schmid analysis needs a critical resolved shear stress, "
                "and this dataset carries none. Add it to the case config as "
                "dataset.material_override.crss, in MPa (150.0 for the "
                "BCC datasets in this project)."
            )
        return self.crss

    def require_sigma_y(self) -> float:
        """Return the scalar yield strength, or explain its absence."""
        if self.sigma_y is None:
            raise ConfigError(
                "the von Mises analysis needs a yield strength, and this "
                "dataset carries none. Add it to the case config as "
                "dataset.material_override.sigma_y, in MPa."
            )
        return self.sigma_y

    @classmethod
    def from_dict(cls, payload: dict) -> MaterialProperties:
        """Rebuild from :meth:`to_dict`, as stored in a yield-point artefact.

        Reading a cached artefact must give back what produced it, including
        the provenance -- otherwise a rerun that skipped the dataset would
        report its material as coming from nowhere.
        """
        return cls(
            crss=payload.get("crss"),
            structure=str(payload.get("structure", DEFAULT_STRUCTURE)),
            sigma_y=payload.get("sigma_y"),
            elastic=dict(payload.get("elastic") or {}),
            metadata=dict(payload.get("metadata") or {}),
            provenance=dict(payload.get("provenance") or {}),
        )

    def to_dict(self) -> dict:
        """Serialise for the model bundle and the run report."""
        return {
            "crss": self.crss,
            "structure": self.structure,
            "sigma_y": self.sigma_y,
            "elastic": dict(self.elastic),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    def describe(self) -> str:
        """Human-readable summary, each value followed by its source."""
        lines = []
        for name, value, unit in (
            ("crss", self.crss, "MPa"),
            ("sigma_y", self.sigma_y, "MPa"),
            ("structure", self.structure, ""),
        ):
            if value is None:
                continue
            shown = f"{value:g} {unit}".strip() if isinstance(value, float) else str(value)
            lines.append(f"{name:<10}: {shown}  ({self.provenance.get(name, 'unknown')})")

        if self.elastic:
            constants = ", ".join(f"{k}={v:g}" for k, v in sorted(self.elastic.items()))
            lines.append(f"{'elastic':<10}: {constants} GPa  "
                         f"({self.provenance.get('elastic', 'unknown')})")
        return "\n".join(lines)


def read_material(
    reader: DatasetReader,
    override: dict | None = None,
) -> MaterialProperties:
    """Read the material from a dataset, letting the override fill the gaps.

    Parameters
    ----------
    reader : DatasetReader
        An open reader.
    override : dict, optional
        ``dataset.material_override`` from the case config. Values are in
        MPa for stresses and GPa for stiffnesses. An override wins over the
        file: a value in a config is a deliberate statement, and refusing to
        honour it would leave no way to correct a mislabelled dataset.

    Returns
    -------
    MaterialProperties
    """
    override = dict(override or {})
    _check_override_keys(override)

    attrs = reader.attrs
    gen_config = attrs.get("gen_config") or {}
    if not isinstance(gen_config, dict):
        gen_config = {}

    provenance: dict[str, str] = {}

    crss = _resolve_stress(
        name="crss",
        override=override,
        file_value=attrs.get("crss", gen_config.get("crss")),
        file_source=(
            "dataset attribute 'crss'" if "crss" in attrs else "dataset attribute 'gen_config'"
        ),
        provenance=provenance,
    )
    sigma_y = _resolve_stress(
        name="sigma_y",
        override=override,
        file_value=attrs.get("sigma_y"),
        file_source="dataset attribute 'sigma_y'",
        provenance=provenance,
    )

    structure = str(override.get("structure", DEFAULT_STRUCTURE)).upper()
    provenance["structure"] = (
        "case config material_override" if "structure" in override else "package default"
    )
    if structure not in KNOWN_STRUCTURES:
        raise ConfigError(
            f"dataset.material_override.structure is {structure!r}; "
            f"known structures are {list(KNOWN_STRUCTURES)}"
        )

    elastic = _read_elastic(gen_config, override, provenance)

    return MaterialProperties(
        crss=crss,
        structure=structure,
        sigma_y=sigma_y,
        elastic=elastic,
        metadata={
            key: attrs[key]
            for key in ("generator", "noise_level", "units", "noise_spec")
            if key in attrs
        },
        provenance=provenance,
    )


# --- Helpers -----------------------------------------------------------------


def _resolve_stress(
    name: str,
    override: dict,
    file_value: object,
    file_source: str,
    provenance: dict[str, str],
) -> float | None:
    """Resolve one stress-valued property, override first, then the file."""
    if name in override:
        value = _as_mpa(name, override[name], source="case config", already_mpa=True)
        provenance[name] = "case config material_override"
        return value

    if file_value is not None:
        value = _as_mpa(name, file_value, source="dataset", already_mpa=False)
        provenance[name] = file_source
        return value

    return None


def _as_mpa(name: str, raw: object, source: str, already_mpa: bool) -> float:
    """Convert to MPa and reject a value that cannot be one.

    The dataset writes Pa, the config writes MPa. Both land here, and a
    number outside the plausibility window means the two were mixed up.
    """
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ConfigError(f"material property {name!r} is not a number: {raw!r}") from None

    converted = value if already_mpa else value * PA_TO_MPA

    if not CRSS_MIN_MPA <= converted <= CRSS_MAX_MPA:
        hint = (
            f"values from a {source} are read as "
            f"{'MPa' if already_mpa else 'Pa'}"
        )
        raise ConfigError(
            f"material property {name!r} works out to {converted:g} MPa, "
            f"outside the plausible range "
            f"{CRSS_MIN_MPA:g}..{CRSS_MAX_MPA:g} MPa ({hint}). "
            f"A value off by a factor of 1e6 is a Pa/MPa mix-up."
        )
    return converted


def _read_elastic(
    gen_config: dict,
    override: dict,
    provenance: dict[str, str],
) -> dict[str, float]:
    """Collect the cubic stiffness constants, in GPa.

    Generated datasets record these in Pa inside ``gen_config``; the
    override states them in GPa, as elastic constants are normally quoted.
    """
    constants: dict[str, float] = {}
    sources: set[str] = set()

    for key in ("c11", "c12", "c44"):
        if key in override:
            constants[key] = float(override[key])
            sources.add("case config material_override")
        elif key in gen_config:
            constants[key] = float(gen_config[key]) * 1e-9
            sources.add("dataset attribute 'gen_config'")

    if constants:
        provenance["elastic"] = " and ".join(sorted(sources))
    return constants


def _check_override_keys(override: dict) -> None:
    """Reject override keys this package does not know.

    A misspelled key would otherwise be dropped in silence and the run would
    proceed on the dataset's own value, which is indistinguishable from the
    override having been applied.
    """
    unknown = sorted(set(override) - set(OVERRIDE_KEYS))
    if unknown:
        raise ConfigError(
            f"dataset.material_override does not accept {unknown}; "
            f"accepted keys are {list(OVERRIDE_KEYS)}"
        )
