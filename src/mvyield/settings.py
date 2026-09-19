"""Typed configuration: load, merge with defaults, validate, freeze.

Design rules that the rest of the package relies on:

* **A config states intent; a bundle records fact.** Nothing here is ever
  written back to disk by a command. Calibration results and resolved
  material properties live in artefacts, so a config file always reflects
  what the user asked for, not what a previous run happened to produce.
* **Fail at load, not mid-run.** Every constraint that can be checked without
  touching data is checked here -- an unknown method name, a ``primary`` that
  is not in ``methods``, a DPI below the publication floor. A twenty-minute
  KDE evaluation should never die on a typo.
* **Scalars expand to specs.** ``power: 8.0`` and
  ``power: {mode: manual, value: 8.0}`` mean the same thing, so simple cases
  stay simple while the calibration machinery still sees a uniform shape.

Only ``defaults.yaml`` supplies defaults; dataclass field defaults exist so
that objects can be constructed in tests without a file.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mvyield.mechanics.voigt import VoigtConvention, get_convention

MIN_DPI = 300
"""Publication floor for raster output; enforced rather than suggested."""

KNOWN_METHODS = ("analytic", "kde_ray", "kde_slice", "pinn")
KNOWN_FIELD_SOURCES = ("analytic", "ansys", "pinn")
KNOWN_CASES = ("kirsch", "ansys")


class ConfigError(ValueError):
    """Raised when a configuration is malformed or self-inconsistent.

    Messages name the offending key path so the user can find it without
    reading a traceback.
    """


# --- Hyper-parameters --------------------------------------------------------


@dataclass(frozen=True)
class HyperParam:
    """A tunable value that may be set by hand or calibrated.

    Parameters
    ----------
    mode : str
        ``"manual"`` uses ``value`` as given. ``"auto"`` asks the method's
        calibrator to select one. ``"frozen"`` takes whatever is already
        recorded in the model bundle and refuses to recompute it.
    value : float, optional
        Required for ``"manual"``.
    objective : str, optional
        Name of the selection criterion for ``"auto"``.
    grid : list of float, optional
        Candidate values searched by ``"auto"``.
    refine : bool
        Whether to refine around the best grid point.
    """

    mode: str = "manual"
    value: float | None = None
    objective: str | None = None
    grid: tuple[float, ...] | None = None
    refine: bool = False

    def __post_init__(self) -> None:
        """Check the combination of mode and payload."""
        if self.mode not in ("manual", "auto", "frozen"):
            raise ConfigError(
                f"hyper-parameter mode must be manual, auto or frozen; got {self.mode!r}"
            )
        if self.mode == "manual" and self.value is None:
            raise ConfigError("hyper-parameter mode 'manual' requires a value")
        if self.mode == "auto" and not self.grid:
            raise ConfigError("hyper-parameter mode 'auto' requires a search grid")

    @classmethod
    def parse(cls, raw: Any, *, where: str) -> HyperParam:
        """Build from either a bare scalar or a mapping.

        Parameters
        ----------
        raw : float or dict
            ``8.0`` is shorthand for ``{"mode": "manual", "value": 8.0}``.
        where : str
            Key path used in error messages.
        """
        if raw is None:
            raise ConfigError(f"{where}: missing hyper-parameter")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return cls(mode="manual", value=float(raw))
        if not isinstance(raw, dict):
            raise ConfigError(
                f"{where}: expected a number or a mapping, got {type(raw).__name__}"
            )

        grid = raw.get("grid")
        try:
            return cls(
                mode=str(raw.get("mode", "manual")),
                value=None if raw.get("value") is None else float(raw["value"]),
                objective=raw.get("objective"),
                grid=tuple(float(g) for g in grid) if grid else None,
                refine=bool(raw.get("refine", False)),
            )
        except ConfigError as exc:
            raise ConfigError(f"{where}: {exc}") from None

    def describe(self) -> str:
        """Short human-readable form for logs."""
        if self.mode == "manual":
            return f"{self.value:g} (manual)"
        if self.mode == "frozen":
            return "from bundle (frozen)"
        return f"auto via {self.objective} over {list(self.grid or ())}"


# --- Sections ----------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Input 1: the dataset, which is also the material definition.

    Material properties are read from the HDF5 attributes where present;
    ``material_override`` fills only what the file does not carry. There is
    deliberately no separate material config file -- a third source of truth
    about the material would eventually disagree with the dataset.
    """

    source: Path | None = None
    analysis: str = "SCHMID"
    select_step: str = "max_macro_sz"
    material_override: dict = field(default_factory=dict)
    plots_preset: str = "ingest_diagnostics"

    def __post_init__(self) -> None:
        """Normalise and validate the analysis type."""
        if self.analysis.upper() not in ("SCHMID", "VON_MISES"):
            raise ConfigError(
                f"dataset.analysis must be SCHMID or VON_MISES; got {self.analysis!r}"
            )
        self.analysis = self.analysis.upper()
        if self.source is not None:
            self.source = Path(self.source)


@dataclass
class FieldConfig:
    """Input 2: the stress field of the problem being assessed."""

    source: str = "analytic"
    path: Path | None = None
    format: str = "prnsol_csv"
    units: dict = field(default_factory=lambda: {"stress": "MPa", "length": "mm"})
    columns: dict = field(default_factory=dict)
    convention: str = "mandel_xy_last"
    frame: str = "global"
    sigma_applied: float = 150.0
    hole_radius: dict = field(
        default_factory=lambda: {"auto": True, "fraction": 0.15, "fallback": 1.0}
    )
    domain_from: str = "dataset"
    grid: dict = field(default_factory=lambda: {"resolution": 400})

    def __post_init__(self) -> None:
        """Validate the source and its required companions."""
        if self.source not in KNOWN_FIELD_SOURCES:
            raise ConfigError(
                f"field.source must be one of {KNOWN_FIELD_SOURCES}; got {self.source!r}"
            )
        if self.source == "ansys" and self.path is None:
            raise ConfigError("field.source 'ansys' requires field.path")
        if self.path is not None:
            self.path = Path(self.path)
        get_convention(self.convention)

    @property
    def voigt_convention(self) -> VoigtConvention:
        """Resolved convention object for this field."""
        return get_convention(self.convention)


@dataclass
class ModelConfig:
    """Yield model: shared space, plus one estimator per requested method."""

    path: str = "auto"
    deviatoric: bool = True
    convention: str = "mandel_xy_last"
    normalise: bool = False
    methods: tuple[str, ...] = ("kde_ray",)
    primary: str | None = None
    kde_ray: dict = field(default_factory=dict)
    kde_slice: dict = field(default_factory=dict)
    analytic: dict = field(default_factory=dict)
    pinn: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate method names, the primary choice and hyper-parameters."""
        self.methods = tuple(self.methods)
        if not self.methods:
            raise ConfigError("model.methods must list at least one method")

        unknown = [m for m in self.methods if m not in KNOWN_METHODS]
        if unknown:
            raise ConfigError(
                f"model.methods contains unknown entries {unknown}; "
                f"known methods are {list(KNOWN_METHODS)}"
            )
        duplicates = {m for m in self.methods if self.methods.count(m) > 1}
        if duplicates:
            raise ConfigError(f"model.methods lists {sorted(duplicates)} more than once")

        if self.primary is None:
            self.primary = self.methods[0]
        if self.primary not in self.methods:
            raise ConfigError(
                f"model.primary is {self.primary!r} but methods are "
                f"{list(self.methods)}; primary must be one of them"
            )
        get_convention(self.convention)

        # Parse hyper-parameters eagerly so a bad spec fails at load time.
        self.hyperparams: dict[str, HyperParam] = {}
        if "kde_ray" in self.methods and "power" in self.kde_ray:
            self.hyperparams["power"] = HyperParam.parse(
                self.kde_ray["power"], where="model.kde_ray.power"
            )
        if "kde_slice" in self.methods and "kappa" in self.kde_slice:
            self.hyperparams["kappa"] = HyperParam.parse(
                self.kde_slice["kappa"], where="model.kde_slice.kappa"
            )

    @property
    def voigt_convention(self) -> VoigtConvention:
        """Resolved convention object for the model space."""
        return get_convention(self.convention)

    @property
    def runs_comparison(self) -> bool:
        """Whether a method comparison is meaningful for this configuration."""
        return len(self.methods) >= 2


@dataclass
class PlotConfig:
    """Which figures to draw and how to write them out."""

    preset: str = "paper"
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    formats: tuple[str, ...] = ("png", "svg")
    dpi: int = MIN_DPI

    def __post_init__(self) -> None:
        """Enforce the raster floor and the vector-output requirement."""
        self.include = tuple(self.include)
        self.exclude = tuple(self.exclude)
        self.formats = tuple(f.lower().lstrip(".") for f in self.formats)

        if not self.formats:
            raise ConfigError("plots.formats must list at least one format")
        unknown = [f for f in self.formats if f not in ("png", "svg", "pdf")]
        if unknown:
            raise ConfigError(f"plots.formats contains unsupported entries {unknown}")
        if "png" in self.formats and self.dpi < MIN_DPI:
            raise ConfigError(
                f"plots.dpi is {self.dpi}; PNG output for publication requires "
                f"at least {MIN_DPI}. Raise the value or drop 'png' from formats."
            )
        overlap = set(self.include) & set(self.exclude)
        if overlap:
            raise ConfigError(f"plots.include and plots.exclude both list {sorted(overlap)}")


@dataclass
class ViewConfig:
    """How a 3D field is reduced to 2D for plotting."""

    reductions: tuple[dict, ...] = ()

    def __post_init__(self) -> None:
        """Validate reduction kinds."""
        self.reductions = tuple(self.reductions)
        known = ("slice", "unroll", "projection")
        for i, red in enumerate(self.reductions):
            kind = red.get("kind")
            if kind not in known:
                raise ConfigError(
                    f"view.reductions[{i}].kind is {kind!r}; expected one of {known}"
                )


@dataclass
class ProbeConfig:
    """Sampling paths and points for profile plots."""

    paths: tuple[dict, ...] = ()
    points: tuple[dict, ...] = ()

    def __post_init__(self) -> None:
        """Validate path kinds and require names for cross-referencing."""
        self.paths = tuple(self.paths)
        self.points = tuple(self.points)
        known_paths = ("line", "arc", "ray")
        for i, path in enumerate(self.paths):
            if path.get("kind") not in known_paths:
                raise ConfigError(
                    f"probes.paths[{i}].kind is {path.get('kind')!r}; "
                    f"expected one of {known_paths}"
                )
            if not path.get("name"):
                raise ConfigError(f"probes.paths[{i}] needs a name")


@dataclass
class CompareConfig:
    """Method comparison; active only when at least two methods are run."""

    p_crit: float = 0.5
    reference_locus: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the decision threshold."""
        if not 0.0 < self.p_crit < 1.0:
            raise ConfigError(f"compare.p_crit must lie in (0, 1); got {self.p_crit}")


@dataclass
class ExportConfig:
    """Non-figure outputs."""

    vtk: bool = False
    results_npz: bool = True


@dataclass
class CaseConfig:
    """A fully resolved case: everything one run needs.

    Built by :func:`load_case`, never constructed directly from user input,
    so that every instance has already passed validation.
    """

    case: str
    name: str
    dataset: DatasetConfig
    field: FieldConfig
    model: ModelConfig
    plots: PlotConfig
    view: ViewConfig
    probes: ProbeConfig
    compare: CompareConfig
    export: ExportConfig
    seed: int = 0
    report_lang: str = "en"
    source_path: Path | None = None
    raw: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate cross-section consistency."""
        if self.case not in KNOWN_CASES:
            raise ConfigError(f"case must be one of {KNOWN_CASES}; got {self.case!r}")
        if self.case == "ansys" and self.field.source != "ansys":
            raise ConfigError(
                f"case 'ansys' expects field.source 'ansys', got {self.field.source!r}"
            )
        if self.report_lang not in ("uk", "en"):
            raise ConfigError(f"report.lang must be 'uk' or 'en'; got {self.report_lang!r}")

    @property
    def needs_dataset(self) -> bool:
        """Whether any requested step must open the dataset.

        Mode B -- evaluating an existing model on a new field -- does not,
        which is what keeps h5py out of the import path there.
        """
        return self.dataset.source is not None

    def model_path(self, models_dir: Path, dataset_key: str) -> Path:
        """Resolve where this case's model bundle lives.

        Parameters
        ----------
        models_dir : Path
            Root of the model store.
        dataset_key : str
            Stable identifier of the dataset, used when ``model.path`` is
            ``"auto"`` so that two datasets never collide.
        """
        if self.model.path == "auto":
            return models_dir / f"{dataset_key}.model.npz"
        return Path(self.model.path)

    def describe(self) -> str:
        """Multi-line summary printed by ``--dry-run``."""
        lines = [
            f"case         : {self.case} ({self.name})",
            f"dataset      : {self.dataset.source or '(none -- mode B)'}",
            f"field        : {self.field.source}"
            + (f" <- {self.field.path}" if self.field.path else ""),
            f"space        : {self.model.convention}, "
            + ("deviatoric" if self.model.deviatoric else "full"),
            f"methods      : {', '.join(self.model.methods)}  (primary: {self.model.primary})",
            f"comparison   : {'on' if self.model.runs_comparison else 'off (single method)'}",
            f"plots        : preset {self.plots.preset}, "
            f"{'+'.join(self.plots.formats)} @ {self.plots.dpi} dpi",
        ]
        for key, spec in self.model.hyperparams.items():
            lines.append(f"  {key:<11s}: {spec.describe()}")
        return "\n".join(lines)


# --- Loading -----------------------------------------------------------------


def load_yaml(path: Path) -> dict:
    """Read a YAML file into a dictionary.

    Raises
    ------
    ConfigError
        If the file is missing or does not contain a mapping.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Nested mappings are merged key by key; every other type, lists included,
    is replaced wholesale. Replacing lists is deliberate: a user writing
    ``methods: [kde_slice]`` means exactly that, not "add to the defaults".
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    """Apply ``--set key.path=value`` arguments to a config dictionary.

    Values are parsed as YAML, so ``true``, ``3``, ``[a, b]`` and ``null``
    all behave as they would inside the file.
    """
    result = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        key_path, raw_value = item.split("=", 1)
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ConfigError(f"--set {item}: cannot parse value ({exc})") from None

        node = result
        keys = key_path.strip().split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ConfigError(f"--set {item}: {key!r} is not a mapping")
        node[keys[-1]] = value
    return result


def build_case(data: dict, source_path: Path | None = None) -> CaseConfig:
    """Turn a merged dictionary into a validated :class:`CaseConfig`.

    Parameters
    ----------
    data : dict
        Result of merging defaults with a case file and any ``--set``
        overrides.
    source_path : Path, optional
        Origin of the case file, recorded for provenance.
    """
    dataset_raw = dict(data.get("dataset") or {})
    plots_preset = (dataset_raw.pop("plots", {}) or {}).get("preset", "ingest_diagnostics")

    model_raw = dict(data.get("model") or {})
    plots_raw = dict(data.get("plots") or {})
    report_raw = dict(data.get("report") or {})

    try:
        return CaseConfig(
            case=str(data.get("case", "kirsch")),
            name=str(data.get("name") or (source_path.stem if source_path else "unnamed")),
            dataset=DatasetConfig(plots_preset=plots_preset, **dataset_raw),
            field=FieldConfig(**(data.get("field") or {})),
            model=ModelConfig(**model_raw),
            plots=PlotConfig(**plots_raw),
            view=ViewConfig(**(data.get("view") or {})),
            probes=ProbeConfig(**(data.get("probes") or {})),
            compare=CompareConfig(**(data.get("compare") or {})),
            export=ExportConfig(**(data.get("export") or {})),
            seed=int(data.get("seed", 0)),
            report_lang=str(report_raw.get("lang", "en")),
            source_path=source_path,
            raw=data,
        )
    except TypeError as exc:
        raise ConfigError(f"unexpected or missing configuration key: {exc}") from None


def load_case(
    case_path: Path,
    defaults_path: Path | None = None,
    overrides: list[str] | None = None,
) -> CaseConfig:
    """Load a case config, merged over the defaults and validated.

    Parameters
    ----------
    case_path : Path
        Case YAML file.
    defaults_path : Path, optional
        Defaults file. Located next to the case file's ``configs`` root when
        omitted.
    overrides : list of str, optional
        ``key.path=value`` strings from the command line.

    Returns
    -------
    CaseConfig
        A validated configuration. Any problem raises :class:`ConfigError`
        before any data is read.
    """
    case_path = Path(case_path)
    case_raw = load_yaml(case_path)

    if defaults_path is None:
        candidate = case_path.parent.parent / "defaults.yaml"
        defaults_path = candidate if candidate.exists() else None

    merged = load_yaml(defaults_path) if defaults_path else {}
    merged = deep_merge(merged, case_raw)
    if overrides:
        merged = apply_overrides(merged, overrides)

    # A preset name is resolved to a plot list later, by the viz registry;
    # the plot_presets table itself is not part of the case object.
    merged.pop("plot_presets", None)

    return build_case(merged, source_path=case_path)


def dump_resolved(cfg: CaseConfig, path: Path) -> None:
    """Write the fully merged configuration into a run directory.

    This is what makes a run reproducible: the case file may change later,
    but the copy stored beside the results records exactly what ran.
    """
    payload = copy.deepcopy(cfg.raw)
    payload["_resolved"] = {
        "name": cfg.name,
        "seed": cfg.seed,
        "methods": list(cfg.model.methods),
        "primary": cfg.model.primary,
        "source_path": str(cfg.source_path) if cfg.source_path else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
