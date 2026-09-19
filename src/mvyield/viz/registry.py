"""The plot registry: which figures exist, and when each one applies.

A figure is not universally applicable. Some need a particular method to
have run, some only make sense on a regular grid, some describe the material
and work for any problem at all. In the pre-refactor code these conditions
lived in the control flow of the case class, so every figure was drawn every
run and a new problem meant editing that flow.

Here each plotting function declares its own conditions and the registry
answers the question "what can be drawn for this result?".

Spaces
------
Sorting the forty-odd legacy functions by what they actually draw gives four
groups, and three of them do not depend on the problem at all:

``material``
    The yield-point cloud, the locus, the pi-plane. These describe the
    material. They were the dataset-analysis figures and they work unchanged
    for any case, because they never touch the geometry.
``aggregate``
    Distributions, method comparisons, calibration curves. A histogram of
    ``P(G)`` does not know whether it came from a plate or a wire.
``field``
    Quantities over the body. The only group that depends on topology.
``probe``
    Values along a path or at a point. Depends on the sampling paths, which
    the config supplies.

So the work of supporting a new problem falls almost entirely on ``field``
and ``probe``, and a figure that needed changing to support a wire is a
figure attached to the wrong layer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from mvyield.mechanics.field import Topology

log = logging.getLogger(__name__)


class Space(StrEnum):
    """What kind of thing a figure draws."""

    MATERIAL = "material"
    AGGREGATE = "aggregate"
    FIELD = "field"
    PROBE = "probe"

    @property
    def depends_on_geometry(self) -> bool:
        """Whether the figure needs the problem's geometry to make sense."""
        return self in (Space.FIELD, Space.PROBE)


ALL_TOPOLOGIES = tuple(Topology)


@dataclass(frozen=True)
class PlotSpec:
    """One registered figure and the conditions under which it applies.

    Parameters
    ----------
    name : str
        Identifier used in presets, configs and filenames.
    function : callable
        The plotting function.
    space : Space
        Which group it belongs to.
    groups : tuple of str
        Preset tags such as ``"paper"`` or ``"diagnostics"``.
    needs : tuple of str
        Methods that must have run. Empty means no requirement.
    topology : tuple of Topology
        Field arrangements it supports. Ignored outside :attr:`Space.FIELD`.
    needs_diagnostics : tuple of str
        Per-point diagnostic fields it requires, such as ``"eff_n"``.
    per_reduction : bool
        Whether to draw once per 2D reduction of a 3D field. Field figures
        do; a histogram over the whole volume does not.
    layout : tuple of int, optional
        Rows and columns of a small-multiple grid. ``None`` gives a single
        axes. A figure that needs a panel per stress component should say so
        here rather than dismantling the axes it was handed.
    figsize : tuple of float, optional
        Figure size in inches. Defaults to something sensible for the
        layout.
    description : str
        One line shown by ``mvy plot --list``.
    """

    name: str
    function: Callable
    space: Space
    groups: tuple[str, ...] = ()
    needs: tuple[str, ...] = ()
    topology: tuple[Topology, ...] = ALL_TOPOLOGIES
    needs_diagnostics: tuple[str, ...] = ()
    per_reduction: bool = False
    layout: tuple[int, int] | None = None
    figsize: tuple[float, float] | None = None
    description: str = ""

    def applies_to(
        self,
        methods: tuple[str, ...],
        topology: Topology | None = None,
        diagnostics: tuple[str, ...] = (),
    ) -> tuple[bool, str]:
        """Decide whether this figure can be drawn.

        Returns
        -------
        applies : bool
        reason : str
            Why not, when it does not apply. Logged so a missing figure is
            explained rather than merely absent.
        """
        missing = [m for m in self.needs if m not in methods]
        if missing:
            return False, f"needs method(s) {', '.join(missing)}"

        if self.space is Space.FIELD and topology is not None:
            if topology not in self.topology:
                return False, f"does not support topology {topology.value}"

        absent = [d for d in self.needs_diagnostics if d not in diagnostics]
        if absent:
            return False, f"needs diagnostic field(s) {', '.join(absent)}"

        return True, ""


REGISTRY: dict[str, PlotSpec] = {}
"""Every registered figure, keyed by name."""


def plot(
    name: str,
    space: Space | str,
    groups: tuple[str, ...] | list[str] = (),
    needs: tuple[str, ...] | list[str] = (),
    topology: tuple[Topology, ...] | list[Topology] = ALL_TOPOLOGIES,
    needs_diagnostics: tuple[str, ...] | list[str] = (),
    per_reduction: bool | None = None,
    layout: tuple[int, int] | None = None,
    figsize: tuple[float, float] | None = None,
    description: str = "",
) -> Callable:
    """Register a plotting function.

    Examples
    --------
    ::

        @plot("kde_probability", space=Space.FIELD, groups=["paper"],
              needs=["kde_ray"], description="Yield probability over the body")
        def plot_kde_probability(result, ax, theme):
            ...

    Notes
    -----
    ``per_reduction`` defaults to ``True`` for field figures, since a 3D
    field must be reduced before it can be drawn, and to ``False``
    otherwise.
    """
    space = Space(space)
    if per_reduction is None:
        per_reduction = space is Space.FIELD

    def decorator(function: Callable) -> Callable:
        if name in REGISTRY:
            raise ValueError(
                f"plot {name!r} is already registered by {REGISTRY[name].function.__module__}"
            )
        REGISTRY[name] = PlotSpec(
            name=name,
            function=function,
            space=space,
            groups=tuple(groups),
            needs=tuple(needs),
            topology=tuple(topology),
            needs_diagnostics=tuple(needs_diagnostics),
            per_reduction=per_reduction,
            layout=tuple(layout) if layout else None,
            figsize=tuple(figsize) if figsize else None,
            description=description or (function.__doc__ or "").strip().split("\n")[0],
        )
        return function

    return decorator


def resolve_preset(
    preset: str,
    presets: dict[str, list[str]],
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> list[str]:
    """Turn a preset name plus adjustments into a list of figure names.

    Parameters
    ----------
    preset : str
        Preset key, or ``"all"``.
    presets : dict
        The ``plot_presets`` table from the defaults file.
    include, exclude : tuple of str
        Names added to or removed from the preset.

    Raises
    ------
    KeyError
        For an unknown preset or an unknown figure name, listing the valid
        options. A typo in a config should not silently produce no figure.
    """
    if preset not in presets and preset != "all":
        raise KeyError(
            f"unknown plot preset {preset!r}; available: {sorted(presets)} (or 'all')"
        )

    entries = presets.get(preset, ["*"]) if preset != "all" else ["*"]
    names = sorted(REGISTRY) if entries == ["*"] else list(entries)

    for name in (*include, *exclude):
        if name not in REGISTRY:
            raise KeyError(
                f"unknown plot {name!r} in plots.include/exclude; "
                f"run 'mvy plot --list' to see registered names"
            )

    for name in include:
        if name not in names:
            names.append(name)

    excluded = set(exclude)
    return [name for name in names if name not in excluded]


def select(
    names: list[str],
    methods: tuple[str, ...],
    topology: Topology | None = None,
    diagnostics: tuple[str, ...] = (),
) -> tuple[list[PlotSpec], dict[str, str]]:
    """Filter requested figures down to those that can actually be drawn.

    Unregistered names are reported as skipped rather than raising, because
    a preset may legitimately name figures that a later package will add.
    Names given explicitly in a config are validated by
    :func:`resolve_preset` instead.

    Returns
    -------
    selected : list of PlotSpec
    skipped : dict
        Figure name to the reason it was skipped.
    """
    selected: list[PlotSpec] = []
    skipped: dict[str, str] = {}

    for name in names:
        spec = REGISTRY.get(name)
        if spec is None:
            skipped[name] = "not registered"
            continue

        applies, reason = spec.applies_to(methods, topology, diagnostics)
        if applies:
            selected.append(spec)
        else:
            skipped[name] = reason

    for name, reason in skipped.items():
        log.info("[skip] plot %r: %s", name, reason)

    return selected, skipped


def describe_registry() -> str:
    """Render the registry as a table, for ``mvy plot --list``."""
    if not REGISTRY:
        return "No plots registered."

    width = max(len(name) for name in REGISTRY)
    lines = [f"{'name'.ljust(width)}  {'space':<10} {'needs':<22} description"]
    lines.append("-" * (width + 60))

    for name in sorted(REGISTRY):
        spec = REGISTRY[name]
        needs = ", ".join(spec.needs) or "-"
        lines.append(
            f"{name.ljust(width)}  {spec.space.value:<10} {needs:<22} {spec.description}"
        )
    return "\n".join(lines)


def clear_registry() -> None:
    """Empty the registry. Test support only."""
    REGISTRY.clear()
