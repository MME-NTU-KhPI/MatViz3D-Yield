"""Problems: what turns a case config into a stress field.

A case supplies input 2 -- the stress field -- and nothing else. The
analytical Kirsch solution and an imported ANSYS export are two cases over
the same everything else: same dataset, same model, same figures. That is
the whole point of the split, and it is why a new problem is a config file
rather than a new class hierarchy.

A builder takes the ``field`` section and returns a
:class:`~mvyield.mechanics.field.StressField`. It does not know about
models, probabilities or plots, so a case can be built and inspected before
any of those exist.
"""

from __future__ import annotations

from collections.abc import Callable

from mvyield.cases import ansys, kirsch
from mvyield.mechanics.field import StressField
from mvyield.settings import ConfigError, FieldConfig

BUILDERS: dict[str, Callable[[FieldConfig, float | None], StressField]] = {
    "kirsch": kirsch.build_field,
    "ansys": ansys.build_field,
}
"""Case name to the builder that produces its field."""


def build_field(
    case: str,
    config: FieldConfig,
    extent: float | None = None,
) -> StressField:
    """Build the stress field of one case.

    Parameters
    ----------
    case : str
        Case name, from the config's ``case:`` key.
    config : FieldConfig
        The ``field`` section.
    extent : float, optional
        Width of the dataset's coordinate box. Only the analytical case uses
        it, and only when ``field.domain_from`` is ``"dataset"``.

    Returns
    -------
    StressField
    """
    try:
        builder = BUILDERS[case]
    except KeyError:
        raise ConfigError(
            f"unknown case {case!r}; available: {sorted(BUILDERS)}"
        ) from None
    return builder(config, extent)


def available_cases() -> tuple[str, ...]:
    """Case names that can be built."""
    return tuple(sorted(BUILDERS))


__all__ = ["BUILDERS", "ansys", "available_cases", "build_field", "kirsch"]
