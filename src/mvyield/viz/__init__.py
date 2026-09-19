"""Figures: the registry, the shared drawing infrastructure, and the plots.

Importing this package is what makes figures exist. Every plotting module
registers itself on import, so :func:`describe_registry` and
:func:`~mvyield.viz.registry.select` see the full set only once this
package has been imported -- which is also when Matplotlib is pulled in,
and why ``mvy run`` defers that import until it actually draws something.
"""

from __future__ import annotations

from mvyield.viz.registry import (
    REGISTRY,
    PlotSpec,
    Space,
    describe_registry,
    plot,
    resolve_preset,
    select,
)

# Importing the plots package is what fills the registry. It comes last so
# that the registry API above is already defined when the figures import it.
from mvyield.viz import plots  # noqa: E402,F401  (import for side effect)

__all__ = [
    "REGISTRY",
    "PlotSpec",
    "Space",
    "describe_registry",
    "plot",
    "resolve_preset",
    "select",
]
