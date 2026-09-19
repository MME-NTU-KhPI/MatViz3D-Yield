"""The figures themselves.

Importing this package registers every figure. Modules are grouped by what
they draw rather than by which mode draws them, matching the four spaces of
:mod:`mvyield.viz.registry`: a figure of the material is the same figure
whether a Kirsch plate or a wire prompted it.
"""

from __future__ import annotations

from mvyield.viz.plots import diagnostics, field, material, surface

__all__ = ["diagnostics", "field", "material", "surface"]
