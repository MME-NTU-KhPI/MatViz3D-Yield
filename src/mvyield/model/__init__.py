"""Yield models: the space they live in, the estimators, and the bundle.

Submodules are imported at the point of use rather than here, because the
kernel estimators pull in SciPy and a caller that only wants to read a
bundle's provenance should not pay for it.
"""

from __future__ import annotations
