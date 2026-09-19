"""Data-driven yield probability from microstructure datasets.

Two inputs define a run: a dataset, which is also the material definition,
and a stress field. Everything between them is shared, which is what lets a
new problem be a config file rather than new code.

Nothing is imported here on purpose. ``mvy run`` must not pay for ``h5py``
or ``matplotlib`` just to read the version, so every subpackage is imported
at the point of use.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
