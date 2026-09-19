"""The steps a run is made of: build a model, evaluate a field, report.

Each step takes and returns an artefact, so any of them can be run alone
and the CLI is a thin dispatcher over them rather than the place where the
work lives.
"""

from __future__ import annotations
