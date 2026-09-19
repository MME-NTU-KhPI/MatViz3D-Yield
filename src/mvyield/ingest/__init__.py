"""Dataset analysis: HDF5 in, yield points and material properties out.

This is the only package that touches ``h5py``, so evaluating an existing
model on a new field never imports it. Keep it that way: the import belongs
inside this package, not at the top of a shared module.

The dataset is both the measurements and the material definition -- see
:mod:`mvyield.ingest.material` for why there is no separate material file.
"""

from __future__ import annotations

from mvyield.ingest.analysis import (
    SELECTION_RULES,
    EigenTracker,
    analyse_dataset,
    schmid_step,
    von_mises_step,
)
from mvyield.ingest.artefacts import IngestResult, StepTable
from mvyield.ingest.material import (
    KNOWN_STRUCTURES,
    MaterialProperties,
    read_material,
)
from mvyield.ingest.reader import (
    DATASET_CONVENTION,
    DatasetError,
    DatasetReader,
    Geometry,
    Step,
)
from mvyield.ingest.slip import euler_to_rotation, slip_systems

__all__ = [
    "DATASET_CONVENTION",
    "SELECTION_RULES",
    "DatasetError",
    "DatasetReader",
    "EigenTracker",
    "Geometry",
    "IngestResult",
    "KNOWN_STRUCTURES",
    "MaterialProperties",
    "Step",
    "StepTable",
    "analyse_dataset",
    "euler_to_rotation",
    "read_material",
    "schmid_step",
    "slip_systems",
    "von_mises_step",
]
