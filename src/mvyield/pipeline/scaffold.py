"""Writing a starter case file.

A new problem should begin from something that runs, not from a blank file
and the reference documentation. The templates below are minimal but
complete: every key they set is one a reader has to decide about, and every
key they leave out has a working default in ``configs/defaults.yaml``.

They are written as text rather than dumped from a dataclass so that the
comments survive -- which is most of what makes a config file readable.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mvyield.settings import ConfigError

log = logging.getLogger(__name__)

KIRSCH_TEMPLATE = """\
case: kirsch
name: {name}

# Input 1: the dataset, which is also the material definition. Crystal
# structure and CRSS are read from the file; material_override fills only
# what a particular file does not carry, in MPa.
dataset:
  source: data/datasets/CHANGE_ME.hdf5
  analysis: SCHMID              # SCHMID | VON_MISES
  select_step: max_macro_sz
  material_override: {{}}

# Input 2: the stress field. A hole in a plate under remote tension.
field:
  source: analytic
  sigma_applied: 150.0          # MPa
  hole_radius:
    auto: true
    fraction: 0.15              # of the dataset's coordinate box
    fallback: 1.0
  domain_from: dataset          # 'config' sizes the window in hole radii
  grid:
    resolution: 400

model:
  path: auto
  deviatoric: true
  methods: [kde_slice]
  primary: kde_slice
  kde_slice:
    mode: polar
    kappa: {{mode: auto, objective: cv_loglik, grid: [5, 10, 20, 50, 100]}}
    tau_floor: 2.0

probes:
  paths:
    - name: ligament
      kind: ray
      angle_deg: 90.0           # across the ligament, where yielding starts
      r_from: 1.0
      r_to: 4.0
      units: hole_radii
      n: 200

plots:
  preset: paper
"""

ANSYS_TEMPLATE = """\
case: ansys
name: {name}

# The dataset defines the material; drop this section to run against an
# already built model without opening it.
dataset:
  source: data/datasets/CHANGE_ME.hdf5
  analysis: SCHMID

# Input 2: a field exported from ANSYS.
#   /POST1
#   SET,LAST
#   PRNSOL,S,COMP        ! NODE, SX, SY, SZ, SXY, SYZ, SXZ
# PRNSOL writes no coordinates: add NLIST, or export from Mechanical
# (right-click the stress result -> Export -> Text File).
field:
  source: ansys
  path: data/ansys/CHANGE_ME.csv
  format: prnsol_csv
  convention: mandel_xy_first   # ANSYS shear order; the sqrt(2) is applied on import
  units:
    stress: Pa                  # converted to MPa on import
    length: m
  columns:
    node: NODE
    xyz: [X, Y, Z]
    stress: [SX, SY, SZ, SXY, SYZ, SXZ]
  frame: global                 # must match the frame the dataset was built in

# A 3D field needs a reduction before it can be drawn.
view:
  reductions:
    - {{kind: slice, plane: z, value: 0.0, tol: 1.0e-4, name: midspan}}

model:
  path: auto
  deviatoric: true
  methods: [kde_ray]
  primary: kde_ray
  kde_ray:
    localized: true
    power: 8.0

plots:
  preset: minimal

export:
  vtk: true                     # real 3D inspection belongs in ParaView
"""

TEMPLATES = {"kirsch": KIRSCH_TEMPLATE, "ansys": ANSYS_TEMPLATE}


def write_case(name: str, template: str, directory: Path) -> Path:
    """Write a starter case file and return its path.

    Refuses to overwrite: a case file is written by hand after this point,
    and losing that to a repeated command would be the tool's fault.
    """
    if template not in TEMPLATES:
        raise ConfigError(
            f"unknown template {template!r}; available: {sorted(TEMPLATES)}"
        )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}.yaml"

    if target.exists():
        raise ConfigError(
            f"{target} already exists; choose another name or delete it first"
        )

    target.write_text(TEMPLATES[template].format(name=name), encoding="utf-8")
    log.info("wrote %s", target)
    return target
