"""The Kirsch problem: a circular hole in a plate under remote tension.

The classical elasticity solution, in polar components about the hole::

    sigma_rr = s/2 (1 - xi) + s/2 (1 - 4 xi + 3 xi^2) cos 2t
    sigma_tt = s/2 (1 + xi) - s/2 (1 + 3 xi^2)        cos 2t
    tau_rt   =            - s/2 (1 + 2 xi - 3 xi^2)   sin 2t

with ``xi = (a/r)^2``. At the hole edge on the transverse axis this gives
``sigma_tt = 3 s``: the stress concentration factor of three that makes the
problem worth solving at all, because a remote load a third of the yield
strength already yields at the hole.

Scale
-----
Every stress depends on position only through ``r/a`` and ``theta``, so the
field is scale-free: doubling both the plate and the hole changes nothing.
The domain therefore fixes how much of the plate is drawn and nothing else,
which is why taking it from the dataset's coordinate extent -- as the legacy
code did -- is defensible despite the RVE's voxel grid having no physical
relation to a plate. It is a plotting window, not a geometry.

Plane stress
------------
The solution is plane stress: ``sigma_zz``, ``tau_yz`` and ``tau_xz`` are
zero. They are carried anyway, as zeros, because the model works in the full
six-component space and a plane problem is simply one that occupies part of
it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from mvyield.mechanics import voigt
from mvyield.mechanics.field import (
    FieldMeta,
    GeometryHints,
    Outline,
    StressField,
    Topology,
)
from mvyield.settings import ConfigError, FieldConfig

log = logging.getLogger(__name__)

DEFAULT_HALF_WIDTH_IN_RADII = 4.0
"""Plate half-width when nothing else fixes it, in hole radii.

Far enough that the remote field is recovered to under a per cent, and it
matches the reach of the ligament probe in the shipped case file.
"""

EDGE_MARGIN = 1e-9
"""Points closer than this to the hole edge count as inside it."""


@dataclass(frozen=True)
class Domain:
    """The plotting window and the hole it contains.

    Parameters
    ----------
    half_width : float
        Half the side of the square window, in the dataset's length unit.
    hole_radius : float
        Hole radius ``a``, same unit.
    source : str
        How these were chosen, for the run report.
    """

    half_width: float
    hole_radius: float
    source: str = "config"

    def __post_init__(self) -> None:
        """Check that the hole fits inside the window."""
        if self.hole_radius <= 0.0:
            raise ConfigError(f"hole radius must be positive, got {self.hole_radius}")
        if self.half_width <= self.hole_radius:
            raise ConfigError(
                f"the plotting window (half-width {self.half_width:g}) does not "
                f"contain the hole (radius {self.hole_radius:g}); raise "
                f"field.hole_radius.fraction or widen the domain"
            )

    @property
    def half_width_in_radii(self) -> float:
        """Window half-width measured in hole radii."""
        return self.half_width / self.hole_radius


def resolve_domain(config: FieldConfig, extent: float | None = None) -> Domain:
    """Decide the hole radius and the window from the config.

    Parameters
    ----------
    config : FieldConfig
        The ``field`` section.
    extent : float, optional
        Width of the dataset's coordinate box, required when
        ``domain_from`` is ``"dataset"``.

    Returns
    -------
    Domain
    """
    settings = dict(config.hole_radius or {})
    automatic = bool(settings.get("auto", True))
    fraction = float(settings.get("fraction", 0.15))
    fallback = float(settings.get("fallback", 1.0))

    if config.domain_from == "dataset":
        if extent is None:
            raise ConfigError(
                "field.domain_from is 'dataset', so the dataset's coordinate "
                "extent is needed to size the plate, but no dataset was "
                "opened. Set dataset.source, or use field.domain_from: "
                "'config' to size the window in hole radii instead."
            )
        radius = fraction * extent if automatic else fallback
        return Domain(half_width=0.51 * extent, hole_radius=radius, source="dataset extent")

    if config.domain_from != "config":
        raise ConfigError(
            f"field.domain_from must be 'dataset' or 'config'; "
            f"got {config.domain_from!r}"
        )

    radius = fallback
    half_width = float((config.grid or {}).get("half_width_in_radii",
                                               DEFAULT_HALF_WIDTH_IN_RADII)) * radius
    return Domain(half_width=half_width, hole_radius=radius, source="config")


def kirsch_polar(
    r: np.ndarray,
    theta: np.ndarray,
    hole_radius: float,
    sigma_applied: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polar stress components about the hole.

    Parameters
    ----------
    r, theta : ndarray
        Polar coordinates; ``r`` must be at or outside the hole.
    hole_radius : float
        ``a``.
    sigma_applied : float
        Remote tension along x, MPa.

    Returns
    -------
    sigma_rr, sigma_tt, tau_rt : ndarray
        MPa.
    """
    xi = (hole_radius / r) ** 2
    xi2 = xi**2
    cos2t, sin2t = np.cos(2.0 * theta), np.sin(2.0 * theta)
    half = 0.5 * sigma_applied

    sigma_rr = half * (1.0 - xi) + half * (1.0 - 4.0 * xi + 3.0 * xi2) * cos2t
    sigma_tt = half * (1.0 + xi) - half * (1.0 + 3.0 * xi2) * cos2t
    tau_rt = -half * (1.0 + 2.0 * xi - 3.0 * xi2) * sin2t
    return sigma_rr, sigma_tt, tau_rt


def polar_to_cartesian(
    sigma_rr: np.ndarray,
    sigma_tt: np.ndarray,
    tau_rt: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate polar components into the global x-y frame."""
    cos, sin = np.cos(theta), np.sin(theta)
    cc, ss, cs = cos**2, sin**2, cos * sin

    sigma_xx = sigma_rr * cc + sigma_tt * ss - 2.0 * tau_rt * cs
    sigma_yy = sigma_rr * ss + sigma_tt * cc + 2.0 * tau_rt * cs
    tau_xy = (sigma_rr - sigma_tt) * cs + tau_rt * (cc - ss)
    return sigma_xx, sigma_yy, tau_xy


def build_field(config: FieldConfig, extent: float | None = None) -> StressField:
    """Evaluate the Kirsch solution over a regular grid.

    Points inside the hole are kept in the arrays but marked invalid, so the
    grid stays rectangular -- which is what lets the field figures use
    ``imshow`` rather than a triangulation -- while nothing downstream
    evaluates a stress state that does not exist.

    Parameters
    ----------
    config : FieldConfig
        The ``field`` section.
    extent : float, optional
        Dataset coordinate extent, when ``domain_from`` is ``"dataset"``.

    Returns
    -------
    StressField
        A :attr:`~mvyield.mechanics.field.Topology.GRID2D` field.
    """
    domain = resolve_domain(config, extent)
    resolution = int((config.grid or {}).get("resolution", 400))
    if resolution < 2:
        raise ConfigError(f"field.grid.resolution must be at least 2, got {resolution}")

    axis = np.linspace(-domain.half_width, domain.half_width, resolution)
    grid_x, grid_y = np.meshgrid(axis, axis)
    x, y = grid_x.ravel(), grid_y.ravel()

    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)

    inside = radius < domain.hole_radius - EDGE_MARGIN
    safe_radius = np.where(inside, np.inf, radius)

    sigma_rr, sigma_tt, tau_rt = kirsch_polar(
        safe_radius, theta, domain.hole_radius, config.sigma_applied
    )
    sigma_xx, sigma_yy, tau_xy = polar_to_cartesian(sigma_rr, sigma_tt, tau_rt, theta)

    sigma6 = voigt.plane_stress(sigma_xx, sigma_yy, tau_xy, config.voigt_convention)
    sigma6[inside] = 0.0

    log.info(
        "Kirsch field: a = %.4g, window +-%.4g (%.1f a), %d x %d grid, "
        "sigma_0 = %.4g MPa, K_t sigma_0 = %.4g MPa",
        domain.hole_radius,
        domain.half_width,
        domain.half_width_in_radii,
        resolution,
        resolution,
        config.sigma_applied,
        3.0 * config.sigma_applied,
    )

    return StressField(
        points=np.column_stack([x, y, np.zeros_like(x)]),
        sigma6=sigma6,
        topology=Topology.GRID2D,
        meta=FieldMeta(
            source="analytic",
            convention=config.voigt_convention,
            stress_unit="MPa",
            length_unit=str((config.units or {}).get("length", "mm")),
            frame=config.frame,
            grid_shape=(resolution, resolution),
            extra={
                "case": "kirsch",
                "hole_radius": domain.hole_radius,
                "half_width": domain.half_width,
                "domain_source": domain.source,
                "sigma_applied": config.sigma_applied,
                "stress_concentration": 3.0,
            },
        ),
        valid=~inside,
        hints=GeometryHints(
            length_scale=domain.hole_radius,
            length_symbol="a",
            length_unit=str((config.units or {}).get("length", "mm")),
            outlines=[
                Outline(
                    kind="circle",
                    points=np.zeros((1, 2)),
                    radius=domain.hole_radius,
                )
            ],
        ),
    )
