"""The mapping between raw stress vectors and the space a model lives in.

Every data-driven estimator compares a query against a cloud of yield points.
That comparison is only meaningful if both went through exactly the same
preparation: the same component order, the same shear scaling, the same
choice about hydrostatic pressure, the same normalisation.

In the pre-refactor code these choices were spread across a module-level
config, two docstrings and a comment, and were re-applied independently at
build time and at query time. This class makes the mapping a single object
that is built once, stored inside the model bundle, and read back from the
bundle at evaluation time -- so the dataset space and the query space cannot
disagree, whatever the config says later.

The scale factor exists mainly for the neural surrogate: a network needs
inputs of order one, but the same scaling must then apply to queries. Keeping
it here rather than in a training script means it cannot be forgotten at
inference.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from mvyield.mechanics import voigt
from mvyield.mechanics.voigt import VoigtConvention

from . import deviator


@dataclass(frozen=True)
class SpaceTransform:
    """Raw stress vectors in, model coordinates out.

    Parameters
    ----------
    convention : VoigtConvention
        Convention of the model space. Inputs declaring a different
        convention are converted first.
    deviatoric : bool
        Whether to remove the hydrostatic part. ``True`` suits pressure-
        insensitive metals; set ``False`` for pressure-sensitive materials
        (Drucker-Prager-like behaviour, soils, polymers, porous media),
        where the hydrostatic axis carries real information and dropping it
        would discard it.
    scale : float
        Divisor applied after the deviatoric step, MPa. ``1.0`` leaves the
        data in MPa. Set from the point cloud via :meth:`fitted_scale`.
    scale_kind : str
        How ``scale`` was chosen, recorded for provenance: ``"none"`` or
        ``"rms_norm"``.
    """

    convention: VoigtConvention
    deviatoric: bool = True
    scale: float = 1.0
    scale_kind: str = "none"

    def forward(
        self,
        sigma6: np.ndarray,
        source: VoigtConvention | None = None,
    ) -> np.ndarray:
        """Map raw stress vectors into model space.

        Parameters
        ----------
        sigma6 : ndarray of shape (..., 6)
            Raw stress vectors, MPa.
        source : VoigtConvention, optional
            Convention of the input. Defaults to this transform's own
            convention. Pass it explicitly when ingesting an external source
            such as an ANSYS export, so the conversion happens here rather
            than in the caller.

        Returns
        -------
        ndarray
            Same shape as the input, in model coordinates.
        """
        arr = np.asarray(sigma6, dtype=np.float64)
        if source is not None and source != self.convention:
            arr = voigt.convert(arr, source, self.convention)
        if self.deviatoric:
            arr = deviator.to_deviatoric(arr)
        if self.scale != 1.0:
            arr = arr / self.scale
        return arr

    def inverse(self, sigma_model: np.ndarray) -> np.ndarray:
        """Map model coordinates back to MPa.

        Only the scaling is inverted. The deviatoric projection is not
        invertible -- the removed pressure is not recoverable from the
        projected vector -- so the result is the deviator in MPa, not the
        original stress state.
        """
        arr = np.asarray(sigma_model, dtype=np.float64)
        return arr * self.scale if self.scale != 1.0 else arr.copy()

    def fitted_scale(self, sigma6: np.ndarray) -> SpaceTransform:
        """Return a copy whose scale normalises the cloud to unit RMS radius.

        Parameters
        ----------
        sigma6 : ndarray of shape (N, 6)
            Raw yield points, MPa.

        Returns
        -------
        SpaceTransform
            New instance; the original is left unchanged.
        """
        unscaled = replace(self, scale=1.0, scale_kind="none")
        prepared = unscaled.forward(sigma6)
        rms = float(np.sqrt(np.mean(np.sum(prepared**2, axis=-1))))
        if rms < voigt.EPS_NORM:
            raise ValueError("cannot fit scale: the point cloud is degenerate")
        return replace(self, scale=rms, scale_kind="rms_norm")

    def describe(self) -> str:
        """One-line human-readable summary for logs and plot captions."""
        parts = [self.convention.name]
        parts.append("deviatoric" if self.deviatoric else "full (pressure kept)")
        if self.scale != 1.0:
            parts.append(f"scaled by {self.scale:.4g} MPa ({self.scale_kind})")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        """Serialise for storage in a model bundle."""
        return {
            "convention": self.convention.to_dict(),
            "deviatoric": self.deviatoric,
            "scale": self.scale,
            "scale_kind": self.scale_kind,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SpaceTransform:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            convention=VoigtConvention.from_dict(data["convention"]),
            deviatoric=bool(data["deviatoric"]),
            scale=float(data.get("scale", 1.0)),
            scale_kind=str(data.get("scale_kind", "none")),
        )
