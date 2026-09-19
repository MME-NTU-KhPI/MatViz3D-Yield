"""Stress conventions and the field container.

The layer that knows what a stress vector *is*: which slot holds which
component, how shear is scaled, and how a set of stresses over a body is
carried around. It knows nothing about yield models or plotting, so both
can depend on it without depending on each other.
"""

from __future__ import annotations
