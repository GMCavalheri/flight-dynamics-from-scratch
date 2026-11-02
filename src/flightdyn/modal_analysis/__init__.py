"""Eigenvalues sorted into the classical flight modes, mode shapes and approximations."""

from . import approximations
from .modes import Mode, eigen, lateral_modes, longitudinal_modes, mode_shape

__all__ = ["Mode", "approximations", "eigen", "lateral_modes", "longitudinal_modes", "mode_shape"]
