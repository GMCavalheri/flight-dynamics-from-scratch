"""Mass properties shared by every model of a rigid aircraft."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MassProperties:
    """Mass [kg] and body-axis moments of inertia [kg m^2] about the centre of gravity.

    The aircraft is symmetric about its x-z plane, so Ixy = Iyz = 0. ``Ixz`` is the product of
    inertia ``integral of x z dm`` (Etkin & Reid ch. 4; NASA CR-2144 Appendix A), which enters the
    inertia tensor with a minus sign.
    """

    mass: float
    Ixx: float
    Iyy: float
    Izz: float
    Ixz: float = 0.0

    @property
    def inertia(self) -> np.ndarray:
        return np.array(
            [[self.Ixx, 0.0, -self.Ixz], [0.0, self.Iyy, 0.0], [-self.Ixz, 0.0, self.Izz]]
        )

    @property
    def inertia_inv(self) -> np.ndarray:
        return np.linalg.inv(self.inertia)
