"""An aircraft as the flight-dynamics model sees it: mass, geometry, aerodynamics, thrust."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .aero_derivatives import Geometry, StabilityDerivatives


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


@dataclass(frozen=True)
class ThrustLine:
    """Thrust of magnitude T acting along a line inclined ``incidence`` [rad] nose-up from the
    body x axis, passing ``offset`` [m] below the centre of gravity (so thrust pitches the nose
    up when ``offset`` > 0). CR-2144 Appendix A: i_th and l_th. Thrust does not vary with
    speed in this model (T_u = 0).
    """

    incidence: float = 0.0
    offset: float = 0.0

    def force_moment(self, T: float) -> tuple[np.ndarray, np.ndarray]:
        ci, si = np.cos(self.incidence), np.sin(self.incidence)
        return T * np.array([ci, 0.0, -si]), np.array([0.0, T * self.offset, 0.0])


@dataclass(frozen=True)
class Aircraft:
    """Rigid aircraft: mass properties, reference geometry, derivative model and thrust line."""

    name: str
    mass: MassProperties
    geometry: Geometry
    aero: StabilityDerivatives
    thrust: ThrustLine = ThrustLine()

    def with_(self, **changes) -> Aircraft:
        return replace(self, **changes)

    @property
    def weight(self) -> float:
        from .units import G0

        return self.mass.mass * G0
