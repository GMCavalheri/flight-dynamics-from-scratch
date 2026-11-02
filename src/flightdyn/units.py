"""Exact unit conversion factors (to SI). Internal quantities are always SI.

Source: NIST SP 811 (2008), Appendix B — the foot, pound-mass, pound-force and knot are defined
exactly in SI; the slug is the mass accelerated at 1 ft/s^2 by 1 lbf.
"""

from __future__ import annotations

import numpy as np

FT = 0.3048  # m per foot (exact)
LB = 0.45359237  # kg per pound-mass (exact)
G0 = 9.80665  # m/s^2, standard gravity (exact)
LBF = LB * G0  # N per pound-force (exact)
SLUG = LBF / FT  # kg per slug = 14.593902937...
KT = 1852.0 / 3600.0  # m/s per knot (exact)
SLUG_FT2 = SLUG * FT**2  # kg m^2 per slug ft^2
PSF = LBF / FT**2  # Pa per lbf/ft^2
SLUG_FT3 = SLUG / FT**3  # kg/m^3 per slug/ft^3
G0_FT = G0 / FT  # standard gravity in ft/s^2 = 32.1740...
DEG = np.pi / 180.0  # rad per degree
