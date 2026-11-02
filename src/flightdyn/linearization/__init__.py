"""Linear models about trim: numerical Jacobians and analytic small-perturbation equations."""

from .numerical import (
    LATERAL,
    LONGITUDINAL,
    LinearModel,
    coupling,
    jacobians,
    lateral,
    linearize,
    longitudinal,
)
from .small_perturbation import analytic_model, lateral_model, longitudinal_model, rereference

__all__ = [
    "analytic_model",
    "rereference",
    "LATERAL",
    "LONGITUDINAL",
    "LinearModel",
    "coupling",
    "jacobians",
    "lateral",
    "lateral_model",
    "linearize",
    "longitudinal",
    "longitudinal_model",
]
