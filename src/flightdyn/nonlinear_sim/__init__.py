"""Nonlinear simulation, control inputs and mode identification from time histories."""

from .identification import damped_pairs, hankel, matrix_pencil
from .inputs import CHANNELS, Doublet, Pulse, Step, breakpoints, control_increment
from .simulate import SimResult, simulate, simulate_linear

__all__ = [
    "CHANNELS",
    "Doublet",
    "Pulse",
    "SimResult",
    "Step",
    "breakpoints",
    "control_increment",
    "damped_pairs",
    "hankel",
    "matrix_pencil",
    "simulate",
    "simulate_linear",
]
