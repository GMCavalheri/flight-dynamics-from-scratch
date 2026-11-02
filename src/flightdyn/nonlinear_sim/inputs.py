"""Control inputs: increments added to the trim controls, piecewise constant in time.

Each input knows its breakpoints (the times where it jumps), so the integrator can stop and
restart there instead of stepping blindly across a discontinuity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHANNELS = {"de": 0, "da": 1, "dr": 2, "thrust": 3}


@dataclass(frozen=True)
class Step:
    """``amplitude`` added to ``channel`` from ``start`` on."""

    channel: str
    amplitude: float
    start: float = 0.0

    def value(self, t: float) -> float:
        return self.amplitude if t >= self.start else 0.0

    @property
    def breakpoints(self) -> tuple[float, ...]:
        return (self.start,)


@dataclass(frozen=True)
class Pulse:
    """``amplitude`` on ``channel`` for ``width`` seconds from ``start``."""

    channel: str
    amplitude: float
    start: float
    width: float

    def value(self, t: float) -> float:
        return self.amplitude if self.start <= t < self.start + self.width else 0.0

    @property
    def breakpoints(self) -> tuple[float, ...]:
        return (self.start, self.start + self.width)


@dataclass(frozen=True)
class Doublet:
    """+amplitude for ``width`` seconds, then -amplitude for ``width`` seconds: a zero-mean
    input whose spectrum, 4 a sin^2(w T / 2) / w with T = width, peaks at w = 2.33 / T and
    vanishes at w = 0, so it excites the faster modes without leaving a trim offset."""

    channel: str
    amplitude: float
    start: float
    width: float

    def value(self, t: float) -> float:
        if self.start <= t < self.start + self.width:
            return self.amplitude
        if self.start + self.width <= t < self.start + 2 * self.width:
            return -self.amplitude
        return 0.0

    @property
    def breakpoints(self) -> tuple[float, ...]:
        return (self.start, self.start + self.width, self.start + 2 * self.width)


def control_increment(inputs, t: float) -> np.ndarray:
    """Sum of the inputs' increments at time t, as a (de, da, dr, thrust) vector."""
    du = np.zeros(4)
    for inp in inputs:
        du[CHANNELS[inp.channel]] += inp.value(t)
    return du


def breakpoints(inputs, t_end: float) -> list[float]:
    pts = sorted({0.0, t_end, *(b for i in inputs for b in i.breakpoints if 0.0 < b < t_end)})
    return pts
