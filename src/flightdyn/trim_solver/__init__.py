"""Trim of the nonlinear aircraft model: steady level, climbing and turning flight."""

from .trim import TrimResult, pitch_attitude, trim, trim_guess, trim_state

__all__ = ["TrimResult", "pitch_attitude", "trim", "trim_guess", "trim_state"]
