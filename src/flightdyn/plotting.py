"""Shared matplotlib styling and plot helpers for the validation and lesson figures.

Series colours follow a fixed categorical order (never cycled or re-assigned by rank), lines
are thin, and grids/axes are recessive, so every figure in the repo reads as one set.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#0b0b0b"  # closed-form / published reference curves are drawn in ink, dashed

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_MUTED,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": TEXT,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "font.size": 10,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "svg.hashsalt": "flightdyn",
    "path.simplify": True,
}

FIGURES = Path(__file__).resolve().parents[2] / "docs" / "figures"


def use_style() -> None:
    plt.rcParams.update(STYLE)


def save(fig, name: str, directory: Path = FIGURES) -> Path:
    """Save a figure as PNG without a timestamp, so reruns are byte-identical."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    return path


# --- aircraft glyphs for mode animations -------------------------------------------------
# Outlines in units of fuselage length, nose at +x. Schematic, not any particular aircraft.
_SIDE = [(0.50, 0.00), (0.45, 0.05), (0.32, 0.08), (-0.34, 0.07), (-0.44, 0.24), (-0.53, 0.24),
         (-0.53, 0.06), (-0.56, 0.04), (-0.52, 0.00), (-0.30, -0.05), (0.38, -0.06),
         (0.47, -0.03)]  # fmt: skip
_TOP = [(0.50, 0.00), (0.40, 0.04), (0.10, 0.05), (-0.10, 0.48), (-0.20, 0.48), (-0.12, 0.05),
        (-0.40, 0.04), (-0.52, 0.18), (-0.56, 0.18), (-0.52, 0.00), (-0.56, -0.18),
        (-0.52, -0.18), (-0.40, -0.04), (-0.12, -0.05), (-0.20, -0.48), (-0.10, -0.48),
        (0.10, -0.05), (0.40, -0.04)]  # fmt: skip
_REAR = [(-0.50, 0.02), (-0.06, 0.04), (-0.05, 0.22), (0.00, 0.30), (0.05, 0.22), (0.06, 0.04),
         (0.50, 0.02), (0.50, -0.01), (0.06, -0.03), (0.00, -0.06), (-0.06, -0.03),
         (-0.50, -0.01)]  # fmt: skip


def _place(outline, x, y, angle, scale):
    pts = np.array(outline) * scale
    c, s = np.cos(angle), np.sin(angle)
    return pts @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def aircraft_side(x, y, pitch, scale=1.0):
    """Side-view outline polygon, nose-up pitch angle [rad] counter-clockwise."""
    return _place(_SIDE, x, y, pitch, scale)


def aircraft_top(x, y, heading, scale=1.0):
    """Top-view outline, heading [rad] measured counter-clockwise from +x on the page."""
    return _place(_TOP, x, y, heading, scale)


def aircraft_rear(roll, scale=1.0):
    """Rear-view outline (looking forward); right-wing-down roll is clockwise on the page."""
    return _place(_REAR, 0.0, 0.0, -roll, scale)


def save_gif(anim, name: str, fps: int = 15, directory: Path = FIGURES) -> Path:
    from matplotlib.animation import PillowWriter

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(anim._fig)
    return path
