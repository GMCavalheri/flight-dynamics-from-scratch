"""Shared matplotlib styling and plot helpers for the validation and lesson figures.

Series colours follow a fixed categorical order (never cycled or re-assigned by rank), lines
are thin, and grids/axes are recessive, so every figure in the repo reads as one set.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

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
