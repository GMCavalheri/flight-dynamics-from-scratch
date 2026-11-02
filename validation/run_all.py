"""Regenerate every validation figure and results table.

Run with ``uv run python validation/run_all.py``. The optional AERO-01 cross-check
(``vlm_navion.py``) runs only if the ``aero`` package is installed; see its docstring.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "derivative_validation.py",
    "mode_validation.py",
    "simulation_validation.py",
    "trade_studies.py",
    "mode_animations.py",
    "vlm_navion.py",
]

if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    for name in SCRIPTS:
        print(f"== {name}")
        try:
            runpy.run_path(str(HERE / name), run_name="__main__")
        except SystemExit as exit_:  # vlm_navion.py exits with its status
            if exit_.code:
                raise
