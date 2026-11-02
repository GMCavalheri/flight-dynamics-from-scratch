"""Load the cited reference values in ``validation/reference_data`` (report units, as printed).

* ``b747(fc)`` — NASA CR-2144 Tables IX-3, IX-4, IX-5, IX-8, IX-9 for flight condition 1..10.
* ``b747_nondimensional(fc)`` — NASA CR-2144 Tables IX-1 (fc = 1) and IX-2 (fc = 2).
* ``navion()`` — Teper, STI TR-176-1 (1969), Tables X-A to X-G.

Values stay in the reports' imperial units here; ``reference_aircraft.py`` converts to SI.
"""

from __future__ import annotations

import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent / "reference_data"
B747_FLIGHT_CONDITIONS = tuple(range(1, 11))


def load(name: str) -> list[dict[str, str]]:
    with open(DATA / name) as f:
        rows = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(rows))


def b747(fc: int) -> dict[str, float]:
    """All CR-2144 Table IX-3/4/5/8/9 values of one flight condition, keyed by mnemonic."""
    if fc not in B747_FLIGHT_CONDITIONS:
        raise KeyError(f"B-747 flight condition {fc} (CR-2144 has 1..10)")
    return {r["quantity"]: float(r[f"fc{fc}"]) for r in load("b747_cr2144.csv")}


def b747_nondimensional(fc: int) -> dict[str, float]:
    """CR-2144 Table IX-1 (landing, fc 1) or IX-2 (power approach, fc 2) derivatives."""
    if fc not in (1, 2):
        raise KeyError("CR-2144 tabulates nondimensional 747 derivatives only for fc 1 and 2")
    return {r["quantity"]: float(r[f"fc{fc}"]) for r in load("b747_cr2144_nondimensional.csv")}


def navion() -> dict[str, float]:
    """Teper (1969) Navion data, keyed by the CSV quantity name."""
    return {r["quantity"]: float(r["value"]) for r in load("navion_teper1969.csv")}
