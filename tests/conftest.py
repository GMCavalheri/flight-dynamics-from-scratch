import sys
from pathlib import Path

# tests check the model against the cited reference data and aircraft in validation/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
