"""Project paths, resolved from the repository root.

Everything that reads or writes files imports from here, so no module hard-codes a path
and the pipeline behaves the same no matter which directory it is launched from.
"""

from pathlib import Path

# src/fairprice/paths.py -> parents[2] is the repository root.
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42


def ensure_dirs() -> None:
    """Create the git-ignored output folders, which do not exist on a fresh clone."""
    for directory in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
