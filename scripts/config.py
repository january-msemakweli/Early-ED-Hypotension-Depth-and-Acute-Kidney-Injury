"""Shared paths and constants for the ED hypotension depth analysis.

All scripts import from here so data locations and pre-specified thresholds
have a single source of truth. Nothing in this file runs analysis.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Prefer the PhysioNet layout used by the ventilation-liberation repository:
#   <workspace>/physionet.org/files/...
# with this analysis repository as a sibling of physionet.org. Also accept the
# same tree one level higher (e.g. when the repo lives on the Desktop next to
# a workspace folder that already contains PhysioNet downloads).
_DATA_CANDIDATES = [
    PROJECT_DIR.parent / "physionet.org" / "files",
    PROJECT_DIR.parent / "MIMIC IV" / "physionet.org" / "files",
    PROJECT_DIR.parent.parent / "physionet.org" / "files",
    PROJECT_DIR / "physionet.org" / "files",
]


def _resolve_data_root() -> Path:
    for root in _DATA_CANDIDATES:
        ed = root / "mimic-iv-ed" / "2.2" / "ed"
        hosp = root / "mimiciv" / "3.1" / "hosp"
        if ed.exists() and hosp.exists():
            return root
    return _DATA_CANDIDATES[0]


DATA_ROOT = _resolve_data_root()
ED_DIR = DATA_ROOT / "mimic-iv-ed" / "2.2" / "ed"
HOSP_DIR = DATA_ROOT / "mimiciv" / "3.1" / "hosp"
ICU_DIR = DATA_ROOT / "mimiciv" / "3.1" / "icu"

TABLES_DIR = PROJECT_DIR / "tables"
CACHE_DIR = TABLES_DIR / "cache"
FIGURES_DIR = PROJECT_DIR / "figures"

for _d in (TABLES_DIR, CACHE_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def require_mimic() -> None:
    """Raise a clear error if the PhysioNet MIMIC tables are not where expected."""
    missing = [p for p in (ED_DIR, HOSP_DIR, ICU_DIR) if not p.exists()]
    if missing:
        tried = "\n".join(f"  - {c}" for c in _DATA_CANDIDATES)
        raise FileNotFoundError(
            "MIMIC-IV / MIMIC-IV-ED tables not found.\n"
            f"Looked under:\n{tried}\n"
            "Expected subfolders:\n"
            "  mimic-iv-ed/2.2/ed/\n"
            "  mimiciv/3.1/hosp/\n"
            "  mimiciv/3.1/icu/\n"
            "Obtain the data from PhysioNet and place the gzipped tables under "
            "one of those roots (or edit DATA_ROOT in config.py)."
        )


# ---------------------------------------------------------------------------
# Pre-specified analytic constants
# ---------------------------------------------------------------------------
MAP_THRESHOLD = 65.0
MAP_PROFOUND = 55.0
SUSTAIN_MIN = 15.0
MAX_GAP_MIN = 120.0
BURDEN_WINDOW_H = 2.0
AKI_RATIO = 2.0
HORIZON_DAYS = 7.0
N_BOOT = 500
N_MICE = 5
RANDOM_SEED = 20260726
