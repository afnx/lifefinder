import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("LIFEFINDER_DATA", ROOT / "data"))
RAW_DIR = Path(os.getenv("LIFEFINDER_RAW", DATA_DIR / "raw"))
CACHE_DIR = Path(os.getenv("LIFEFINDER_CACHE", ROOT / "cache"))

# NASA TAP api endpoint
NASA_TAP_SYNC = os.getenv(
    "NASA_TAP_SYNC",
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
)

# default cache filename
DEFAULT_EXOPLANETS_CSV = RAW_DIR / "exoplanets.csv"
