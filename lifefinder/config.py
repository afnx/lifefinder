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

NASA_API_LIMIT = int(os.getenv("NASA_API_LIMIT", 100000))

# Default cache filename
DEFAULT_EXOPLANETS_CSV = RAW_DIR / "exoplanets.csv"

# Training configuration
TRAINING_CONFIG = {
    "batch_size": int(os.getenv("BATCH_SIZE", 32)),
    "epochs": int(os.getenv("EPOCHS", 10)),
    "learning_rate": float(os.getenv("LEARNING_RATE", 1e-3)),
    "hidden_dim": int(os.getenv("HIDDEN_DIM", 64)),
    "dropout": float(os.getenv("DROPOUT", 0.3)),
    "val_split": float(os.getenv("VAL_SPLIT", 0.2)),
    "random_state": int(os.getenv("RANDOM_STATE", 42)),
}