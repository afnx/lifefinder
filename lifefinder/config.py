import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = Path(os.getenv("LIFEFINDER_DATA", ROOT / "data"))
RAW_DIR = Path(os.getenv("LIFEFINDER_RAW", DATA_DIR / "raw"))
PROCESSED_DIR = Path(os.getenv("LIFEFINDER_PROCESSED", DATA_DIR / "processed"))

# Cache and model directories
CACHE_DIR = Path(os.getenv("LIFEFINDER_CACHE", ROOT / "cache"))
MODELS_DIR = Path(os.getenv("LIFEFINDER_MODELS", ROOT / "models"))

# Model checkpoint path
MODEL_CHECKPOINT = MODELS_DIR / "exoplanet_model.pt"
PIPELINE_PATH = MODELS_DIR / "exoplanet_pipeline.pkl"

# Training log file
TRAINING_LOG = CACHE_DIR / "training_log.json"

# Default cache filename
DEFAULT_EXOPLANETS_CSV = RAW_DIR / "exoplanets.csv"

# NASA TAP api endpoint
NASA_TAP_SYNC = os.getenv(
    "NASA_TAP_SYNC",
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
)

# NASA API rate limit
NASA_API_LIMIT = int(os.getenv("NASA_API_LIMIT", 100000))
FORCE_NASA_API_FETCH = os.getenv("FORCE_NASA_API_FETCH", "False").lower() in ("true", "1", "t")

# Training configuration
TRAINING_CONFIG = {
    "batch_size": int(os.getenv("BATCH_SIZE", 32)),
    "epochs": int(os.getenv("EPOCHS", 10)),
    "learning_rate": float(os.getenv("LEARNING_RATE", 1e-3)),
    "hidden_dim": int(os.getenv("HIDDEN_DIM", 64)),
    "dropout": float(os.getenv("DROPOUT", 0.3)),
    "val_split": float(os.getenv("VAL_SPLIT", 0.2)),
    "random_state": int(os.getenv("RANDOM_STATE", 42)),
    "patience": int(os.getenv("PATIENCE", 5)),
    "hz_sigma": float(os.getenv("HABITABLE_ZONE_SIGMA", 100.0)),
    "hz_threshold": float(os.getenv("HABITABLE_ZONE_THRESHOLD", 0.5)),
}