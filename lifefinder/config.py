import os

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_file = Path.home() / ".lifefinder" / ".env"
load_dotenv(dotenv_path=env_file)

# Root directory for lifefinder configurations and data
ROOT = Path.home() / ".lifefinder"

# Directory for storing artifacts like models and logs
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))

# Version
VERSION = "0.0.2"

# Data directories
DATA_DIR = Path(ARTIFACTS_DIR / "data")
RAW_DIR = Path(DATA_DIR / "raw")
PROCESSED_DIR = Path(DATA_DIR / "processed")

# Cache and model directories
CACHE_DIR = Path(ARTIFACTS_DIR / "cache")
MODELS_DIR = Path(ARTIFACTS_DIR / "models")
EVALUATION_DIR = Path(ARTIFACTS_DIR / "evaluation")

# Default cache filename
DEFAULT_EXOPLANETS_CSV = RAW_DIR / "exoplanets.csv"

# NASA TAP api endpoint
NASA_TAP_SYNC = os.getenv(
    "NASA_TAP_SYNC", "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
)

# NASA API rate limit
NASA_API_LIMIT = int(os.getenv("NASA_API_LIMIT", 100000))
FORCE_NASA_API_FETCH = os.getenv("FORCE_NASA_API_FETCH", "False").lower() in (
    "true",
    "1",
    "t",
)

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
    "hz_sigma": float(os.getenv("HABITABLE_ZONE_SIGMA", 1.0)),
    "hz_threshold": float(os.getenv("HABITABLE_ZONE_THRESHOLD", 0.5)),
}

TARGET_FEATURE = os.getenv("TARGET_FEATURE", "habitable_zone_index")

NUMERIC_FEATURES = [
    "sy_snum",
    "sy_pnum",
    "disc_year",
    "pl_orbper",
    "pl_orbsmax",
    "pl_rade",
    "pl_radj",
    "pl_bmasse",
    "pl_bmassj",
    "pl_dens",
    "pl_orbeccen",
    # "pl_insol", Removed to avoid data leakage as habitable_zone_index derived from it
    "pl_eqt",
    "st_teff",
    "st_rad",
    "st_mass",
    "st_met",
    "st_logg",
    "sy_dist",
]

ENGINEERED_NUMERIC_FEATURES = [
    "orbit_star_ratio",
    "planet_star_mass_ratio",
    "relative_radius_ratio",
    "log_pl_rade",
    "log_pl_bmasse",
    "log_orbit_star_ratio",
]

CATEGORICAL_FEATURES = [
    "host_name",
    "discoverymethod",
    "disc_facility",
    "st_spectype",
    "rastr",
    "decstr",
]
