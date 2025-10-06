import pytest
import importlib
import sys
import os
from unittest.mock import patch

from lifefinder import config as cfg


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment once per session."""
    # Store original environment variables
    original_env = {}

    # Override environment variables at the session level
    test_env_vars = {
        "ARTIFACTS_DIR": "/tmp/pytest_lifefinder_artifacts",
        "NASA_API_LIMIT": "100",
        "BATCH_SIZE": "32",
        "EPOCHS": "2",
        "LEARNING_RATE": "1e-3",
        "HIDDEN_DIM": "64",
        "DROPOUT": "0.3",
        "VAL_SPLIT": "0.2",
        "RANDOM_STATE": "42",
        "PATIENCE": "5",
        "HABITABLE_ZONE_SIGMA": "1.0",
        "HABITABLE_ZONE_THRESHOLD": "0.5",
        "TARGET_FEATURE": "habitable_zone_index",
    }

    # Store original values and set test values
    for key, value in test_env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    # Store original modules for cleanup
    original_modules = {}
    lifefinder_modules = [
        name for name in sys.modules.keys() if name.startswith("lifefinder")
    ]

    for module_name in lifefinder_modules:
        original_modules[module_name] = sys.modules[module_name]

    yield

    # Cleanup: restore original environment variables
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

    # Cleanup: restore original modules
    for module_name, module in original_modules.items():
        sys.modules[module_name] = module


@pytest.fixture(scope="function", autouse=True)
def setup_tmp_dirs_and_reload_config(tmp_path):
    """Reload config and redirect all paths to tmp folder for each test."""

    # Mock load_dotenv to prevent loading any .env file
    with patch("lifefinder.config.load_dotenv"):
        # Get all lifefinder modules currently in sys.modules
        lifefinder_modules = [
            name for name in sys.modules.keys() if name.startswith("lifefinder")
        ]

        # Reload config first
        importlib.reload(cfg)

        # Reload all other lifefinder modules in dependency order
        for module_name in sorted(lifefinder_modules):
            if module_name != "lifefinder.config" and module_name in sys.modules:
                try:
                    importlib.reload(sys.modules[module_name])
                except Exception:
                    # If reload fails, remove from sys.modules to force reimport
                    sys.modules.pop(module_name, None)

        # Override paths to use tmp directories (this overrides env vars)
        cfg.ROOT = tmp_path
        cfg.ARTIFACTS_DIR = tmp_path / "artifacts"
        cfg.DATA_DIR = cfg.ARTIFACTS_DIR / "data"
        cfg.RAW_DIR = cfg.DATA_DIR / "raw"
        cfg.PROCESSED_DIR = cfg.DATA_DIR / "processed"
        cfg.CACHE_DIR = cfg.ARTIFACTS_DIR / "cache"
        cfg.MODELS_DIR = cfg.ARTIFACTS_DIR / "models"
        cfg.EVALUATION_DIR = cfg.ARTIFACTS_DIR / "evaluation"
        cfg.DEFAULT_EXOPLANETS_CSV = cfg.RAW_DIR / "exoplanets.csv"

        # Create all necessary directories
        cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
        cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cfg.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

        yield tmp_path
