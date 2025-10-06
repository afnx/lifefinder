import os
import re
import pytest
import importlib

import pandas as pd
import numpy as np

from unittest.mock import patch

from lifefinder.train import train
from lifefinder import config as cfg
from lifefinder.models.trainer import Trainer


@pytest.fixture(scope="function", autouse=True)
def setup_tmp_dirs_and_reload_config(tmp_path):
    """Reload config and redirect all paths to tmp folder for each test."""
    # Reload config to get fresh state
    importlib.reload(cfg)

    # Set up temporary directories
    cfg.ROOT = tmp_path
    cfg.DATA_DIR = tmp_path / "data"
    cfg.RAW_DIR = cfg.DATA_DIR / "raw"
    cfg.PROCESSED_DIR = cfg.DATA_DIR / "processed"
    cfg.CACHE_DIR = tmp_path / "cache"
    cfg.MODELS_DIR = tmp_path / "models"
    cfg.MODEL_CHECKPOINT = cfg.MODELS_DIR / "exoplanet_model.pt"
    cfg.PIPELINE_PATH = cfg.MODELS_DIR / "exoplanet_pipeline.pkl"
    cfg.TRAINING_LOG = cfg.CACHE_DIR / "training_log.json"

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    return tmp_path


@pytest.fixture
def dummy_exoplanet_df():
    """Provide a dummy exoplanet-like DataFrame with at least 100 randomly generated rows."""
    n = 100
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "pl_name": [f"Planet{i}" for i in range(n)],
            "pl_orbper": rng.uniform(50, 1000, n),
            "pl_rade": rng.uniform(0.5, 15.0, n),
            "pl_bmasse": rng.uniform(0.1, 10.0, n),
            "st_teff": rng.uniform(3000, 7000, n),
            "st_rad": rng.uniform(0.5, 2.0, n),
            "st_mass": rng.uniform(0.1, 2.0, n),
            "st_metfe": rng.uniform(-0.5, 0.5, n),
            "pl_insol": rng.uniform(0.1, 100.0, n),
        }
    )


@patch("lifefinder.train.NasaExoplanetClient")
def test_training_runs(mock_client, dummy_exoplanet_df):
    # Mock the NASA client to return dummy data
    mock_instance = mock_client.return_value
    mock_instance.fetch_exoplanets.return_value = dummy_exoplanet_df

    # Override training config for speed
    cfg.TRAINING_CONFIG.update(
        {
            "batch_size": 2,
            "epochs": 2,
            "learning_rate": 1e-3,
            "hidden_dim": 8,
            "dropout": 0.1,
            "val_split": 0.5,
            "random_state": 42,
            "patience": 2,
            "hz_sigma": 10.0,
            "hz_threshold": 0.5,
        }
    )
    cfg.NASA_API_LIMIT = 2
    cfg.FORCE_NASA_API_FETCH = True

    result = train(device="cpu")

    assert result["model"] is not None
    assert isinstance(result["trainer"], Trainer)
    assert "best_f1" in result
    assert result["best_f1"] >= 0.0

    model_files = os.listdir(cfg.MODELS_DIR)

    # Find all model and pipeline files with their version string
    model_versions = [
        re.match(r"model_(f1-\d+\.\d{3}_\d{8}-\d{6})\.pt", f) for f in model_files
    ]
    model_versions = [m.group(1) for m in model_versions if m]

    # Only require model files if best_f1 > 0
    if result["best_f1"] > 0:
        assert model_versions, "No model files found"

        # Use the latest version string for retraining
        version = sorted(model_versions)[-1]
        model_file = f"model_{version}.pt"
        pipeline_file = f"pipeline_{version}.pkl"
        metrics_file = f"metrics_{version}.json"

        model_path = os.path.join(cfg.MODELS_DIR, model_file)
        pipeline_path = os.path.join(cfg.MODELS_DIR, pipeline_file)
        metrics_path = os.path.join(cfg.MODELS_DIR, metrics_file)

        assert os.path.exists(model_path), f"Model file not found: {model_path}"
        assert os.path.exists(pipeline_path), (
            f"Pipeline file not found: {pipeline_path}"
        )
        assert os.path.exists(metrics_path), f"Metrics file not found: {metrics_path}"
    else:
        # If best_f1 is 0, it's expected that no model files are saved
        assert not model_versions, "Model files should not be saved when F1 is 0"


@patch("lifefinder.train.NasaExoplanetClient")
def test_training_retrain(mock_client, dummy_exoplanet_df):
    # Mock the NASA client to return dummy data
    mock_instance = mock_client.return_value
    mock_instance.fetch_exoplanets.return_value = dummy_exoplanet_df

    # Override training config for speed and reproducibility
    training_config = {
        "batch_size": 2,
        "epochs": 2,
        "learning_rate": 1e-3,
        "hidden_dim": 8,
        "dropout": 0.1,
        "val_split": 0.5,
        "random_state": 42,
        "patience": 2,
        "hz_sigma": 10.0,
        "hz_threshold": 0.5,
    }

    cfg.TRAINING_CONFIG.update(training_config)
    cfg.NASA_API_LIMIT = 2
    cfg.FORCE_NASA_API_FETCH = True

    # First train to create initial model and pipeline files
    result = train(device="cpu")
    model_files = os.listdir(cfg.MODELS_DIR)

    # Find all model and pipeline files with their version string
    model_versions = [
        re.match(r"model_(f1-\d+\.\d{3}_\d{8}-\d{6})\.pt", f) for f in model_files
    ]
    model_versions = [m.group(1) for m in model_versions if m]

    if result["best_f1"] > 0:
        assert model_versions, "No model files found"

    # Use the latest version string for retraining
    version = sorted(model_versions)[-1]
    model_file = f"model_{version}.pt"
    pipeline_file = f"pipeline_{version}.pkl"

    model_path = os.path.join(cfg.MODELS_DIR, model_file)
    pipeline_path = os.path.join(cfg.MODELS_DIR, pipeline_file)

    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    assert os.path.exists(pipeline_path), f"Pipeline file not found: {pipeline_path}"

    # Ensure same config for retraining to avoid architecture mismatch
    cfg.TRAINING_CONFIG.update(training_config)

    # Retrain using the existing model and pipeline files
    result_retrain = train(
        retrain_pipeline_file=pipeline_path,
        retrain_model_file=model_path,
        device="cpu",
    )

    # Check retrain results
    assert result_retrain["model"] is not None
    assert isinstance(result_retrain["trainer"], Trainer)
    assert result_retrain["best_f1"] >= 0.0

    # Only require model files if best_f1 > 0
    if result_retrain["best_f1"] > 0:
        retrain_files = os.listdir(cfg.MODELS_DIR)
        model_pattern = re.compile(r"^model_f1-.*.pt$")
        pipeline_pattern = re.compile(r"^pipeline_f1-.*.pkl$")
        metrics_pattern = re.compile(r"^metrics_f1-.*.json$")
        assert any(model_pattern.match(f) for f in retrain_files)
        assert any(pipeline_pattern.match(f) for f in retrain_files)
        assert any(metrics_pattern.match(f) for f in retrain_files)
    else:
        # If best_f1 is 0, it's expected that no model files are saved
        retrain_files = os.listdir(cfg.MODELS_DIR)
        model_pattern = re.compile(r"^model_f1-.*.pt$")
        assert not any(model_pattern.match(f) for f in retrain_files)


def test_training_with_empty_data():
    with patch("lifefinder.train.NasaExoplanetClient") as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.fetch_exoplanets.return_value = pd.DataFrame()

        cfg.NASA_API_LIMIT = 0
        cfg.FORCE_NASA_API_FETCH = True

        with pytest.raises(ValueError):
            train(device="cpu")
