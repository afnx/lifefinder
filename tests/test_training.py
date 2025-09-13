import os
import pandas as pd
import torch
import pytest
import numpy as np
from unittest.mock import patch

from lifefinder import config as cfg
from scripts.train import train
from lifefinder.models.trainer import Trainer
from lifefinder.models.pytorch_classifier import ExoplanetNN
from lifefinder.data.preprocessor import build_exoplanet_pipeline


@pytest.fixture(scope="session", autouse=True)
def setup_tmp_dirs(tmp_path_factory):
    """Redirect all config paths to a tmp folder so tests don't overwrite real files."""
    tmpdir = tmp_path_factory.mktemp("lifefinder_tests")
    cfg.ROOT = tmpdir
    cfg.DATA_DIR = tmpdir / "data"
    cfg.RAW_DIR = cfg.DATA_DIR / "raw"
    cfg.PROCESSED_DIR = cfg.DATA_DIR / "processed"
    cfg.CACHE_DIR = tmpdir / "cache"
    cfg.MODELS_DIR = tmpdir / "models"
    cfg.MODEL_CHECKPOINT = cfg.MODELS_DIR / "exoplanet_model.pt"
    cfg.PIPELINE_PATH = cfg.MODELS_DIR / "exoplanet_pipeline.pkl"
    cfg.TRAINING_LOG = cfg.CACHE_DIR / "training_log.json"

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    return tmpdir

@pytest.fixture
def dummy_exoplanet_df():
    """Provide a dummy exoplanet-like DataFrame with at least 100 randomly generated rows."""
    n = 100
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "pl_name": [f"Planet{i}" for i in range(n)],
        "pl_orbper": rng.uniform(50, 1000, n),
        "pl_rade": rng.uniform(0.5, 15.0, n),
        "pl_bmasse": rng.uniform(0.1, 10.0, n),
        "st_teff": rng.uniform(3000, 7000, n),
        "st_rad": rng.uniform(0.5, 2.0, n),
        "st_mass": rng.uniform(0.1, 2.0, n),
        "st_metfe": rng.uniform(-0.5, 0.5, n),
        "pl_insol": rng.uniform(0.1, 100.0, n),
    })

@patch("scripts.train.NasaExoplanetClient")
def test_training_runs(mock_client, dummy_exoplanet_df):
    # Mock the NASA client to return dummy data
    mock_instance = mock_client.return_value
    mock_instance.fetch_exoplanets.return_value = dummy_exoplanet_df

    # Override training config for speed
    cfg.TRAINING_CONFIG.update({
        "epochs": 2,
        "batch_size": 2,
        "hidden_dim": 8,
        "patience": 2,
        "dropout": 0.1,
        "val_split": 0.5,
        "random_state": 42,
        "learning_rate": 1e-3,
    })
    cfg.NASA_API_LIMIT = 2
    cfg.FORCE_NASA_API_FETCH = True

    result = train(device="cpu")

    assert result["model"] is not None
    assert isinstance(result["trainer"], Trainer)
    assert len(result["metrics_log"]) > 0
    assert result["best_f1"] >= 0.0
    assert os.path.exists(cfg.MODEL_CHECKPOINT)
    assert os.path.exists(cfg.PIPELINE_PATH)

@pytest.mark.parametrize("input_data", [
    pd.DataFrame({
        "pl_name": ["PlanetA"],
        "pl_orbper": [365.0],
        "pl_rade": [1.0],
        "st_teff": [5778],
        "st_mass": [1.0],
        "pl_insol": [1.0],
    }),
    pd.DataFrame({
        "pl_name": ["PlanetB"],
        "pl_orbper": [200.0],
        "pl_rade": [2.0],
        "st_teff": [5000],
        "st_mass": [0.9],
        "pl_insol": [5.0],
    }),
])
def test_predict_pipeline(input_data):
    # Test that the pipeline and model can process input data and produce probabilities
    pipeline = build_exoplanet_pipeline()
    X = pipeline.fit_transform(input_data)

    model = ExoplanetNN(
        input_dim=X.shape[1],
        hidden_dim=cfg.TRAINING_CONFIG.get("hidden_dim", 8),
        dropout=cfg.TRAINING_CONFIG.get("dropout", 0.1),
    )

    trainer = Trainer(model)
    # Model checkpoint may not exist if not trained, so skip loading if missing
    if os.path.exists(cfg.MODEL_CHECKPOINT):
        try:
            trainer.load_checkpoint(cfg.MODEL_CHECKPOINT)
        except RuntimeError as e:
            # Skip loading if shape mismatch
            print(f"Skipping checkpoint load due to shape mismatch: {e}")

    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X, dtype=torch.float32)).detach().cpu().numpy()
    
    probs = probs.squeeze()
    if probs.ndim == 0:
        probs = probs[None]  # make it 1D
    assert probs.shape[0] == input_data.shape[0]
    assert (probs >= 0).all() and (probs <= 1).all()

def test_training_with_empty_data():
    with patch("scripts.train.NasaExoplanetClient") as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.fetch_exoplanets.return_value = pd.DataFrame()

        cfg.NASA_API_LIMIT = 0
        cfg.FORCE_NASA_API_FETCH = True

        result = train(device="cpu")
        assert result["model"] is None
        assert result["metrics_log"] == []
