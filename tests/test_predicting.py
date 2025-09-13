import os
import pandas as pd
import pytest
import torch
from unittest.mock import patch, MagicMock
from lifefinder import config as cfg
from scripts.predict import predict

@pytest.fixture
def sample_input_csv(tmp_path):
    # Create a minimal fake exoplanet dataset
    csv_file = tmp_path / "input.csv"
    df = pd.DataFrame({
        "pl_name": ["TestPlanet"],
        "pl_orbper": [365.0],
        "pl_rade": [1.0],
        "pl_bmasse": [1.0],
        "st_teff": [5778],
        "st_rad": [1.0],
        "st_mass": [1.0],
        "st_metfe": [0.0],
    })
    df.to_csv(csv_file, index=False)
    return csv_file

def test_predict_runs_with_checkpoint(sample_input_csv):
    # Skip if model checkpoint is missing
    if not cfg.MODEL_CHECKPOINT.exists():
        pytest.skip("No trained model checkpoint available")

    df = predict(str(sample_input_csv))
    assert "habitability_prob" in df.columns
    assert df["habitability_prob"].between(0, 1).all()

@pytest.mark.parametrize("missing_artifact", ["pipeline", "model"])
def test_predict_missing_artifacts(sample_input_csv, monkeypatch, missing_artifact):
    def fake_exists(path) -> bool:
        if missing_artifact == "pipeline" and str(cfg.PIPELINE_PATH) in str(path):
            return False
        if missing_artifact == "model" and str(cfg.MODEL_CHECKPOINT) in str(path):
            return False
        return True

    monkeypatch.setattr(os.path, "exists", fake_exists)

    with patch("joblib.load", return_value=MagicMock(transform=lambda df: df.values)):
        result_df = predict(str(sample_input_csv))
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty

@pytest.mark.parametrize("input_df", [
    pd.DataFrame(),  # Empty DataFrame
    pd.DataFrame({"pl_name": ["A"], "pl_orbper": [None]}),  # Bad input
])
def test_predict_with_bad_input(tmp_path, input_df, monkeypatch):
    csv_file = tmp_path / "bad_input.csv"
    input_df.to_csv(csv_file, index=False)

    # Mock os.path.exists to always return True for model and pipeline
    monkeypatch.setattr(os.path, "exists", lambda path: True)

    with patch("joblib.load", return_value=MagicMock(transform=lambda df: df.values)), \
         patch("lifefinder.models.pytorch_classifier.ExoplanetNN") as mock_nn, \
         patch("lifefinder.models.trainer.Trainer") as mock_trainer:

        # Make the mock model callable directly
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.forward.return_value = torch.tensor([0.5] * len(input_df), dtype=torch.float32)
        mock_model.__call__ = mock_model.forward  # ensure calling model() works
        mock_nn.return_value = mock_model
        mock_trainer.return_value = MagicMock(load_checkpoint=lambda x: None)

        df = predict(str(csv_file))
        assert isinstance(df, pd.DataFrame)

def test_predict_multiple_rows(tmp_path, monkeypatch):
    # Create a CSV with multiple exoplanets
    csv_file = tmp_path / "multi_input.csv"
    df = pd.DataFrame({
        "pl_name": ["A", "B"],
        "pl_orbper": [365.0, 200.0],
        "pl_rade": [1.0, 2.0],
        "pl_bmasse": [1.0, 2.0],
        "st_teff": [5778, 5000],
        "st_rad": [1.0, 0.8],
        "st_mass": [1.0, 0.9],
        "st_metfe": [0.0, 0.1],
    })
    df.to_csv(csv_file, index=False)

    monkeypatch.setattr(os.path, "exists", lambda path: True)

    with patch("joblib.load", return_value=MagicMock(transform=lambda df: df.select_dtypes(include="number").values)), \
         patch("lifefinder.models.pytorch_classifier.ExoplanetNN") as mock_nn, \
         patch("lifefinder.models.trainer.Trainer") as mock_trainer, \
         patch("torch.load", return_value={}), \
         patch.object(torch.nn.Module, "load_state_dict", lambda self, state_dict: None):

        # Make the mock model callable directly
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.forward.return_value = torch.tensor([0.5, 0.7], dtype=torch.float32)
        mock_model.__call__ = mock_model.forward
        mock_nn.return_value = mock_model
        mock_trainer.return_value = MagicMock(load_checkpoint=lambda x: None)

        result_df = predict(str(csv_file))
        assert "habitability_prob" in result_df.columns
        assert len(result_df) == 2
        assert result_df["habitability_prob"].between(0, 1).all()
