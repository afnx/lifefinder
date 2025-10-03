import matplotlib

matplotlib.use("Agg")

import os
import pytest
import numpy as np
import pandas as pd

from unittest.mock import patch, MagicMock

from lifefinder.predict import predict
from lifefinder.models.pytorch_classifier import ExoplanetNN


@pytest.fixture
def sample_input_csv(tmp_path):
    # Create a minimal fake exoplanet dataset
    csv_file = tmp_path / "input.csv"
    df = pd.DataFrame(
        {
            "pl_name": ["TestPlanet"],
            "pl_orbper": [365.0],
            "pl_rade": [1.0],
            "pl_bmasse": [1.0],
            "st_teff": [5778],
            "st_rad": [1.0],
            "st_mass": [1.0],
            "st_metfe": [0.0],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


def dummy_state_dict(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.transform.side_effect = lambda df: df.select_dtypes(
        include="number"
    ).values
    pipeline.named_steps = {
        "preprocessor": MagicMock(
            get_feature_names_out=lambda: [
                "pl_orbper",
                "pl_rade",
                "pl_bmasse",
                "st_teff",
                "st_rad",
                "st_mass",
                "st_metfe",
            ]
        )
    }
    return pipeline


@pytest.fixture
def mock_trainer(request):
    num_features = request.param
    trainer = MagicMock()
    trainer.model = ExoplanetNN(num_features)
    trainer.load_checkpoint = MagicMock()
    return trainer


@pytest.fixture(autouse=True)
def patch_file_utils(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        "lifefinder.utils.file_utils.validate_file", lambda *a, **kw: None
    )


@pytest.mark.parametrize("mock_trainer", [7], indirect=True)
def test_predict_success(sample_input_csv, mock_pipeline, mock_trainer):
    with (
        patch("joblib.load", return_value=mock_pipeline),
        patch(
            "lifefinder.models.pytorch_classifier.ExoplanetNN",
            return_value=mock_trainer.model,
        ),
        patch("lifefinder.models.trainer.Trainer", return_value=mock_trainer),
        patch("torch.load", return_value=dummy_state_dict(mock_trainer.model)),
    ):
        df = predict(str(sample_input_csv), "pipeline.pkl", "model.pt")
        assert "habitability_prob" in df.columns
        assert df["habitability_prob"].between(0, 1).all()
        assert len(df) == 1


@pytest.mark.parametrize(
    "input_df",
    [
        pd.DataFrame({"pl_name": ["A"], "pl_orbper": [None]}),  # Bad input
    ],
)
@pytest.mark.parametrize("mock_trainer", [1], indirect=True)
def test_predict_with_bad_input(tmp_path, input_df, mock_pipeline, mock_trainer):
    csv_file = tmp_path / "bad_input.csv"
    input_df.to_csv(csv_file, index=False)
    with (
        patch("joblib.load", return_value=mock_pipeline),
        patch(
            "lifefinder.models.pytorch_classifier.ExoplanetNN",
            return_value=mock_trainer.model,
        ),
        patch("lifefinder.models.trainer.Trainer", return_value=mock_trainer),
        patch("torch.load", return_value=dummy_state_dict(mock_trainer.model)),
    ):
        df = predict(str(csv_file), "pipeline.pkl", "model.pt")
        assert isinstance(df, pd.DataFrame)


@pytest.mark.parametrize("mock_trainer", [7], indirect=True)
def test_predict_multiple_rows(tmp_path, mock_pipeline, mock_trainer):
    csv_file = tmp_path / "multi_input.csv"
    df = pd.DataFrame(
        {
            "pl_name": ["A", "B"],
            "pl_orbper": [365.0, 200.0],
            "pl_rade": [1.0, 2.0],
            "pl_bmasse": [1.0, 2.0],
            "st_teff": [5778, 5000],
            "st_rad": [1.0, 0.8],
            "st_mass": [1.0, 0.9],
            "st_metfe": [0.0, 0.1],
        }
    )
    df.to_csv(csv_file, index=False)
    with (
        patch("joblib.load", return_value=mock_pipeline),
        patch(
            "lifefinder.models.pytorch_classifier.ExoplanetNN",
            return_value=mock_trainer.model,
        ),
        patch("lifefinder.models.trainer.Trainer", return_value=mock_trainer),
        patch("torch.load", return_value=dummy_state_dict(mock_trainer.model)),
    ):
        result_df = predict(str(csv_file), "pipeline.pkl", "model.pt")
        assert "habitability_prob" in result_df.columns
        assert len(result_df) == 2
        assert result_df["habitability_prob"].between(0, 1).all()


@pytest.mark.parametrize("mock_trainer", [7], indirect=True)
def test_predict_with_report_path(
    tmp_path, sample_input_csv, mock_pipeline, mock_trainer
):
    report_file = tmp_path / "report.csv"
    with (
        patch("joblib.load", return_value=mock_pipeline),
        patch(
            "lifefinder.models.pytorch_classifier.ExoplanetNN",
            return_value=mock_trainer.model,
        ),
        patch("lifefinder.models.trainer.Trainer", return_value=mock_trainer),
        patch("torch.load", return_value=dummy_state_dict(mock_trainer.model)),
        patch("lifefinder.reports.report.save_report") as mock_save_report,
    ):
        result_df = predict(
            str(sample_input_csv),
            "pipeline.pkl",
            "model.pt",
            report_path=str(report_file),
        )
        assert "habitability_prob" in result_df.columns
        mock_save_report.assert_called_once()
        assert result_df["habitability_prob"].between(0, 1).all()


@pytest.mark.parametrize("mock_trainer", [7], indirect=True)
def test_predict_with_shap_path(
    tmp_path, sample_input_csv, mock_pipeline, mock_trainer
):
    shap_file = tmp_path / "shap.png"
    with (
        patch("joblib.load", return_value=mock_pipeline),
        patch(
            "lifefinder.models.pytorch_classifier.ExoplanetNN",
            return_value=mock_trainer.model,
        ),
        patch("lifefinder.models.trainer.Trainer", return_value=mock_trainer),
        patch("torch.load", return_value=dummy_state_dict(mock_trainer.model)),
        patch(
            "lifefinder.interpret.shap_utils.compute_shap_values",
            return_value=(np.array([[0.1] * 7]), np.array([[1.0] * 7])),
        ),
        patch("matplotlib.pyplot.savefig") as mock_savefig,
    ):
        result_df = predict(
            str(sample_input_csv), "pipeline.pkl", "model.pt", shap_path=str(shap_file)
        )
        assert "habitability_prob" in result_df.columns
        mock_savefig.assert_called_once()
        assert result_df["habitability_prob"].between(0, 1).all()
