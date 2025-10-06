import pytest
import pandas as pd
import numpy as np
import torch

from unittest.mock import patch, MagicMock

from lifefinder.evaluate import evaluate
from lifefinder import config as cfg


@pytest.fixture(scope="session", autouse=True)
def setup_tmp_dirs(tmp_path_factory):
    """Redirect all config paths to a tmp folder so tests don't overwrite real files."""
    tmpdir = tmp_path_factory.mktemp("lifefinder_eval_tests")
    cfg.ROOT = tmpdir
    cfg.DATA_DIR = tmpdir / "data"
    cfg.EVALUATION_DIR = tmpdir / "evaluations"
    cfg.MODELS_DIR = tmpdir / "models"

    cfg.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Set up training config
    cfg.TRAINING_CONFIG = {"hidden_dim": 64, "dropout": 0.2, "hz_threshold": 0.5}
    cfg.TARGET_FEATURE = "habitable_zone_index"

    return tmpdir


@pytest.fixture
def mock_files(tmp_path):
    """Create mock files for testing."""
    input_csv = tmp_path / "test_data.csv"
    model_pt = tmp_path / "model.pt"
    pipeline_pkl = tmp_path / "pipeline.pkl"

    # Create dummy CSV data
    df = pd.DataFrame(
        {
            "pl_name": ["Planet1", "Planet2", "Planet3"],
            "pl_orbper": [1.0, 2.0, 3.0],
            "pl_rade": [1.1, 2.0, 1.5],
            "habitable_zone_index": [0.3, 0.7, 0.4],
        }
    )
    df.to_csv(input_csv, index=False)

    # Create dummy model file
    model_pt.touch()

    # Create dummy pipeline file
    pipeline_pkl.touch()

    return {
        "input_csv": str(input_csv),
        "model_pt": str(model_pt),
        "pipeline_pkl": str(pipeline_pkl),
    }


@pytest.fixture
def mock_pipeline():
    """Create a mock preprocessing pipeline."""
    pipeline = MagicMock()

    # Mock cleaning step
    cleaning_step = MagicMock()
    cleaning_step.transform.return_value = pd.DataFrame(
        {
            "pl_name": ["Planet1", "Planet2", "Planet3"],
            "pl_orbper": [1.0, 2.0, 3.0],
            "pl_rade": [1.1, 2.0, 1.5],
            "habitable_zone_index": [0.3, 0.7, 0.4],
        }
    )

    # Mock features step
    features_step = MagicMock()
    features_step.transform.return_value = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [0.5, 1.0, 1.5],
            "habitable_zone_index": [0.3, 0.7, 0.4],
        }
    )

    # Mock preprocessor step
    preprocessor_step = MagicMock()
    X_processed = np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]])
    preprocessor_step.transform.return_value = X_processed
    preprocessor_step.get_feature_names_out.return_value = ["feature1", "feature2"]

    # Set up named_steps
    pipeline.named_steps = {
        "cleaning": cleaning_step,
        "features": features_step,
        "preprocessor": preprocessor_step,
    }

    return pipeline


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model."""
    model = MagicMock()
    model.eval.return_value = None

    # Mock predictions
    predictions = torch.tensor([0.3, 0.8, 0.4], dtype=torch.float32)
    model.return_value = predictions

    return model


@pytest.fixture
def mock_trainer(mock_model):
    """Create a mock trainer with model."""
    trainer = MagicMock()
    trainer.model = mock_model
    trainer.load_checkpoint.return_value = None
    return trainer


@patch("lifefinder.evaluate.joblib.load")
@patch("lifefinder.evaluate.pd.read_csv")
@patch("lifefinder.evaluate.fu.validate_file")
@patch("lifefinder.evaluate.ExoplanetNN")
@patch("lifefinder.evaluate.Trainer")
@patch("lifefinder.evaluate.plot_shap_summary")
@patch("lifefinder.evaluate.compute_shap_values")
@patch("lifefinder.evaluate.report.save_report")
def test_evaluate_basic_functionality(
    mock_save_report,
    mock_compute_shap,
    mock_plot_shap,
    mock_trainer_class,
    mock_model_class,
    mock_validate_file,
    mock_read_csv,
    mock_joblib_load,
    mock_files,
    mock_pipeline,
    mock_trainer,
):
    """Test basic evaluation functionality with ground truth labels."""

    # Setup mocks
    df = pd.DataFrame(
        {
            "pl_name": ["Planet1", "Planet2", "Planet3"],
            "pl_orbper": [1.0, 2.0, 3.0],
            "pl_rade": [1.1, 2.0, 1.5],
            "habitable_zone_index": [0.3, 0.7, 0.4],
        }
    )
    mock_read_csv.return_value = df
    mock_joblib_load.return_value = mock_pipeline
    mock_trainer_class.return_value = mock_trainer
    mock_compute_shap.return_value = (
        np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]]),
    )

    # Run evaluation
    result = evaluate(
        input_file=mock_files["input_csv"],
        model_file=mock_files["model_pt"],
        pipeline_file=mock_files["pipeline_pkl"],
    )

    # Verify result structure
    assert "metadata" in result
    assert "predictions" in result
    assert "metrics" in result
    assert "file_paths" in result
    assert "model_info" in result
    assert result["has_ground_truth"] is True
    assert result["shap_computed"] is True

    # Verify metadata
    assert result["metadata"]["n_samples"] == 3
    assert result["metadata"]["n_features"] == 2

    # Verify predictions
    assert len(result["predictions"]["y_pred"]) == 3
    assert len(result["predictions"]["y_probs"]) == 3

    # Verify metrics exist
    assert "accuracy" in result["metrics"]
    assert "precision" in result["metrics"]
    assert "recall" in result["metrics"]
    assert "f1" in result["metrics"]
    assert "roc_auc" in result["metrics"]


@patch("lifefinder.evaluate.fu.validate_file")
def test_evaluate_file_validation_errors(mock_validate_file, mock_files):
    """Test that evaluation raises proper errors for invalid files."""

    # Test missing input file
    mock_validate_file.side_effect = [
        FileNotFoundError("Input file not found"),
        None,
        None,
    ]

    with pytest.raises(FileNotFoundError):
        evaluate(
            input_file="nonexistent.csv",
            model_file=mock_files["model_pt"],
            pipeline_file=mock_files["pipeline_pkl"],
        )

    # Test missing pipeline file
    mock_validate_file.side_effect = [
        None,
        FileNotFoundError("Pipeline file not found"),
        None,
    ]

    with pytest.raises(FileNotFoundError):
        evaluate(
            input_file=mock_files["input_csv"],
            model_file=mock_files["model_pt"],
            pipeline_file="nonexistent.pkl",
        )

    # Test missing model file
    mock_validate_file.side_effect = [
        None,
        None,
        FileNotFoundError("Model file not found"),
    ]

    with pytest.raises(FileNotFoundError):
        evaluate(
            input_file=mock_files["input_csv"],
            model_file="nonexistent.pt",
            pipeline_file=mock_files["pipeline_pkl"],
        )


@patch("lifefinder.evaluate.joblib.load")
@patch("lifefinder.evaluate.pd.read_csv")
@patch("lifefinder.evaluate.fu.validate_file")
@patch("lifefinder.evaluate.ExoplanetNN")
@patch("lifefinder.evaluate.Trainer")
@patch("lifefinder.evaluate.report.save_report")
def test_evaluate_no_ground_truth(
    mock_save_report,
    mock_trainer_class,
    mock_model_class,
    mock_validate_file,
    mock_read_csv,
    mock_joblib_load,
    mock_files,
    mock_pipeline,
    mock_trainer,
):
    """Test evaluation without ground truth labels."""

    # Create pipeline that doesn't include target column in features
    features_step = MagicMock()
    features_step.transform.return_value = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [0.5, 1.0, 1.5],
            # No habitable_zone_index column
        }
    )
    mock_pipeline.named_steps["features"] = features_step

    # Setup mocks
    df = pd.DataFrame(
        {
            "pl_name": ["Planet1", "Planet2", "Planet3"],
            "pl_orbper": [1.0, 2.0, 3.0],
            "pl_rade": [1.1, 2.0, 1.5],
            # No target column
        }
    )
    mock_read_csv.return_value = df
    mock_joblib_load.return_value = mock_pipeline
    mock_trainer_class.return_value = mock_trainer

    # Run evaluation
    result = evaluate(
        input_file=mock_files["input_csv"],
        model_file=mock_files["model_pt"],
        pipeline_file=mock_files["pipeline_pkl"],
        compute_shap=False,
    )

    # Verify result structure
    assert result["has_ground_truth"] is False
    assert result["metrics"] is None
    assert result["shap_computed"] is False
    assert "confusion_matrix" not in result["file_paths"]
    assert "roc_curve" not in result["file_paths"]


@patch("lifefinder.evaluate.joblib.load")
@patch("lifefinder.evaluate.pd.read_csv")
@patch("lifefinder.evaluate.fu.validate_file")
@patch("lifefinder.evaluate.ExoplanetNN")
@patch("lifefinder.evaluate.Trainer")
@patch("lifefinder.evaluate.report.save_report")
def test_evaluate_edge_cases_all_negative_predictions(
    mock_save_report,
    mock_trainer_class,
    mock_model_class,
    mock_validate_file,
    mock_read_csv,
    mock_joblib_load,
    mock_files,
    mock_pipeline,
    mock_trainer,
):
    """Test evaluation with all negative predictions (precision undefined)."""

    # Create model that predicts all negatives
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    predictions = torch.tensor(
        [0.1, 0.2, 0.3], dtype=torch.float32
    )  # All below 0.5 threshold
    mock_model.return_value = predictions
    mock_trainer.model = mock_model

    # Setup mocks
    df = pd.DataFrame(
        {
            "pl_name": ["Planet1", "Planet2", "Planet3"],
            "pl_orbper": [1.0, 2.0, 3.0],
            "pl_rade": [1.1, 2.0, 1.5],
            "habitable_zone_index": [0.3, 0.7, 0.4],
        }
    )
    mock_read_csv.return_value = df
    mock_joblib_load.return_value = mock_pipeline
    mock_trainer_class.return_value = mock_trainer

    # Run evaluation
    result = evaluate(
        input_file=mock_files["input_csv"],
        model_file=mock_files["model_pt"],
        pipeline_file=mock_files["pipeline_pkl"],
        compute_shap=False,
    )

    # Verify precision is NaN but other metrics exist
    assert np.isnan(result["metrics"]["precision"])
    assert not np.isnan(result["metrics"]["accuracy"])
    assert result["metrics"]["n_positive_pred"] == 0


@patch("lifefinder.evaluate.joblib.load")
@patch("lifefinder.evaluate.pd.read_csv")
@patch("lifefinder.evaluate.fu.validate_file")
@patch("lifefinder.evaluate.ExoplanetNN")
@patch("lifefinder.evaluate.Trainer")
@patch("lifefinder.evaluate.report.save_report")
def test_evaluate_edge_cases_no_positive_labels(
    mock_save_report,
    mock_trainer_class,
    mock_model_class,
    mock_validate_file,
    mock_read_csv,
    mock_joblib_load,
    mock_files,
    mock_pipeline,
    mock_trainer,
):
    """Test evaluation with no positive labels in ground truth (recall undefined)."""

    # Create features with all negative labels
    features_step = MagicMock()
    features_step.transform.return_value = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [0.5, 1.0, 1.5],
            "habitable_zone_index": [0.1, 0.2, 0.3],  # All below 0.5 threshold
        }
    )
    mock_pipeline.named_steps["features"] = features_step

    # Setup mocks
    df = pd.DataFrame(
        {
            "pl_name": ["Planet1", "Planet2", "Planet3"],
            "pl_orbper": [1.0, 2.0, 3.0],
            "pl_rade": [1.1, 2.0, 1.5],
            "habitable_zone_index": [0.1, 0.2, 0.3],
        }
    )
    mock_read_csv.return_value = df
    mock_joblib_load.return_value = mock_pipeline
    mock_trainer_class.return_value = mock_trainer

    # Run evaluation
    result = evaluate(
        input_file=mock_files["input_csv"],
        model_file=mock_files["model_pt"],
        pipeline_file=mock_files["pipeline_pkl"],
        compute_shap=False,
    )

    # Verify recall is NaN but other metrics exist
    assert np.isnan(result["metrics"]["recall"])
    assert not np.isnan(result["metrics"]["accuracy"])
    assert result["metrics"]["n_positive_true"] == 0


@patch("lifefinder.evaluate.joblib.load")
@patch("lifefinder.evaluate.pd.read_csv")
@patch("lifefinder.evaluate.fu.validate_file")
@patch("lifefinder.evaluate.ExoplanetNN")
@patch("lifefinder.evaluate.Trainer")
@patch("lifefinder.evaluate.compute_shap_values")
@patch("lifefinder.evaluate.report.save_report")
def test_evaluate_shap_computation_fails(
    mock_save_report,
    mock_compute_shap,
    mock_trainer_class,
    mock_model_class,
    mock_validate_file,
    mock_read_csv,
    mock_joblib_load,
    mock_files,
    mock_pipeline,
    mock_trainer,
):
    """Test evaluation when SHAP computation fails."""

    # Setup mocks
    df = pd.DataFrame(
        {
            "pl_name": ["Planet1", "Planet2", "Planet3"],
            "pl_orbper": [1.0, 2.0, 3.0],
            "pl_rade": [1.1, 2.0, 1.5],
            "habitable_zone_index": [0.3, 0.7, 0.4],
        }
    )
    mock_read_csv.return_value = df
    mock_joblib_load.return_value = mock_pipeline
    mock_trainer_class.return_value = mock_trainer
    mock_compute_shap.return_value = (None, None)  # SHAP computation fails

    # Run evaluation
    result = evaluate(
        input_file=mock_files["input_csv"],
        model_file=mock_files["model_pt"],
        pipeline_file=mock_files["pipeline_pkl"],
        compute_shap=True,
    )

    # Verify SHAP failed but evaluation completed
    assert result["shap_computed"] is False
    assert "shap_summary" not in result["file_paths"]
    assert result["has_ground_truth"] is True  # Other parts should still work


@patch("lifefinder.evaluate.joblib.load")
@patch("lifefinder.evaluate.pd.read_csv")
@patch("lifefinder.evaluate.fu.validate_file")
@patch("lifefinder.evaluate.ExoplanetNN")
@patch("lifefinder.evaluate.Trainer")
@patch("lifefinder.evaluate.report.save_report")
def test_evaluate_sparse_matrix_handling(
    mock_save_report,
    mock_trainer_class,
    mock_model_class,
    mock_validate_file,
    mock_read_csv,
    mock_joblib_load,
    mock_files,
    mock_pipeline,
    mock_trainer,
):
    """Test evaluation handles sparse matrices correctly."""

    # Create a mock sparse matrix
    from scipy.sparse import csr_matrix

    sparse_X = csr_matrix(np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]]))

    preprocessor_step = MagicMock()
    preprocessor_step.transform.return_value = sparse_X
    preprocessor_step.get_feature_names_out.return_value = ["feature1", "feature2"]
    mock_pipeline.named_steps["preprocessor"] = preprocessor_step

    # Setup mocks
    df = pd.DataFrame(
        {
            "pl_name": ["Planet1", "Planet2", "Planet3"],
            "pl_orbper": [1.0, 2.0, 3.0],
            "pl_rade": [1.1, 2.0, 1.5],
            "habitable_zone_index": [0.3, 0.7, 0.4],
        }
    )
    mock_read_csv.return_value = df
    mock_joblib_load.return_value = mock_pipeline
    mock_trainer_class.return_value = mock_trainer

    # Run evaluation
    result = evaluate(
        input_file=mock_files["input_csv"],
        model_file=mock_files["model_pt"],
        pipeline_file=mock_files["pipeline_pkl"],
        compute_shap=False,
    )

    # Verify evaluation completed successfully
    assert result["has_ground_truth"] is True
    assert len(result["predictions"]["y_pred"]) == 3
    assert len(result["predictions"]["y_probs"]) == 3
