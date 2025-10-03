import joblib
import torch
import numpy as np
import pandas as pd

import lifefinder.utils.file_utils as fu

from typing import Optional

from lifefinder.models.pytorch_classifier import ExoplanetNN
from lifefinder.models.trainer import Trainer
from lifefinder.utils.logger import get_logger
from lifefinder import config as cfg

logger = get_logger("predict")


def predict(
    input_file: str,
    pipeline_file: str,
    model_file: str,
    report_path: Optional[str] = None,
    shap_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Predict habitability of exoplanets from input CSV file.
    Args:
        input_file (str): Path to the input CSV file containing exoplanet data.
        pipeline_file (str): Path to the preprocessing pipeline file.
        model_file (str): Path to the trained model checkpoint file.
    Returns:
        pd.DataFrame: DataFrame with original data and predicted habitability probabilities.
    """

    # Ensure input file exists
    fu.validate_file(input_file, "input file", ["csv"])

    # Ensure pipeline exists
    fu.validate_file(pipeline_file, "pipeline file", ["pkl", "joblib"])

    # Ensure model checkpoint exists
    fu.validate_file(model_file, "model file", ["pt"])

    # Load new exoplanet data
    df = pd.read_csv(input_file)
    logger.info(f"Input data shape: {df.shape}")

    # Load the preprocessing pipeline
    pipeline = joblib.load(pipeline_file)
    X = pipeline.transform(df)
    logger.info(f"Transformed data shape: {X.shape}")

    # Load model
    input_dim = X.shape[1]
    model = ExoplanetNN(
        input_dim=input_dim,
        hidden_dim=cfg.TRAINING_CONFIG["hidden_dim"],
        dropout=cfg.TRAINING_CONFIG["dropout"],
    )
    trainer = Trainer(model)
    trainer.load_checkpoint(model_file)

    # Predict
    model.eval()
    with torch.no_grad():
        # Convert sparse matrix to dense if needed
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = X
        X_tensor = torch.tensor(X_dense, dtype=torch.float32)
        probs = model(X_tensor).squeeze().numpy()

    df["habitability_prob"] = probs

    # Optionally save report
    if report_path:
        from lifefinder.reports.report import save_report

        save_report(df, report_path)

    # Optionally compute SHAP values
    if shap_path:
        from lifefinder.interpret.shap_utils import (
            compute_shap_values,
            plot_shap_summary,
        )

        features = X.toarray() if hasattr(X, "toarray") else np.array(X)

        shap_values, shap_features = compute_shap_values(model, features)
        if shap_values is not None and shap_features is not None:
            # Get feature names from the preprocessor step
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            plot_shap_summary(shap_values, feature_names, shap_features, shap_path)

    return df
