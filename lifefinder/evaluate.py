import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lifefinder.utils.file_utils as fu

from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from lifefinder.models.pytorch_classifier import ExoplanetNN
from lifefinder.models.trainer import Trainer
from lifefinder.reports import report
from lifefinder.interpret.shap_utils import compute_shap_values, plot_shap_summary
from lifefinder.utils.logger import get_logger
from lifefinder import config as cfg

logger = get_logger("evaluate")


def evaluate(
    input_file: str,
    model_file: str,
    pipeline_file: str,
    label_column: str = cfg.TARGET_FEATURE,
    compute_shap: bool = True,
):
    """
    Evaluate a trained model on a dataset, generate metrics, plots, SHAP explanations, and report.
    Args:
        input_file (str): Path to the input CSV file containing exoplanet data.
        model_file (str): Path to the trained model checkpoint file.
        pipeline_file (str): Path to the preprocessing pipeline file.
        label_column (str): Column name for true labels in the input data. Defaults to cfg.TARGET_FEATURE.
        compute_shap (bool): Whether to compute SHAP explanations. Defaults to True.
    Returns:
        dict: Dictionary containing evaluation results, metrics, and file paths.
    """

    # Ensure input file exists
    fu.validate_file(input_file, "input file", ["csv"])

    # Ensure pipeline exists
    fu.validate_file(pipeline_file, "pipeline file", ["pkl", "joblib"])

    # Ensure model checkpoint exists
    fu.validate_file(model_file, "model file", ["pt"])

    # Create a unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(cfg.EVALUATION_DIR)
    output_dir = base_output_dir / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load new exoplanet data
    df = pd.read_csv(input_file)
    logger.info(f"Input data shape: {df.shape}")

    # Load the preprocessing pipeline
    pipeline = joblib.load(pipeline_file)

    # Extract engineered features for X and target y
    df_features = pipeline.named_steps["features"].transform(
        pipeline.named_steps["cleaning"].transform(df)
    )

    # Extract true labels if available
    if label_column in df_features.columns:
        y_true_raw = df_features[label_column].values
        # Convert continuous labels to binary using the same threshold as predictions
        y_true = (y_true_raw > cfg.TRAINING_CONFIG["hz_threshold"]).astype(int)
        # Remove target from X
        X_df = df_features.drop(columns=[label_column])
    else:
        y_true = None
        # Use all features for prediction
        X_df = df_features

    # Preprocess X
    X = pipeline.named_steps["preprocessor"].transform(X_df)

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

    # Predictions
    trainer.model.eval()
    with torch.no_grad():
        # Convert sparse matrix to dense if needed
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = X

        # Convert to tensor
        X_tensor = torch.tensor(X_dense, dtype=torch.float32)
        # Get probabilities
        y_probs = trainer.model(X_tensor).squeeze().numpy()
        # Binarize predictions at habitable zone threshold
        y_pred = (y_probs > cfg.TRAINING_CONFIG["hz_threshold"]).astype(int)

    # Save predictions
    df_preds = pd.DataFrame({"prediction": y_pred, "probability": y_probs})
    preds_file = output_dir / "predictions.csv"
    df_preds.to_csv(preds_file, index=False)
    logger.info(f"Predictions saved to {preds_file}")

    evaluation_results = {
        "metadata": {
            "timestamp": timestamp,
            "output_dir": str(output_dir),
            "input_file": input_file,
            "model_file": model_file,
            "pipeline_file": pipeline_file,
            "n_samples": len(df),
            "n_features": X.shape[1],
        },
        "predictions": {
            "y_pred": y_pred.tolist(),
            "y_probs": y_probs.tolist(),
        },
        "file_paths": {
            "predictions_csv": str(preds_file),
        },
        "model_info": {
            "input_dim": input_dim,
            "hidden_dim": cfg.TRAINING_CONFIG["hidden_dim"],
            "dropout": cfg.TRAINING_CONFIG["dropout"],
            "hz_threshold": cfg.TRAINING_CONFIG["hz_threshold"],
        },
    }

    # Evaluate metrics if labels exist
    if y_true is not None:
        # Convert y_true to numpy array if it's a list
        y_true_np = np.asarray(y_true)

        # Check for edge cases and provide informative logging
        n_positive_pred = np.sum(y_pred)
        n_positive_true = np.sum(y_true_np)
        n_samples = len(y_true_np)

        logger.info(
            f"Evaluation summary: {n_samples} samples, {n_positive_true} true "
            f"positives, {n_positive_pred} predicted positives"
        )

        # Calculate metrics with explicit handling
        accuracy = accuracy_score(y_true_np, y_pred)

        if n_positive_pred == 0:
            logger.warning("No positive predictions made - precision undefined")
            precision = np.nan
        else:
            precision = precision_score(y_true_np, y_pred)

        if n_positive_true == 0:
            logger.warning("No positive samples in dataset - recall undefined")
            recall = np.nan
        else:
            recall = recall_score(y_true_np, y_pred)

        # F1 score - use nan when components are undefined
        if np.isnan(precision) or np.isnan(recall):
            f1 = np.nan
        else:
            f1 = f1_score(y_true_np, y_pred)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_samples": n_samples,
            "n_positive_true": int(n_positive_true),
            "n_positive_pred": int(n_positive_pred),
        }

        # Confusion matrix
        display = ConfusionMatrixDisplay.from_predictions(y_true_np, y_pred)
        cm_file = output_dir / "confusion_matrix.png"
        display.figure_.savefig(cm_file, bbox_inches="tight")
        plt.close(display.figure_)
        logger.info(f"Confusion matrix saved to {cm_file}")

        # ROC AUC
        # Check if we have both classes present
        if len(np.unique(y_true_np)) > 1:
            auc_score = roc_auc_score(y_true_np, y_probs)
        else:
            logger.warning("Only one class present in y_true - ROC AUC undefined")
            auc_score = np.nan
        metrics["roc_auc"] = auc_score

        # ROC Curve
        # Check if only plot if we have both classes
        roc_file = None
        if not np.isnan(auc_score):
            display = RocCurveDisplay.from_predictions(
                y_true_np, y_probs, name="ROC Curve"
            )
            roc_file = output_dir / "roc_curve.png"
            display.figure_.savefig(roc_file, bbox_inches="tight")
            plt.close(display.figure_)
            logger.info(f"ROC curve saved to {roc_file}")

        # Save metrics
        evaluation_results["metrics"] = metrics
        evaluation_results["has_ground_truth"] = True

        # Add file paths for plots
        evaluation_results["file_paths"]["confusion_matrix"] = str(cm_file)
        if roc_file:
            evaluation_results["file_paths"]["roc_curve"] = str(roc_file)
    else:
        evaluation_results["metrics"] = None
        evaluation_results["has_ground_truth"] = False
        logger.info("No ground truth labels available - skipping metric calculation")

    # SHAP explanations
    if compute_shap:
        features = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

        shap_values, X_sample = compute_shap_values(trainer.model, features)

        if shap_values is not None and X_sample is not None:
            # Get feature names from the preprocessor step
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            shap_file = output_dir / "shap_summary.png"
            plot_shap_summary(
                shap_values,
                feature_names=feature_names,
                features=X_sample,
                output_path=str(shap_file),
            )

            evaluation_results["file_paths"]["shap_summary"] = str(shap_file)
            evaluation_results["shap_computed"] = True
        else:
            evaluation_results["shap_computed"] = False
            logger.warning("SHAP values could not be computed")
    else:
        evaluation_results["shap_computed"] = False

    # Generate human-readable report
    report_file = output_dir / "report.csv"
    report.save_report(df_preds, str(report_file), fmt="csv")

    evaluation_results["file_paths"]["report"] = str(report_file)

    return evaluation_results
