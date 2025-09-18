import shap
import torch
import numpy as np

from typing import Optional, Any

from lifefinder.utils.logger import get_logger

logger = get_logger("shap")


def compute_shap_values(model: torch.nn.Module, X: np.ndarray, max_samples: int = 100):
    """
    Compute SHAP values for the given model and input data.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        X (np.ndarray): Input features (preprocessed).
        max_samples (int): Limit for SHAP sampling to reduce compute. Pick a number between 100 and 1000.

    Returns:
        shap_values: SHAP values array.
        X_sample: Sampled input features used for SHAP computation.
    """
    try:
        model.eval()
        background = torch.tensor(X[: min(50, len(X))], dtype=torch.float32)
        X_sample = shap.utils.sample(X, min(max_samples, len(X)))
        X_tensor = torch.tensor(X_sample, dtype=torch.float32)

        # Use KernelExplainer for high-dimensional data
        explainer = shap.DeepExplainer(model, background)
        shap_values_raw: Any = explainer.shap_values(X_tensor, check_additivity=False)

        # Unwrap list (binary classification)
        if isinstance(shap_values_raw, list):
            shap_values_raw = shap_values_raw[0]  # could be 3D

        # Squeeze last dimension if needed
        if (
            isinstance(shap_values_raw, np.ndarray)
            and shap_values_raw.ndim == 3
            and shap_values_raw.shape[-1] == 1
        ):
            shap_values_raw = shap_values_raw.squeeze(-1)

        return shap_values_raw, X_sample
    except Exception as e:
        logger.error(f"Error computing SHAP values: {e}", exc_info=True)
        return None, None


def plot_shap_summary(
    shap_values,
    feature_names: list,
    features: np.ndarray,
    output_path: Optional[str] = None,
):
    """
    Create and optionally save a SHAP summary plot.

    Args:
        shap_values: SHAP values array.
        feature_names (list): Names of the features.
        features (np.ndarray): Original feature values.
        output_path (str): File path to save the plot. File must end with .png or .pdf.

    Returns:
        None
    """
    try:
        shap.summary_plot(
            shap_values, features=features, feature_names=feature_names, show=False
        )
        if output_path:
            import matplotlib.pyplot as plt

            plt.savefig(output_path, bbox_inches="tight")
            plt.close()

            logger.info(f"SHAP summary plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error plotting SHAP summary: {e}", exc_info=True)
