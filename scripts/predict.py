import joblib
import torch
import pandas as pd

import lifefinder.utils.file_utils as fu
import lifefinder.utils.cli_utils as cli

from lifefinder.models.pytorch_classifier import ExoplanetNN
from lifefinder.models.trainer import Trainer
from lifefinder.logger import get_logger
from lifefinder import config as cfg

logger = get_logger("predict")


def predict(input_file: str, pipeline_file: str, model_file: str) -> pd.DataFrame:
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
    return df


if __name__ == "__main__":
    try:
        logger.info("Lifefinder Prediction Script")
        logger.info("============================")
        logger.info("Press Ctrl+C to abort at any time.")
        logger.info("Please provide the following parameters:\n")

        input_file = input(
            "Path to the input CSV file (e.g., /home/user/exoplanets.csv): "
        ).strip()

        logger.info(f"Input file selected: {input_file}")

        # Validate input file
        fu.validate_file(input_file, "input file", ["csv"])

        # Select model
        model_file, pipeline_file, metrics_file = cli.prompt_model_selection(
            fu, cfg, logger
        )

        display_metrics = (
            input(
                "Would you like to display the training metrics for this model? (y/n) [n]: "
            )
            .strip()
            .lower()
        )

        if display_metrics == "y":
            cli.display_model_metrics(metrics_file, fu, logger)

        hidden_dim = int(
            cli.prompt_with_default(
                "Hidden layer dimension", cfg.TRAINING_CONFIG["hidden_dim"]
            )
        )
        dropout = float(
            cli.prompt_with_default("Dropout rate", cfg.TRAINING_CONFIG["dropout"])
        )

        # Update config with any CLI overrides
        cfg.TRAINING_CONFIG["hidden_dim"] = hidden_dim
        cfg.TRAINING_CONFIG["dropout"] = dropout

        # Run prediction
        result = predict(input_file, pipeline_file, model_file)
        if not result.empty:
            logger.info(
                f"Prediction results:\n{result[['pl_name', 'habitability_prob']]}"
            )
        else:
            logger.error("Prediction failed or returned no results.")
    except KeyboardInterrupt:
        print("\n")
        logger.warning("Prediction interrupted by user.")
    except Exception as e:
        logger.error(f"{e}", exc_info=True)
