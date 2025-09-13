import os
import joblib
import argparse
import pandas as pd
import torch
from lifefinder.data.preprocessor import build_exoplanet_pipeline
from lifefinder.models.pytorch_classifier import ExoplanetNN
from lifefinder.models.trainer import Trainer
from lifefinder.logger import get_logger
from lifefinder import config as cfg

logger = get_logger("predict")


def predict(input_file: str) -> pd.DataFrame:
    """
    Predict habitability of exoplanets from input CSV file.
    Args:
        input_file (str): Path to the input CSV file containing exoplanet data.
    Returns:
        pd.DataFrame: DataFrame with original data and predicted habitability probabilities.
    """
    
    try:
        # Load new exoplanet data
        df = pd.read_csv(input_file)

        # Ensure model and pipeline exist
        if not os.path.exists(cfg.PIPELINE_PATH):
            raise FileNotFoundError(
                f"Pipeline file not found at {cfg.PIPELINE_PATH}. "
                "Please run training first to generate it."
            )

        # Load the preprocessing pipeline
        pipeline = joblib.load(cfg.PIPELINE_PATH)
        X = pipeline.transform(df)

        # Load model
        input_dim = X.shape[1]
        model = ExoplanetNN(
            input_dim=input_dim,
            hidden_dim=cfg.TRAINING_CONFIG["hidden_dim"],
            dropout=cfg.TRAINING_CONFIG["dropout"]
        )
        trainer = Trainer(model)
        trainer.load_checkpoint(cfg.MODEL_CHECKPOINT)

        # Predict
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            probs = model(X_tensor).squeeze().numpy()

        df["habitability_prob"] = probs
        return df
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Exoplanet Habitability")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--hidden_dim", type=int, default=cfg.TRAINING_CONFIG["hidden_dim"], help="Hidden layer dimension")
    parser.add_argument("--dropout", type=float, default=cfg.TRAINING_CONFIG["dropout"], help="Dropout rate")
    args = parser.parse_args()

    # Update config with any CLI overrides
    cfg.TRAINING_CONFIG["hidden_dim"] = args.hidden_dim
    cfg.TRAINING_CONFIG["dropout"] = args.dropout

    result = predict(args.input)
    print(result[["pl_name", "habitability_prob"]])
