import datetime
import joblib
import scipy.sparse
from typing import Optional

import lifefinder.utils.file_utils as fu

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from lifefinder.data.nasa_client import NasaExoplanetClient
from lifefinder.data.preprocessor import build_exoplanet_pipeline
from lifefinder.models.pytorch_classifier import ExoplanetNN
from lifefinder.models.trainer import Trainer
from lifefinder.models.dataset import ExoplanetDataset
from lifefinder.utils.logger import get_logger
from lifefinder import config as cfg

logger = get_logger("train")


def train(
    retrain_pipeline_file: Optional[str] = None,
    retrain_model_file: Optional[str] = None,
    device: Optional[str] = None,
) -> dict:
    """
    Train the exoplanet habitability model and save the trained model and pipeline.
    Args:
        retrain_pipeline_file (str, optional): Path to an existing pipeline file to retrain. Defaults to None.
        retrain_model_file (str, optional): Path to an existing model file to retrain. Defaults to None.
        device (str, optional): Device to use for training (e.g., 'cpu' or 'cuda'). Defaults to None.
    Returns:
        dict: Dictionary containing training results and metrics.
    """

    is_retraining = False
    if retrain_pipeline_file or retrain_model_file:
        if not retrain_pipeline_file or not retrain_model_file:
            raise ValueError(
                "Both pipeline file and model file must be provided for retraining."
            )

        # Validate retrain files
        fu.validate_file(retrain_pipeline_file, "pipeline file", ["pkl", "joblib"])
        fu.validate_file(retrain_model_file, "model file", ["pt"])

        is_retraining = True
        logger.info("Retraining mode enabled.")

    # Fetch raw data from NASA Exoplanet Archive
    client = NasaExoplanetClient()
    raw_df = client.fetch_exoplanets(
        limit=cfg.NASA_API_LIMIT, force=cfg.FORCE_NASA_API_FETCH
    )
    logger.info(f"Fetched raw data with shape: {raw_df.shape}")

    # Build and fit pipeline
    if is_retraining:
        pipeline = joblib.load(retrain_pipeline_file)
        logger.info(f"Loaded existing pipeline from: {retrain_pipeline_file}")
    else:
        pipeline = build_exoplanet_pipeline()
    pipeline.fit(raw_df)

    # Extract engineered features for X and target y
    df_features = pipeline.named_steps["features"].transform(
        pipeline.named_steps["cleaning"].transform(raw_df)
    )

    # Extract target after feature engineering
    if cfg.TARGET_FEATURE not in df_features.columns:
        raise ValueError(f"No {cfg.TARGET_FEATURE} found in features")
    y = (
        (df_features[cfg.TARGET_FEATURE] > cfg.TRAINING_CONFIG["hz_threshold"])
        .astype(int)
        .values
    )

    # Remove target from X
    X_df = df_features.drop(columns=[cfg.TARGET_FEATURE])

    # Preprocess X
    X = pipeline.named_steps["preprocessor"].transform(X_df)

    if scipy.sparse.issparse(X):
        X = X.toarray()

    logger.info(f"Processed data with shape: {X.shape}")

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.TRAINING_CONFIG["val_split"],
        random_state=cfg.TRAINING_CONFIG["random_state"],
        stratify=y,
    )

    # Create datasets
    train_ds = ExoplanetDataset(X_train, y_train)
    val_ds = ExoplanetDataset(X_val, y_val)

    # Initialize data loaders
    train_dataloader = DataLoader(
        train_ds, batch_size=cfg.TRAINING_CONFIG["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=cfg.TRAINING_CONFIG["batch_size"])

    # Initialize model and trainer
    input_dim = X.shape[1]
    model = ExoplanetNN(
        input_dim=input_dim,
        hidden_dim=cfg.TRAINING_CONFIG["hidden_dim"],
        dropout=cfg.TRAINING_CONFIG["dropout"],
    )
    trainer = Trainer(model, lr=cfg.TRAINING_CONFIG["learning_rate"], device=device)

    if is_retraining:
        trainer.load_checkpoint(retrain_model_file)
        logger.info(f"Loaded existing model from: {retrain_model_file}")

    # Training loop with early stopping
    best_f1 = 0.0
    patience_counter = 0
    metrics_log = []

    # Add training config as header in metrics log
    metrics_log.append({"config": dict(cfg.TRAINING_CONFIG)})

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.info("Starting training...")
    for epoch in range(cfg.TRAINING_CONFIG["epochs"]):
        loss = trainer.train_epoch(train_dataloader)
        metrics = trainer.evaluate(val_dataloader)
        acc, f1 = metrics["accuracy"], metrics["f1"]

        metrics_log.append(
            {
                "epoch": epoch + 1,
                "loss": loss,
                "accuracy": acc,
                "f1": f1,
                "timestamp": timestamp,
            }
        )

        logger.info(
            f"Epoch {epoch + 1}/{cfg.TRAINING_CONFIG['epochs']} - "
            f"Loss: {loss:.4f} - "
            f"Acc: {metrics['accuracy']:.3f} - "
            f"F1: {metrics['f1']:.3f}"
        )

        # Check for improvement
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0

            model_version = f"f1-{best_f1:.3f}_{timestamp}"
            model_path = cfg.MODELS_DIR / f"model_{model_version}.pt"
            pipeline_path = cfg.MODELS_DIR / f"pipeline_{model_version}.pkl"
            metrics_path = cfg.MODELS_DIR / f"metrics_{model_version}.json"

            # Save the best model
            trainer.save_checkpoint(model_path)
            # Save the preprocessing pipeline
            joblib.dump(pipeline, pipeline_path)
            # Save training metrics
            Trainer.log_metrics(metrics_log, metrics_path)

            logger.info(f"New best model saved with F1={f1:.3f} at: {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.TRAINING_CONFIG["patience"]:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training complete.")

    return {
        "model": model,
        "trainer": trainer,
        "metrics_log": metrics_log,
        "best_f1": best_f1,
        "pipeline": pipeline,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }
