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
    best_model_path = None
    best_pipeline_path = None
    patience_counter = 0
    training_log = []

    # Create comprehensive training metadata
    training_metadata = {
        "name": "lifefinder",
        "version": cfg.VERSION,
        "experiment": {
            "timestamp": datetime.datetime.now().isoformat(),
            "is_retraining": is_retraining,
            "retrain_files": {
                "pipeline": retrain_pipeline_file,
                "model": retrain_model_file,
            }
            if is_retraining
            else None,
        },
        "config": dict(cfg.TRAINING_CONFIG),
        "data": {
            "raw_shape": raw_df.shape,
            "processed_shape": X.shape,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "feature_count": input_dim,
            "class_distribution": {
                "positive": int(y.sum()),
                "negative": int(len(y) - y.sum()),
            },
        },
        "model": {
            "input_dim": input_dim,
            "output_dim": 1,
            "architecture": "ExoplanetNN",
            "activation": "ReLU",
            "target_feature": cfg.TARGET_FEATURE,
        },
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.info("Starting training...")
    for epoch in range(cfg.TRAINING_CONFIG["epochs"]):
        loss = trainer.train_epoch(train_dataloader)
        val_metrics = trainer.evaluate(
            val_dataloader, cfg.TRAINING_CONFIG["hz_threshold"]
        )
        acc, f1 = val_metrics["accuracy"], val_metrics["f1"]

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": loss,
            "val_accuracy": acc,
            "val_f1": f1,
            "val_precision": val_metrics.get("precision", None),
            "val_recall": val_metrics.get("recall", None),
            "epoch_timestamp": datetime.datetime.now().isoformat(),
            "is_best": False,  # Will update if this is best
        }

        logger.info(
            f"Epoch {epoch + 1}/{cfg.TRAINING_CONFIG['epochs']} - "
            f"Loss: {loss:.4f} - "
            f"Acc: {acc:.3f} - "
            f"F1: {f1:.3f}"
        )

        # Check for improvement
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0

            # Mark this epoch as best
            epoch_log["is_best"] = True

            model_version = f"f1-{best_f1:.3f}_{timestamp}"
            model_path = cfg.MODELS_DIR / f"model_{model_version}.pt"
            pipeline_path = cfg.MODELS_DIR / f"pipeline_{model_version}.pkl"
            metrics_path = cfg.MODELS_DIR / f"metrics_{model_version}.json"

            best_model_path = model_path
            best_pipeline_path = pipeline_path

            # Create final metrics structure
            final_metrics = {
                **training_metadata,
                "training": {
                    "best_epoch": epoch + 1,
                    "best_f1": best_f1,
                    "total_epochs": epoch + 1,
                    "early_stopped": False,
                    "final_patience_counter": patience_counter,
                },
                "history": training_log.copy(),
                "model_artifacts": {
                    "model_path": str(model_path),
                    "pipeline_path": str(pipeline_path),
                    "metrics_path": str(metrics_path),
                },
            }

            # Save artifacts
            trainer.save_checkpoint(model_path)
            joblib.dump(pipeline, pipeline_path)
            Trainer.log_metrics(final_metrics, metrics_path)

            logger.info(f"New best model saved with F1={f1:.3f} at: {model_path}")
        else:
            # Increment patience counter
            # Stop if no improvement for 'patience' epochs
            patience_counter += 1
            if patience_counter >= cfg.TRAINING_CONFIG["patience"]:
                logger.info("Early stopping triggered.")
                break

        # Append epoch log
        training_log.append(epoch_log)

    logger.info("Training complete.")

    return {
        "model": model,
        "trainer": trainer,
        "best_f1": best_f1,
        "best_model_path": best_model_path,
        "best_pipeline_path": best_pipeline_path,
        "pipeline": pipeline,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }
