import argparse
import joblib
import scipy.sparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Optional

from lifefinder.data.nasa_client import NasaExoplanetClient
from lifefinder.data.preprocessor import build_exoplanet_pipeline
from lifefinder.models.pytorch_classifier import ExoplanetNN
from lifefinder.models.trainer import Trainer
from lifefinder.models.dataset import ExoplanetDataset
from lifefinder.logger import get_logger
from lifefinder import config as cfg

logger = get_logger("train")


def train(device: Optional[str] = None) -> dict:
    """
    Train the exoplanet habitability model.
    Args:
        force (bool): Whether to force data fetching from NASA API.
        limit (int): Limit the number of records fetched from NASA API.
    """

    try:
        # Fetch raw data from NASA Exoplanet Archive
        client = NasaExoplanetClient()
        raw_df = client.fetch_exoplanets(limit=cfg.NASA_API_LIMIT, force=cfg.FORCE_NASA_API_FETCH)
        logger.info(f"Fetched raw data with shape: {raw_df.shape}")

        # Build and fit pipeline
        pipeline = build_exoplanet_pipeline()
        pipeline.fit(raw_df)
        
        # Extract engineered features for X and target y
        df_features = pipeline.named_steps["features"].transform(
            pipeline.named_steps["cleaning"].transform(raw_df)
        )

        # Extract target after feature engineering
        if "habitable_zone_index" not in df_features.columns:
            raise ValueError("No habitable_zone_index found in features")
        y = (df_features["habitable_zone_index"] > cfg.TRAINING_CONFIG["hz_threshold"]).astype(int).values

        # Remove target from X
        X_df = df_features.drop(columns=["habitable_zone_index"])

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
            stratify=y
        )

        # Create datasets
        train_ds = ExoplanetDataset(X_train, y_train)
        val_ds = ExoplanetDataset(X_val, y_val)

        # Initialize data loaders
        train_dataloader = DataLoader(
            train_ds, 
            batch_size=cfg.TRAINING_CONFIG["batch_size"], 
            shuffle=True
        )
        val_dataloader = DataLoader(val_ds, batch_size=cfg.TRAINING_CONFIG["batch_size"])

        # Initialize model and trainer
        input_dim = X.shape[1]
        model = ExoplanetNN(
            input_dim=input_dim,
            hidden_dim=cfg.TRAINING_CONFIG["hidden_dim"],
            dropout=cfg.TRAINING_CONFIG["dropout"]
        )
        trainer = Trainer(
            model,
            lr=cfg.TRAINING_CONFIG["learning_rate"],
            device=device
        )

        # Training loop with early stopping
        best_f1 = 0.0
        patience_counter = 0
        metrics_log = []

        logger.info("Starting training...")
        for epoch in range(cfg.TRAINING_CONFIG["epochs"]):
            loss = trainer.train_epoch(train_dataloader)
            metrics = trainer.evaluate(val_dataloader)
            acc, f1 = metrics["accuracy"], metrics["f1"]

            metrics_log.append({"epoch": epoch + 1, "loss": loss, "accuracy": acc, "f1": f1})
            logger.info(
                f"Epoch {epoch+1}/{cfg.TRAINING_CONFIG['epochs']} - "
                f"Loss: {loss:.4f} - "
                f"Acc: {metrics['accuracy']:.3f} - "
                f"F1: {metrics['f1']:.3f}"
            )

            # Check for improvement
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                trainer.save_checkpoint(cfg.MODEL_CHECKPOINT)
                logger.info(f"New best model saved with F1={f1:.3f}")
            else:
                patience_counter += 1
                if patience_counter >= cfg.TRAINING_CONFIG["patience"]:
                    logger.info("Early stopping triggered.")
                    break

        Trainer.log_metrics(metrics_log)
        logger.info("Training complete.")

        # Save the preprocessing pipeline
        joblib.dump(pipeline, cfg.PIPELINE_PATH)
        logger.info(f"Saved preprocessing pipeline to {cfg.PIPELINE_PATH}")

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
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return {
            "model": None,
            "trainer": None,
            "metrics_log": [],
            "best_f1": 0.0,
            "pipeline": None,
            "X_train": None,
            "y_train": None,
            "X_val": None,
            "y_val": None,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Exoplanet Habitability Model")
    parser.add_argument("--force", action="store_true", help="Force data fetching")
    parser.add_argument("--input_limit", type=int, default=cfg.NASA_API_LIMIT, help="Limit number of records to fetch")
    parser.add_argument("--batch_size", type=int, default=cfg.TRAINING_CONFIG["batch_size"], help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=cfg.TRAINING_CONFIG["epochs"], help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=cfg.TRAINING_CONFIG["learning_rate"], help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=cfg.TRAINING_CONFIG["hidden_dim"], help="Hidden layer dimension")
    parser.add_argument("--dropout", type=float, default=cfg.TRAINING_CONFIG["dropout"], help="Dropout rate")
    parser.add_argument("--val_split", type=float, default=cfg.TRAINING_CONFIG["val_split"], help="Validation split")
    parser.add_argument("--random_state", type=int, default=cfg.TRAINING_CONFIG["random_state"], help="Random state for splitting")
    parser.add_argument("--patience", type=int, default=cfg.TRAINING_CONFIG.get("patience", 5), help="Early stopping patience")
    parser.add_argument("--device", type=str, default=None, help="Device to use for training (e.g., 'cpu' or 'cuda')")
    parser.add_argument("--hz_sigma", type=float, default=cfg.TRAINING_CONFIG["hz_sigma"], help="Habitable zone index threshold for classification")
    parser.add_argument("--hz_threshold", type=float, default=cfg.TRAINING_CONFIG["hz_threshold"], help="Habitable zone threshold for classification")

    args = parser.parse_args()

    # Update config with any CLI overrides
    cfg.TRAINING_CONFIG.update({
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "val_split": args.val_split,
        "random_state": args.random_state,
        "patience": args.patience,
        "hz_sigma": args.hz_sigma,
        "hz_threshold": args.hz_threshold,
    })

    cfg.NASA_API_LIMIT = args.input_limit
    cfg.FORCE_NASA_API_FETCH = args.force

    result = train(device=args.device)
    print(f"Best F1 Score: {result['best_f1']:.3f}")
