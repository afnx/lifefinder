from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from lifefinder.data.nasa_client import NasaExoplanetClient
from lifefinder.data.preprocessor import build_exoplanet_pipeline
from lifefinder.models.pytorch_classifier import ExoplanetNN
from lifefinder.models.trainer import Trainer
from lifefinder.models.dataset import ExoplanetDataset
from lifefinder.logger import get_logger
from lifefinder import config as cfg

logger = get_logger("main")


def main():
    try:
        # Fetch raw data from NASA Exoplanet Archive
        client = NasaExoplanetClient()
        raw_df = client.fetch_exoplanets(limit=cfg.NASA_API_LIMIT)
        logger.info(f"Fetched raw data with shape: {raw_df.shape}")

        # Process data using the preprocessing pipeline
        pipeline = build_exoplanet_pipeline()
        X = pipeline.fit_transform(raw_df)
        logger.info(f"Processed data with shape: {X.shape}")

        # Extract target variable if available
        if "habitable_zone_index" in pipeline.named_steps["features"].transform(raw_df).columns:
            y = (pipeline.named_steps["features"].transform(raw_df)["habitable_zone_index"] > 0.5).astype(int).values
        else:
            raise ValueError("No habitable_zone_index found in features")
        
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
        train_loader = DataLoader(
            train_ds, 
            batch_size=cfg.TRAINING_CONFIG["batch_size"], 
            shuffle=True
        )
        val_loader = DataLoader(val_ds, batch_size=cfg.TRAINING_CONFIG["batch_size"])

        # Initialize model and trainer
        input_dim = X.shape[1]
        model = ExoplanetNN(
            input_dim=input_dim,
            hidden_dim=cfg.TRAINING_CONFIG["hidden_dim"],
            dropout=cfg.TRAINING_CONFIG["dropout"]
        )
        trainer = Trainer(
            model,
            lr=cfg.TRAINING_CONFIG["learning_rate"]
        )

        # Train the model
        logger.info("Starting training...")
        for epoch in range(cfg.TRAINING_CONFIG["epochs"]):
            loss = trainer.train_epoch(train_loader)
            metrics = trainer.evaluate(val_loader)
            logger.info(
                f"Epoch {epoch+1}/{cfg.TRAINING_CONFIG['epochs']} - "
                f"Loss: {loss:.4f} - "
                f"Acc: {metrics['accuracy']:.3f} - "
                f"F1: {metrics['f1']:.3f}"
            )

        logger.info("Training complete.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)  


if __name__ == "__main__":
    main()
