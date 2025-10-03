import typer

from typing_extensions import Annotated

app = typer.Typer(
    help="LifeFinder CLI - A tool for training and predicting life expectancy models."
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-v", help="Display the current version of LifeFinder"
        ),
    ] = False,
):
    """LifeFinder CLI - Train and predict life expectancy models."""

    if version:
        from lifefinder import config as cfg

        print(f"LifeFinder version: {cfg.VERSION}")
        raise typer.Exit()

    # If no command is provided and no version flag, show help
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit()


@app.command()
def train(
    default: Annotated[
        bool, typer.Option("--default", help="Use default settings from .env file")
    ] = False,
    retrain: Annotated[
        bool,
        typer.Option(
            "--retrain", help="Retrain using existing pipeline and model files"
        ),
    ] = False,
):
    """Train the life expectancy model."""

    # Import logger here to avoid circular dependencies
    from lifefinder.utils.logger import get_logger

    logger = get_logger("train")

    logger.info("Initializing training process...")
    logger.info("You can abort at any time by pressing Ctrl+C.")

    # Import other dependencies here to avoid circular dependencies
    import lifefinder.utils.cli_utils as cli
    import lifefinder.utils.file_utils as fu
    from lifefinder.train import train as train_model
    from lifefinder import config as cfg

    try:
        # Handle retrain/model selection logic
        model_file, pipeline_file, metrics_file = None, None, None
        if retrain:
            model_file, pipeline_file, metrics_file = cli.prompt_model_selection(
                fu, cfg, logger
            )
            cli.display_model_metrics(metrics_file, fu, logger)

        if default:
            logger.info("Using default settings from .env file.")

            if retrain and model_file and pipeline_file:
                logger.warning(
                    "Ensure the selected model and pipeline match the default config."
                )

            logger.info("Starting the training process...")

            # Run training with default config
            result = train_model(
                retrain_pipeline_file=pipeline_file,
                retrain_model_file=model_file,
                device=None,
            )

            logger.info(f"Best F1 Score: {result['best_f1']:.3f}")

        logger.info(
            "You can also run with --default to use default settings from the .env file."
        )
        logger.info("Please provide the following parameters:\n")

        force = typer.confirm("Force data fetching?", default=False)
        input_limit: int = typer.prompt(
            "Limit number of records to fetch", default=cfg.NASA_API_LIMIT
        )
        batch_size: int = typer.prompt(
            "Batch size for training", default=cfg.TRAINING_CONFIG["batch_size"]
        )
        epochs: int = typer.prompt(
            "Number of training epochs", default=cfg.TRAINING_CONFIG["epochs"]
        )
        learning_rate: float = typer.prompt(
            "Learning rate", default=cfg.TRAINING_CONFIG["learning_rate"]
        )
        hidden_dim: int = typer.prompt(
            "Hidden layer dimension", default=cfg.TRAINING_CONFIG["hidden_dim"]
        )
        dropout: float = typer.prompt(
            "Dropout rate", default=cfg.TRAINING_CONFIG["dropout"]
        )
        val_split: float = typer.prompt(
            "Validation split", default=cfg.TRAINING_CONFIG["val_split"]
        )
        random_state: int = typer.prompt(
            "Random state for splitting", default=cfg.TRAINING_CONFIG["random_state"]
        )
        patience: int = typer.prompt(
            "Early stopping patience", default=cfg.TRAINING_CONFIG.get("patience", 5)
        )
        device: str = typer.prompt(
            "Device to use for training (e.g., 'cpu' or 'cuda')", default=None
        )
        hz_sigma: float = typer.prompt(
            "Habitable zone sigma for classification",
            default=cfg.TRAINING_CONFIG["hz_sigma"],
        )
        hz_threshold: float = typer.prompt(
            "Habitable zone threshold for classification",
            default=cfg.TRAINING_CONFIG["hz_threshold"],
        )

        # Update config with any CLI overrides
        cfg.TRAINING_CONFIG.update(
            {
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "val_split": val_split,
                "random_state": random_state,
                "patience": patience,
                "hz_sigma": hz_sigma,
                "hz_threshold": hz_threshold,
            }
        )

        cfg.NASA_API_LIMIT = input_limit
        cfg.FORCE_NASA_API_FETCH = force

        logger.info("Starting the training process...")

        result = train_model(
            retrain_pipeline_file=pipeline_file,
            retrain_model_file=model_file,
            device=device,
        )
        logger.info(f"Best F1 Score: {result['best_f1']:.3f}")
    except KeyboardInterrupt:
        print("\n")
        logger.warning("Training interrupted by user.")
    except typer.Abort:
        print("\n")
        logger.warning("Training interrupted by user.")
    except Exception as e:
        logger.error(f"{e}", exc_info=True)


@app.command()
def predict():
    """Predict habitability of exoplanets using a trained model."""

    # Import logger here to avoid circular dependencies
    from lifefinder.utils.logger import get_logger

    logger = get_logger("predict")

    logger.info("Initializing prediction process...")
    logger.info("You can abort at any time by pressing Ctrl+C.")

    # Import other dependencies here to avoid circular dependencies
    import lifefinder.utils.cli_utils as cli
    import lifefinder.utils.file_utils as fu
    from lifefinder.predict import predict as predict_habitability
    from lifefinder import config as cfg

    try:
        print("Please provide the following parameters:\n")

        input_file: str = typer.prompt(
            "Path to the input CSV file (e.g., /home/user/exoplanets.csv)"
        ).strip()

        logger.info(f"Input file selected: {input_file}")

        # Validate input file
        fu.validate_file(input_file, "Input file", ["csv"])

        # Select model
        model_file, pipeline_file, metrics_file = cli.prompt_model_selection(
            fu, cfg, logger
        )

        display_metrics = typer.confirm(
            "Would you like to display the training metrics for this model?",
            default=False,
        )

        if display_metrics:
            cli.display_model_metrics(metrics_file, fu, logger)

        hidden_dim: int = typer.prompt(
            "Hidden layer dimension", default=cfg.TRAINING_CONFIG["hidden_dim"]
        )
        dropout: float = typer.prompt(
            "Dropout rate", default=cfg.TRAINING_CONFIG["dropout"]
        )

        save_report_confirm = typer.confirm(
            "Would you like to save the prediction report?", default=False
        )

        report_path = None
        if save_report_confirm:
            report_path = typer.prompt(
                "Path to save prediction report (txt/csv/html) (e.g., /home/user/report.csv)"
            ).strip()

            fu.validate_file(
                report_path,
                "Report file",
                ["txt", "csv", "html"],
                allow_nonexistent=True,
            )

        compute_shap_confirm = typer.confirm(
            "Would you like to compute SHAP values for the predictions?", default=False
        )

        shap_path = None
        if compute_shap_confirm:
            shap_path = typer.prompt(
                "Path to save SHAP summary plot (png/pdf) (e.g., /home/user/shap_summary.png)"
            ).strip()

            fu.validate_file(
                shap_path, "SHAP plot file", ["png", "pdf"], allow_nonexistent=True
            )

        # Update config with any CLI overrides
        cfg.TRAINING_CONFIG["hidden_dim"] = hidden_dim
        cfg.TRAINING_CONFIG["dropout"] = dropout

        # Run prediction
        result = predict_habitability(
            input_file, pipeline_file, model_file, report_path, shap_path
        )

        if not result.empty:
            logger.info(
                f"Prediction results:\n{result[['pl_name', 'habitability_prob']]}"
            )
        else:
            logger.error("Prediction failed or returned no results.")
    except KeyboardInterrupt:
        print("\n")
        logger.warning("Prediction interrupted by user.")
    except typer.Abort:
        print("\n")
        logger.warning("Prediction interrupted by user.")
    except Exception as e:
        logger.error(f"{e}", exc_info=True)


@app.command()
def clean():
    """Clean the input CSV file of exoplanet data."""

    # Import logger here to avoid circular dependencies
    from lifefinder.utils.logger import get_logger

    logger = get_logger("clean")

    logger.info("Initializing data cleaning process...")
    logger.info("You can abort at any time by pressing Ctrl+C.")

    # Import other dependencies here to avoid circular dependencies
    import lifefinder.utils.file_utils as fu
    from lifefinder.clean import clean as clean_data

    try:
        input_file: str = typer.prompt(
            "Path to the input CSV file (e.g., /home/user/exoplanets.csv)"
        ).strip()

        logger.info(f"Input file selected: {input_file}")

        # Validate input file
        fu.validate_file(input_file, "input file", ["csv"])

        output_file: str = typer.prompt(
            "Path to save the cleaned CSV file (e.g., /home/user/cleaned_exoplanets.csv)"
        ).strip()

        logger.info(f"Output file selected: {output_file}")

        # Validate output path
        if not output_file.lower().endswith(".csv"):
            logger.error("The output file must be a .csv file.")
            exit(1)

        # Perform cleaning
        clean_data(input_file, output_file)
    except KeyboardInterrupt:
        print("\n")
        logger.warning("Cleaning interrupted by user.")
    except typer.Abort:
        print("\n")
        logger.warning("Cleaning interrupted by user.")
    except Exception as e:
        logger.error(f"{e}", exc_info=True)


if __name__ == "__main__":
    app()
