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

        print(f"LifeFinder v{cfg.VERSION} ")
        raise typer.Exit()

    # If no command is provided and no version flag, show help
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit()


@app.command()
def configure(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
):
    """Configure the LifeFinder settings."""

    # Import logger here to avoid circular dependencies
    from lifefinder.utils.logger import get_logger

    logger = get_logger("configure")

    from pathlib import Path

    try:
        example_path = Path(__file__).parent / ".env.example"
        env_path = Path.home() / ".lifefinder" / ".env"

        logger.info("Starting configuration process...")
        logger.info("You can abort at any time by pressing Ctrl+C.")

        config = []
        with open(example_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    config.append((line, None))
                else:
                    if "=" in line:
                        key, val = line.split("=", 1)
                        config.append((key.strip(), val.strip()))
                    else:
                        config.append((line, None))

        logger.info("Please enter the configuration values.\n")

        new_config = []
        for entry, value in config:
            if value is None:
                new_config.append((entry, value))
            else:
                user_val = typer.prompt(entry, default=value)
                new_config.append((entry, user_val))

        logger.info(f"Saving configuration to {env_path}")

        # Ensure every missing parent folder in env_path exists
        env_path.parent.mkdir(parents=True, exist_ok=True)

        with open(env_path, "w") as f:
            for entry, value in new_config:
                if value is None:
                    f.write(f"{entry}\n")
                else:
                    f.write(f"{entry}={value}\n")

        logger.info("Success: Configuration saved (overwritten if existed).")
        logger.info("Creating necessary directories...")

        import lifefinder.config as cfg

        # Create necessary directories
        for directory in [
            cfg.ARTIFACTS_DIR,
            cfg.DATA_DIR,
            cfg.RAW_DIR,
            cfg.PROCESSED_DIR,
            cfg.CACHE_DIR,
            cfg.MODELS_DIR,
            cfg.EVALUATION_DIR,
        ]:
            dir_path = Path(directory).expanduser()
            logger.info(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info("Success: All necessary directories are set up.")
        logger.info("Configuration process completed.")
        logger.info("You can run lifefinder configure again to update settings.")
        print("\nThank you for using LifeFinder!")
        print("\nCrafted with ❤️  by afnx (github.com/afnx)\n")
    except KeyboardInterrupt:
        print("\n")
        logger.warning("Training interrupted by user.")
    except typer.Abort:
        print("\n")
        logger.warning("Training interrupted by user.")
    except FileNotFoundError:
        logger.error(
            "Could not create configuration file. Check permissions and try again."
        )
    except Exception as e:
        logger.error(f"{e}", exc_info=verbose)


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
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
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
        # Check configuration before proceeding
        _check_configuration()

        # Default model parameters
        hidden_dim: int = cfg.TRAINING_CONFIG["hidden_dim"]
        dropout: float = cfg.TRAINING_CONFIG["dropout"]

        # Handle retrain/model selection logic
        model_file, pipeline_file, metrics_file = None, None, None
        if retrain:
            model_file, pipeline_file, metrics_file = cli.prompt_model_selection(
                fu, cfg, logger
            )

            # Load config and model info from metrics file
            config, _ = fu.extract_config_and_model_from_metrics(metrics_file)

            if not isinstance(config, dict) or not config:
                logger.error("Config loaded from metrics file is empty or invalid.")
                raise typer.Exit(code=1)

            # Update relevant training config settings
            cfg.TRAINING_CONFIG.update(config)
            # Override hidden_dim and dropout from loaded config
            hidden_dim = cfg.TRAINING_CONFIG["hidden_dim"]
            dropout = cfg.TRAINING_CONFIG["dropout"]

        if default:
            if retrain and model_file and pipeline_file:
                logger.info(
                    "Using hyperparameters from the selected model's metrics file."
                )
            else:
                logger.info("Using default settings from .env file.")

            logger.info("Starting the training process...")

            # Run training with default config
            result = train_model(
                retrain_pipeline_file=pipeline_file,
                retrain_model_file=model_file,
                device=None,
            )

            logger.info(f"Best F1 Score: {result['best_f1']:.3f}")

            # Exit after training with default settings
            return

        logger.info(
            "You can also run with --default to use default settings from the .env file.\n"
        )

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

        # Only prompt for these if not retraining
        if not retrain:
            hidden_dim = typer.prompt(
                "Hidden layer dimension", default=cfg.TRAINING_CONFIG["hidden_dim"]
            )
            dropout = typer.prompt(
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
            "Device to use for training (e.g., 'cpu' or 'cuda')", default="cpu"
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
    except FileNotFoundError:
        logger.error(
            "One or more specified files were not found. Please check the paths and try again."
        )
    except Exception as e:
        logger.error(f"{e}", exc_info=verbose)


@app.command()
def predict(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
):
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
        # Check configuration before proceeding
        _check_configuration()

        input_file: str = typer.prompt(
            "\nPath to the input CSV file (e.g., /home/user/exoplanets.csv)"
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

        # Load config and model info from metrics file
        config, _ = fu.extract_config_and_model_from_metrics(metrics_file)

        if not isinstance(config, dict) or not config:
            logger.error("Config loaded from metrics file is empty or invalid.")
            raise typer.Exit(code=1)

        # Update relevant training config settings
        cfg.TRAINING_CONFIG.update(config)

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
    except FileNotFoundError:
        logger.error(
            "One or more specified files were not found. Please check the paths and try again."
        )
    except Exception as e:
        logger.error(f"{e}", exc_info=verbose)


@app.command()
def evaluate(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
):
    """Evaluate the trained model on a dataset."""

    # Import logger here to avoid circular dependencies
    from lifefinder.utils.logger import get_logger

    logger = get_logger("evaluate")

    logger.info("Initializing evaluation process...")
    logger.info("You can abort at any time by pressing Ctrl+C.")

    # Import other dependencies here to avoid circular dependencies
    import json
    import lifefinder.utils.cli_utils as cli
    import lifefinder.utils.file_utils as fu
    from lifefinder.evaluate import evaluate as evaluate_model
    from lifefinder import config as cfg

    try:
        # Check configuration before proceeding
        _check_configuration()

        input_file: str = typer.prompt(
            "\nPath to the input CSV file (e.g., /home/user/exoplanets.csv)"
        ).strip()

        logger.info(f"Input file selected: {input_file}")

        # Validate input file
        fu.validate_file(input_file, "Input file", ["csv"])

        # Select model
        model_file, pipeline_file, metrics_file = cli.prompt_model_selection(
            fu, cfg, logger
        )

        # Validate metrics file
        fu.validate_file(metrics_file, "Metrics file", ["json"])

        # Load config and model info from metrics file
        config, model = fu.extract_config_and_model_from_metrics(metrics_file)

        # Update target feature if specified in model
        target_feature = model.get("target_feature", cfg.TARGET_FEATURE)
        cfg.TARGET_FEATURE = target_feature

        # Update relevant training config settings
        for key in ["hidden_dim", "dropout", "hz_threshold"]:
            if key in config:
                cfg.TRAINING_CONFIG[key] = config[key]

        compute_shap_confirm = typer.confirm(
            "Would you like to compute SHAP values during evaluation?", default=True
        )

        # Run evaluation
        result = evaluate_model(
            input_file,
            model_file,
            pipeline_file,
            label_column=target_feature,
            compute_shap=compute_shap_confirm,
        )

        if result:
            result_str = json.dumps(result, indent=2)
            logger.info(f"Evaluation results:\n{result_str}")
        else:
            logger.error("Evaluation failed or returned no results.")
    except KeyboardInterrupt:
        print("\n")
        logger.warning("Evaluation interrupted by user.")
    except typer.Abort:
        print("\n")
        logger.warning("Evaluation interrupted by user.")
    except FileNotFoundError:
        logger.error(
            "One or more specified files were not found. Please check the paths and try again."
        )
    except Exception as e:
        logger.error(f"{e}", exc_info=verbose)


@app.command()
def clean(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
):
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
            "\nPath to the input CSV file (e.g., /home/user/exoplanets.csv)"
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
    except FileNotFoundError:
        logger.error(
            "One or more specified files were not found. Please check the paths and try again."
        )
    except Exception as e:
        logger.error(f"{e}", exc_info=verbose)


def _check_configuration():
    """Check if the configuration file exists and is valid."""
    import os

    from pathlib import Path

    env_path = Path.home() / ".lifefinder" / ".env"
    if not env_path.exists():
        raise Exception(
            "Configuration file not found. Please run 'lifefinder configure' first."
        )

    # Basic validation of essential config variables
    required_vars = ["ARTIFACTS_DIR", "NASA_TAP_SYNC"]
    missing_vars = [var for var in required_vars if os.getenv(var) is None]

    if missing_vars:
        raise Exception(
            f"Missing essential configuration variables: {', '.join(missing_vars)}. "
            "Please re-run the configuration."
        )

    # Check if ARTIFACTS_DIR exists
    import lifefinder.config as cfg

    artifacts_dir = cfg.ARTIFACTS_DIR
    if not artifacts_dir or not Path(artifacts_dir).expanduser().exists():
        raise Exception(
            f"Artifacts directory '{artifacts_dir}' does not exist. Please re-run the configuration."
        )


if __name__ == "__main__":
    app()
