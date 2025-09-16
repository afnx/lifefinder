from typing import Tuple, Any

"""
Utility functions for command-line interface interactions.
"""


def prompt_with_default(prompt_text: str, default) -> Any:
    """
    Prompts the user for input, providing a default value if no input is given.

    Args:
        prompt_text (str): The text to display to the user.
        default: The default value to return if the user provides no input.

    Returns:
        The user's input or the default value.
    """
    user_input = input(f"{prompt_text} [{default}]: ").strip()
    return default if user_input == "" else user_input


def prompt_model_selection(fu: Any, cfg: Any, logger: Any) -> Tuple[str, str, str]:
    """
    Prompts the user to select a model from available models and optionally display training metrics.

    Args:
        fu: The file utilities module.
        cfg: The configuration module.
        logger: The logger instance.

    Returns:
        tuple: (model_file, pipeline_file, metrics_file)
    """

    # Ask user how to sort available saved models
    print("\nHow would you like to sort the available saved models?")
    print("1. Sort by best F1 score (descending)")
    print("2. Sort by most recent (time created, descending)")
    sort_choice = input("Enter 1 or 2: ").strip()

    # List available model files in the default model directory
    models_dir = cfg.MODELS_DIR
    if not fu.validate_directory(models_dir, "models directory", throw=False):
        logger.error(f"Model directory not found: {models_dir}")
        exit(1)

    if sort_choice == "1" or sort_choice == "2":
        model_info = fu.sort_models(int(sort_choice))
    else:
        logger.warning("Invalid choice, defaulting to most recent.")
        model_info = fu.sort_models(2)

    if not model_info or len(model_info) == 0:
        logger.error(
            f"No model files found in directory: {models_dir}. "
            "Please run training first to generate models."
        )
        exit(1)

    print("\nAvailable models:")
    for idx, info in enumerate(model_info):
        f1_str = f"{info['f1']:.3f}" if info["f1"] is not None else "N/A"
        if info.get("timestamp") is not None:
            time_str = info["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        else:
            time_str = "N/A"
        print(f"{idx + 1}. {info['fname']} | F1: {f1_str} | Created: {time_str}")

    model_idx = input("Select a model by number: ").strip()
    try:
        model_idx = int(model_idx) - 1
        if not (0 <= model_idx < len(model_info)):
            raise ValueError
    except ValueError:
        logger.error("Invalid model selection.")
        exit(1)

    selected = model_info[model_idx]
    model_file = selected["fpath"]
    pipeline_file = selected["pipeline_fpath"]
    metrics_file = selected["metrics_fpath"]

    logger.info(f"Selected model: {model_file}")
    logger.info(f"Using pipeline: {pipeline_file}")

    return model_file, pipeline_file, metrics_file


def display_model_metrics(metrics_file: str, fu: Any, logger: Any) -> None:
    """
    Display training metrics for a given metrics file.

    Args:
        metrics_file (str): Path to the metrics JSON file.
        fu: The file utilities module.
        logger: The logger instance.

    Returns:
        None
    """

    if fu.validate_file(metrics_file, "metrics file", ["json"], throw=False):
        try:
            import json

            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
            logger.info(f"Training metrics from {metrics_file}:")
            for entry in metrics_data:
                if "config" in entry:
                    config_str = json.dumps(entry["config"], indent=2)
                    logger.info(f"Training configuration:\n{config_str}")
                if "epoch" in entry:
                    logger.info(
                        f"Epoch {entry['epoch']}: Loss={entry['loss']:.4f}, "
                        f"Acc={entry['accuracy']:.3f}, F1={entry['f1']:.3f}"
                    )
        except Exception as e:
            logger.error(f"Failed to read metrics file: {e}")
    else:
        logger.warning(f"Metrics file not found: {metrics_file}")
