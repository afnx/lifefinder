"""
File utility functions for Lifefinder.
"""

import os
import re

from datetime import datetime
from typing import Optional
from pathlib import Path

from lifefinder import config as cfg


def sort_models(type: int, models_path: Optional[str] = None) -> list:
    """
    Return a list of model files sorted by F1 score (descending) or timestamp (descending).
    Follows the naming convention: model_f1-<F1>_YYYYMMDD-HHMMSS.pt

    Args:
        type (int): 1 to sort by F1 score, 2 to sort by timestamp.
        models_path (str, optional): Path to the models directory. Defaults to cfg.MODELS_DIR.

    Returns:
        list: List of dictionaries with model information.
    """
    models_dir = models_path if models_path else cfg.MODELS_DIR
    model_files = [
        f
        for f in os.listdir(models_dir)
        if f.startswith("model_") and f.endswith(".pt")
    ]
    model_info = []
    for fname in model_files:
        fpath = os.path.join(models_dir, fname)

        # Construct pipeline filename
        pipeline_fname = fname.replace("model_", "pipeline_").replace(".pt", ".pkl")
        pipeline_fpath = os.path.join(models_dir, pipeline_fname)

        # Construct metrics filename
        metrics_fname = fname.replace("model_", "metrics_").replace(".pt", ".json")
        metrics_fpath = os.path.join(models_dir, metrics_fname)

        # Extract F1 score from filename
        match_f1 = re.search(r"f1-([0-9.]+)_", fname)
        f1 = float(match_f1.group(1)) if match_f1 else 0.0

        # Extract timestamp from filename
        match_timestamp = re.search(r"_(\d{8}-\d{6})", fname)
        if match_timestamp:
            try:
                timestamp = datetime.strptime(match_timestamp.group(1), "%Y%m%d-%H%M%S")
            except ValueError:
                timestamp = None
        else:
            timestamp = None

        model_info.append(
            {
                "fname": fname,
                "fpath": fpath,
                "pipeline_fname": pipeline_fname,
                "pipeline_fpath": pipeline_fpath,
                "metrics_fname": metrics_fname,
                "metrics_fpath": metrics_fpath,
                "f1": f1,
                "timestamp": timestamp,
            }
        )

    if type == 1:
        # Sort by F1 descending
        model_info.sort(key=lambda x: x["f1"], reverse=True)
    else:
        # Sort by timestamp descending, None last
        model_info.sort(
            key=lambda x: (x["timestamp"] is not None, x["timestamp"]), reverse=True
        )

    return model_info


def validate_file(
    file_path: str | Path,
    file_name: Optional[str],
    allowed_extensions: list,
    throw: bool = True,
) -> bool:
    """
    Validate if the file exists and has an allowed extension.

    Args:
        file_path (str | Path): Path to the file.
        file_name (Optional[str]): Name of the file.
        allowed_extensions (list): List of allowed file extensions. Example: ['txt', 'csv']
        throw (bool): Whether to raise exceptions or return False on failure.

    Returns:
        True if the file is valid, False otherwise.
    """
    try:
        path_obj = Path(file_path)
        if not path_obj.is_file():
            raise FileNotFoundError(
                f"{file_name if file_name else 'File'} is not found at {file_path}."
            )
        ext = path_obj.suffix
        if ext.lower() not in [f".{e.lower().lstrip('.')}" for e in allowed_extensions]:
            raise ValueError(
                f"{file_name if file_name else 'File'} does not have an allowed extension: {ext}"
            )
    except Exception as e:
        if throw:
            raise e
        return False
    return True


def validate_directory(
    dir_path: str | Path, dir_name: Optional[str], throw: bool = True
) -> bool:
    """
    Validate if the directory exists.

    Args:
        dir_path (str | Path): Path to the directory.
        dir_name (Optional[str]): Name of the directory.
        throw (bool): Whether to raise exceptions or return False on failure.

    Returns:
        True if the directory is valid, False otherwise.
    """
    try:
        path_obj = Path(dir_path)
        if not path_obj.is_dir():
            raise NotADirectoryError(
                f"{dir_name if dir_name else 'Directory'} is not found at {dir_path}."
            )
    except Exception as e:
        if throw:
            raise e
        return False
    return True
