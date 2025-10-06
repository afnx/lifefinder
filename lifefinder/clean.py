import pandas as pd

from lifefinder.data.cleaner import ExoplanetCleaner
from lifefinder.utils.logger import get_logger

logger = get_logger("clean")


def clean(input_csv, output_csv):
    """
    Cleans the input CSV and saves to output CSV.
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the cleaned CSV file.
    Returns: None
    """
    df = pd.read_csv(input_csv)
    original_shape = df.shape
    logger.info(f"Input data shape: {original_shape}")

    cleaner = ExoplanetCleaner()
    cleaner.fit(df)
    cleaned_df = cleaner.transform(df)
    cleaned_shape = cleaned_df.shape

    rows_removed = original_shape[0] - cleaned_shape[0]
    cols_removed = original_shape[1] - cleaned_shape[1]
    logger.info(f"Cleaning removed {rows_removed} rows and {cols_removed} columns")
    logger.info(f"Cleaned data shape: {cleaned_shape}")

    if rows_removed > 0:
        removal_rate = (rows_removed / original_shape[0]) * 100
        logger.info(f"Removal rate: {removal_rate:.1f}%")

    cleaned_df.to_csv(output_csv, index=False)
    logger.info(f"Cleaned CSV saved to {output_csv}")
