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
    cleaner = ExoplanetCleaner()
    cleaner.fit(df)
    cleaned_df = cleaner.transform(df)
    cleaned_df.to_csv(output_csv, index=False)
    logger.info(f"Cleaned CSV saved to {output_csv}")
