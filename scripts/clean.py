import pandas as pd

from lifefinder.data.cleaner import ExoplanetCleaner
from lifefinder.utils.file_utils import validate_file
from lifefinder.logger import get_logger

logger = get_logger("train")


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


if __name__ == "__main__":
    try:
        logger.info("Lifefinder Data Cleaning Script")
        logger.info("===============================")

        input_file = input(
            "Path to the input CSV file (e.g., /home/user/exoplanets.csv): "
        ).strip()

        logger.info(f"Input file selected: {input_file}")

        # Validate input file
        validate_file(input_file, "input file", ["csv"])

        output_file = input(
            "Path to save the cleaned CSV file (e.g., /home/user/cleaned_exoplanets.csv): "
        ).strip()

        logger.info(f"Output file selected: {output_file}")

        # Validate output path
        if not output_file.lower().endswith(".csv"):
            logger.error("The output file must be a .csv file.")
            exit(1)

        # Perform cleaning
        clean(input_file, output_file)
    except KeyboardInterrupt:
        print("\n")
        logger.warning("Cleaning interrupted by user.")
    except Exception as e:
        logger.error(f"{e}", exc_info=True)
