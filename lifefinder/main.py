from lifefinder.data.nasa_client import NasaExoplanetClient
from lifefinder.data.preprocessing import ExoplanetDataProcessor
from lifefinder.logger import get_logger

logger = get_logger("main")


def main():
    try:
        client = NasaExoplanetClient()
        raw_df = client.fetch_exoplanets(limit=1000)
        logger.info(f"Fetched raw data with shape: {raw_df.shape}")

        clean_df = ExoplanetDataProcessor.clean_data(raw_df)
        logger.info(f"Cleaned data shape: {clean_df.shape}")

        final_df = ExoplanetDataProcessor.feature_engineering(clean_df)
        logger.info(f"Processed data shape: {final_df.shape}")

        print(final_df.head(10))
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
