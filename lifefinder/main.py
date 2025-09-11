from lifefinder.data.nasa_client import NasaExoplanetClient
from lifefinder.data.preprocessing import build_exoplanet_pipeline
from lifefinder.logger import get_logger


logger = get_logger("main")


def main():
    try:
        client = NasaExoplanetClient()
        raw_df = client.fetch_exoplanets(limit=1000)
        logger.info(f"Fetched raw data with shape: {raw_df.shape}")

        pipeline = build_exoplanet_pipeline()
        final_array = pipeline.fit_transform(raw_df)
        logger.info(f"Processed data with shape: {final_array.shape}")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
