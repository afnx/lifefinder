import argparse
from lifefinder.data.nasa_client import NasaExoplanetClient
from lifefinder.logger import get_logger

logger = get_logger("main")


def main(limit: int = 5, force: bool = False):
    # Fetch and display exoplanet data
    client = NasaExoplanetClient()
    df = client.fetch_exoplanets(force=force, limit=limit)
    logger.info("Fetched %d rows", len(df))

    # Print the first few rows of the dataframe
    print(df.head(limit).to_string(index=False))


# Run this command:
# python -m lifefinder.main --limit 10 --force

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()
    main(limit=args.limit, force=args.force)
