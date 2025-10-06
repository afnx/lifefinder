from pathlib import Path
import pandas as pd


def test_fetch_from_local_sample():
    # Import AFTER the conftest.py fixture has run and reloaded config
    from lifefinder.data.nasa_client import NasaExoplanetClient

    sample = Path("tests/data/sample_exoplanets.csv")
    assert sample.exists(), "Create tests/data/sample_exoplanets.csv"

    # Fetch using the client, but point to local sample file
    client = NasaExoplanetClient()
    df = client.fetch_exoplanets(cache_path=sample, force=False, limit=None)

    assert isinstance(df, pd.DataFrame)
    assert "pl_name" in df.columns
    assert df.shape[0] >= 1
