"""
lifefinder.data.nasa_client
Simple client to fetch exoplanet table (NASA Exoplanet Archive TAP sync CSV).
Caches a CSV to data/raw/exoplanets.csv by default.
"""

from pathlib import Path
import requests
import pandas as pd
from typing import Optional
from lifefinder.config import RAW_DIR, DEFAULT_EXOPLANETS_CSV, NASA_TAP_SYNC
from lifefinder.logger import get_logger

logger = get_logger("nasa_client")


class NasaExoplanetClient:
    """
    Client to interact with the NASA Exoplanet Archive.
    """

    # Columns
    DEFAULT_COLUMNS = [
        # Identifiers
        "pl_name",
        "hostname",
        # System composition
        "sy_snum",
        "sy_pnum",
        # Discovery info
        "discoverymethod",
        "disc_year",
        "disc_facility",
        # Planetary parameters
        "pl_orbper",
        "pl_orbsmax",
        "pl_rade",
        "pl_radj",
        "pl_bmasse",
        "pl_bmassj",
        "pl_dens",
        "pl_orbeccen",
        "pl_insol",
        "pl_eqt",
        # Stellar parameters
        "st_teff",
        "st_rad",
        "st_mass",
        "st_met",
        "st_logg",
        "st_spectype",
        # System data
        "sy_dist",
        "rastr",
        "decstr",
    ]

    def __init__(self):
        self.session = requests.Session()

    def _build_query(self, limit=None, columns=None):
        cols = columns or self.DEFAULT_COLUMNS
        # simple SELECT query against the 'ps' (planetary systems) table
        sel = ",".join(cols)
        q = f"select+top+{limit}+{sel}+from+ps" if limit else f"select+{sel}+from+ps"
        return q

    def fetch_exoplanets(
        self,
        cache_path: Optional[Path] = None,
        force: bool = False,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Returns a pandas DataFrame of exoplanets.
        - If cache exists and force is False -> reads local CSV
        - Otherwise downloads from NASA TAP sync endpoint and saves CSV
        """
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = Path(cache_path or DEFAULT_EXOPLANETS_CSV)

        # Helper to load CSV from cache and handle empty files
        def process_cache(err_msg: str) -> pd.DataFrame:
            if not cache_path.exists() or cache_path.stat().st_size == 0:
                logger.info(err_msg)
                return pd.DataFrame()
            df = pd.read_csv(cache_path)
            return df.head(limit) if limit else df

        if cache_path.exists() and not force:
            logger.info("Loading exoplanets from cache: %s", cache_path)
            return process_cache(err_msg="Cache file is empty. Cannot load data.")

        # build TAP query and request CSV
        query = self._build_query(limit=limit or 1000)
        # build URL by manual quoting (TAP sync expects 'query' param)
        url = f"{NASA_TAP_SYNC}?query={query}&format=csv"

        logger.info("Requesting NASA Exoplanet Archive: %s", url)

        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            logger.exception("Failed to fetch from NASA Exoplanet Archive: %s", e)
            # if a partial cache exists try to load it
            if cache_path.exists():
                logger.info("Falling back to existing cache.")
                return process_cache(err_msg="Cache file is empty. Cannot load data.")
            raise

        # Save CSV to cache
        with open(cache_path, "wb") as fh:
            fh.write(resp.content)
        logger.info("Saved exoplanet CSV to %s", cache_path)

        return process_cache(err_msg="Downloaded CSV is empty. No data available.")
