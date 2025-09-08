import pandas as pd
import numpy as np

class ExoplanetDataProcessor:
    """
    Class to preprocess NASA exoplanet dataset for ML tasks.
    """

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean NASA exoplanet dataset:
        - Drop duplicates
        - Handle missing values (NaNs)
        - Convert units if needed
        - Select only useful features
        """

        # Ensure required columns exist
        required_cols = ["pl_name", "pl_orbper", "pl_rade", "st_teff", "st_mass"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()
        
        # Aggregate by pl_name, keeping the first non-null value for each column
        df = df.groupby("pl_name", as_index=False).agg(lambda x: x.dropna().iloc[0] if x.notna().any() else None)

        # Drop rows missing critical values
        df = df.dropna(subset=["pl_orbper", "pl_rade", "st_teff", "st_mass"])

        # Fill non-critical NaNs with mean or -1 flag
        for col in ["pl_eqt", "pl_insol"]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())

        return df

    @staticmethod
    def _add_orbit_star_ratio(df: pd.DataFrame) -> pd.DataFrame:
        # Convert st_rad from Solar Radius to AU (1 Solar Radius ≈ 0.00465047 AU)
        if "pl_orbsmax" in df.columns and "st_rad" in df.columns:
            st_rad_au = df["st_rad"] * 0.00465047
            df["orbit_star_ratio"] = df["pl_orbsmax"] / st_rad_au
        return df

    @staticmethod
    def _add_planet_star_mass_ratio(df: pd.DataFrame) -> pd.DataFrame:
        # Convert st_mass from Solar mass to Earth mass (1 Solar mass ≈ 333030 Earth masses)
        if "pl_bmasse" in df.columns and "st_mass" in df.columns:
            st_mass_earth = df["st_mass"] * 333030
            df["planet_star_mass_ratio"] = df["pl_bmasse"] / st_mass_earth
        return df
    

    @staticmethod
    def _add_habitable_zone_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a 'habitable_zone_index' feature based on stellar insolation flux (pl_insol),
        using a soft Gaussian decay around Earth's flux (1.0 S⊕).

        - pl_insol: stellar flux relative to Earth (S⊕ units).
        - Earth = 1.0 S⊕ (center).
        - sigma: controls how quickly score decays away from Earth.
        Default = 0.5 (≈ half-width of conservative HZ).

        Formula:
            HZ index = exp( -0.5 * ((pl_insol - 1.0) / sigma)^2 )
        """
        if "pl_insol" in df.columns:
            sigma = 0.5  # width of the "habitable" peak
            df["habitable_zone_index"] = df["pl_insol"].apply(
                lambda insol: np.exp(-0.5 * ((insol - 1.0) / sigma) ** 2)
                if pd.notna(insol) else np.nan
            )
        return df
    

    @staticmethod
    def _add_relative_radius_ratio(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a 'relative_radius_ratio' feature:
        Compares planet radius (Earth radii) to star radius (Solar radii),
        normalized to Earth-Sun scale.

        Formula:
            relative_radius_ratio = pl_rade / (st_rad * 109.076)

        Where:
        - pl_rade: planet radius in Earth radii
        - st_rad: star radius in Solar radii
        - 109.076: ratio of Sun radius to Earth radius

        Interpretation:
        - Earth-Sun system → ratio ≈ 1 / 109.076 ≈ 0.00917
        - Larger planets or smaller stars → higher ratio
        - Smaller planets or larger stars → lower ratio
        """
        if "pl_rade" in df.columns and "st_rad" in df.columns:
            df["relative_radius_ratio"] = df["pl_rade"] / (df["st_rad"] * 109.076)
        return df


    @staticmethod
    def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features useful for ML.
        - Orbit to star radius ratio
        - Planet to star mass ratio
        """
        # Semi-major axis / star radius (proxy for HZ positioning)
        df = ExoplanetDataProcessor._add_orbit_star_ratio(df)

        # Mass ratio
        df = ExoplanetDataProcessor._add_planet_star_mass_ratio(df)

        # Habitable zone index
        df = ExoplanetDataProcessor._add_habitable_zone_index(df)

        # Relative radius ratio
        df = ExoplanetDataProcessor._add_relative_radius_ratio(df)

        return df
