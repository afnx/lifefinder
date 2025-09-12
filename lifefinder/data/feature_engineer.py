from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ExoplanetFeatureEngineer(TransformerMixin, BaseEstimator):
    """
    Adds custom engineered features to the exoplanet dataset:
    - orbit_star_ratio: Ratio of planet's orbital semi-major axis to star radius.
    - planet_star_mass_ratio: Ratio of planet mass to star mass.
    - habitable_zone_index: Gaussian-based index indicating potential habitability based on stellar insolation
    - relative_radius_ratio: Ratio of planet radius to star radius normalized to Earth-Sun scale.
    - Optional log transforms of select features.
    """

    def __init__(self, hz_sigma=0.5, log_transform=True):
        self.hz_sigma = hz_sigma
        self.log_transform = log_transform

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Add orbit to star radius ratio
        X = self._add_orbit_star_ratio(X)

        # Add planet-star mass ratio
        X = self._add_planet_star_mass_ratio(X)

        # Add habitable zone index
        X = self._add_habitable_zone_index(X)

        # Add relative radius ratio
        X = self._add_relative_radius_ratio(X)

        # Optional log features
        if self.log_transform:
            for col in ["pl_rade", "pl_bmasse", "orbit_star_ratio"]:
                if col in X:
                    X[f"log_{col}"] = np.log1p(X[col].clip(lower=0))
        return X

    def _add_orbit_star_ratio(self, X):
        # Convert st_rad from Solar Radius to AU (1 Solar Radius ≈ 0.00465047 AU)
        if "pl_orbsmax" in X.columns and "st_rad" in X.columns:
            st_rad_au = X["st_rad"] * 0.00465047
            X["orbit_star_ratio"] = X["pl_orbsmax"] / st_rad_au.replace(0, np.nan)
        return X

    def _add_planet_star_mass_ratio(self, X):
        # Convert st_mass from Solar mass to Earth mass (1 Solar mass ≈ 333030 Earth masses)
        if "pl_bmasse" in X.columns and "st_mass" in X.columns:
            st_mass_earth = X["st_mass"] * 333030
            X["planet_star_mass_ratio"] = X["pl_bmasse"] / st_mass_earth.replace(0, np.nan)
        return X

    def _add_habitable_zone_index(self, X):
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
        if "pl_insol" in X.columns:
            X["habitable_zone_index"] = X["pl_insol"].apply(
                lambda insol: np.exp(-0.5 * ((insol - 1.0) / self.hz_sigma) ** 2)
                if np.isfinite(insol) else np.nan
            )
        return X

    def _add_relative_radius_ratio(self, X):
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
        if "pl_rade" in X.columns and "st_rad" in X.columns:
            X["relative_radius_ratio"] = X["pl_rade"] / (X["st_rad"] * 109.076).replace(0, np.nan)
        return X
    
    def __repr__(self, N_CHAR_MAX: int = 700):
        return f"ExoplanetFeatureEngineer(hz_sigma={self.hz_sigma}, log_transform={self.log_transform})"
