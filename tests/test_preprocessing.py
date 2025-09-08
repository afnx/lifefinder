import pandas as pd
import pytest
from lifefinder.data.preprocessing import ExoplanetDataProcessor

def test_clean_data_drops_rows_with_nulls():
    df = pd.DataFrame({
        "pl_name": ["a", "b"],
        "pl_orbper": [1.0, None],
        "pl_rade": [1.1, 2.0],
        "st_teff": [5000, None],
        "st_mass": [1.0, 0.8],
    })
    cleaned = ExoplanetDataProcessor.clean_data(df)
    assert isinstance(cleaned, pd.DataFrame)
    assert "b" not in cleaned["pl_name"].values

def test_clean_data_raises_on_missing_columns():
    df = pd.DataFrame({"pl_name": ["a"]})
    with pytest.raises(ValueError):
        ExoplanetDataProcessor.clean_data(df)

def test_feature_engineering_adds_features():
    df = pd.DataFrame({
        "pl_name": ["a", "b"],
        "pl_orbper": [1.0, 2.0],
        "pl_rade": [1.1, 2.0],
        "pl_orbsmax": [1.0, 0.5],
        "st_teff": [5000, 6000],
        "st_mass": [1.0, 2.0],
        "st_rad": [1.0, 2.0],
        "pl_bmasse": [1.0, 5.0],
        "pl_insol": [1.0, 0.2],
    })
    engineered = ExoplanetDataProcessor.feature_engineering(df)
    assert "orbit_star_ratio" in engineered.columns
    assert "planet_star_mass_ratio" in engineered.columns
    assert "habitable_zone_index" in engineered.columns
    assert "orbit_star_ratio" in engineered.columns

