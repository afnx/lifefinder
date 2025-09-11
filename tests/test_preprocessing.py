from lifefinder.data.preprocessing import build_exoplanet_pipeline

import pytest
import pandas as pd
import numpy as np

def test_exoplanet_pipeline_runs():
    # Toy dataframe with minimal dummy data
    df = pd.DataFrame({
        "pl_name": ["a", "b", "c", "d", "e"],
        "host_name": ["H1", "H2", "H1", "H2", "H3"],
        "pl_orbper": [1.0, 2.0, 3.0, 4.0, 5.0],
        "pl_rade": [1.1, 2.0, 1.5, 2.5, 3.0],
        "pl_orbsmax": [0.05, 0.1, 0.2, 0.3, 0.4],
        "pl_bmasse": [1.0, 5.0, 2.0, 10.0, 8.0],
        "pl_insol": [1.0, 0.2, 1.5, 0.8, 2.0],
        "pl_eqt": [300, 250, 350, 400, 450],
        "st_teff": [5000, 5500, 6000, 5800, 5900],
        "st_mass": [1.0, 1.2, 0.9, 1.1, 1.3],
        "st_rad": [1.0, 1.1, 0.9, 1.2, 1.3],
        "sy_snum": [1, 2, 1, 2, 3],
        "sy_pnum": [1, 1, 2, 2, 3],
        "disc_year": [2000, 2001, 2002, 2003, 2004],
        "pl_radj": [0.01, 0.02, 0.015, 0.025, 0.03],
        "pl_bmassj": [0.003, 0.01, 0.004, 0.02, 0.015],
        "pl_dens": [5.5, 6.0, 5.0, 6.5, 5.8],
        "pl_orbeccen": [0.01, 0.05, 0.02, 0.1, 0.03],
        "st_met": [0.0, 0.1, -0.1, 0.05, 0.2],
        "st_logg": [4.4, 4.5, 4.3, 4.2, 4.6],
        "sy_dist": [10, 20, 15, 25, 30],
    })

    pipeline = build_exoplanet_pipeline()
    
    # Fit + transform
    X_processed = pipeline.fit_transform(df)

    # Check type
    assert isinstance(X_processed, np.ndarray)

    # Should have same number of rows
    assert X_processed.shape[0] == df.shape[0]

    # Check some engineered columns exist in the pipeline
    feature_names = [
        "orbit_star_ratio", "planet_star_mass_ratio",
        "habitable_zone_index", "relative_radius_ratio",
        "log_pl_rade", "log_pl_bmasse", "log_orbit_star_ratio"
    ]
    for col in feature_names:
        assert col in pipeline.named_steps["features"].transform(df).columns

@pytest.mark.filterwarnings("ignore:Skipping features without any observed values")
def test_pipeline_handles_missing_and_drops():
    df = pd.DataFrame({
        "pl_name": ["a", "a", "b"],
        "pl_orbper": [1.0, 1.0, np.nan],
        "pl_rade": [1.1, 1.1, 2.0],
        "st_teff": [5000, 5000, 5500],
        "st_mass": [1.0, 1.0, 1.2],
        "pl_insol": [1.0, np.nan, 0.2],
        "pl_refname": ["ref1", "ref1", "ref2"],  # Should be dropped
    })
    pipeline = build_exoplanet_pipeline()
    X_processed = pipeline.fit_transform(df)
    # Only one unique 'a' and one 'b' (but 'b' dropped due to missing pl_orbper)
    assert X_processed.shape[0] == 1

@pytest.mark.filterwarnings("ignore:Skipping features without any observed values")
def test_pipeline_raises_on_missing_required():
    df = pd.DataFrame({
        "pl_name": ["a"],
        "pl_rade": [1.1],
        "st_teff": [5000],
        "st_mass": [1.0],
        # "pl_orbper" missing!
    })
    pipeline = build_exoplanet_pipeline()
    with pytest.raises(ValueError):
        pipeline.fit_transform(df)