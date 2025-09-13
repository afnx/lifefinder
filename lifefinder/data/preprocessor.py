from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Required to enable IterativeImputer, do not remove!  # noqa
from sklearn.experimental import enable_iterative_imputer  # noqa

_ = enable_iterative_imputer 

from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from typing import Optional
from functools import partial

from lifefinder.data.cleaner import ExoplanetCleaner
from lifefinder.data.feature_engineer import ExoplanetFeatureEngineer
from lifefinder import config as cfg


def select_present_columns(df, columns_to_select):
    """Returns a list of columns from columns_to_select that are in df."""
    return [col for col in columns_to_select if col in df.columns]


def build_exoplanet_pipeline(
        numeric_features: Optional[list] = None,
        categorical_features: Optional[list] = None
    ):
    """
    Builds a machine learning pipeline for exoplanet data.
    """

    numeric_features = numeric_features or [
        # Core features
        "sy_snum",
        "sy_pnum",
        "disc_year",
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
        "st_teff",
        "st_rad",
        "st_mass",
        "st_met",
        "st_logg",
        "sy_dist",
    ]

    numeric_features += [
        # Engineered numeric features
        "orbit_star_ratio",
        "planet_star_mass_ratio",
        # "habitable_zone_index", This is a target variable, should not be included in features
        "relative_radius_ratio",
        "log_pl_rade",
        "log_pl_bmasse",
        "log_orbit_star_ratio",
    ]

    categorical_features = categorical_features or [
        # Categorical features
        "host_name",
        "discoverymethod",
        "disc_facility",
        "st_spectype",
        "rastr",
        "decstr",
    ]

    numeric_transformer = Pipeline([
        ("imputer", IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=10, random_state=42),
            max_iter=10,
            random_state=42
        )),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = [
        ("num", numeric_transformer, partial(select_present_columns, columns_to_select=numeric_features)),
        ("cat", cat_transformer, partial(select_present_columns, columns_to_select=categorical_features)),
    ]

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    pipeline = Pipeline([
        ("cleaning", ExoplanetCleaner()),
        ("features", ExoplanetFeatureEngineer(hz_sigma=cfg.TRAINING_CONFIG["hz_sigma"])),
        ("preprocessor", preprocessor)
    ])

    return pipeline
