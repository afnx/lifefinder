from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class ExoplanetCleaner(TransformerMixin, BaseEstimator):
    """
    Cleans raw NASA Exoplanet dataset:
    - Removes unused flag/reference columns
    - Handles missing values for numerical & categorical columns
    """

    def __init__(self, required_cols=None, fill_cols=None, drop_cols=None):
        self.required_cols = required_cols or ["pl_name", "pl_orbper", "pl_rade", "st_teff", "st_mass"]
        self.fill_cols = fill_cols or ["pl_eqt", "pl_insol"]
        self.drop_cols = drop_cols or ["pl_refname", "disc_refname", "disc_pubdate"]

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("ExoplanetCleaner requires a pandas DataFrame.")
        self.fill_values_ = {c: X[c].median() for c in self.fill_cols if c in X}
        return self

    def transform(self, X):
        if not hasattr(self, "fill_values_"):
            raise RuntimeError("ExoplanetCleaner must be fitted before transform.")
        
        X = X.copy()

        # Drop unused reference/flag cols if present
        X = X.drop(columns=[c for c in self.drop_cols if c in X], errors="ignore")

        # Check required cols
        missing = [c for c in self.required_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Drop duplicates
        X = X.groupby("pl_name", as_index=False).agg(lambda x: x.dropna().iloc[0] if x.notna().any() else None)

        # Drop critical missing
        X = X.dropna(subset=["pl_orbper", "pl_rade", "st_teff", "st_mass"])

        # Fill non-critical with training median
        for col, val in self.fill_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)

        return X

    def __repr__(self, N_CHAR_MAX: int = 700):
        return f"ExoplanetCleaner(required={self.required_cols}, fill={self.fill_cols}, drop={self.drop_cols})"
