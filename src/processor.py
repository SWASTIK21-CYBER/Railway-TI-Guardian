"""
processor.py
Builds and applies the feature preprocessing pipeline
(one-hot encoding for categoricals, scaling for numericals).
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils import load_config


def build_preprocessor(config=None):
    """Creates a ColumnTransformer based on config.yaml column definitions."""
    if config is None:
        config = load_config()

    categorical_cols = config["columns"]["categorical"]
    numerical_cols = config["columns"]["numerical"]

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ]
    )


def split_features_target(df):
    """Drops label columns, returns (X, y)."""
    X = df.drop(columns=["label", "difficulty_level", "target"])
    y = df["target"]
    return X, y
