"""
loader.py
Loads the raw NSL-KDD train/test text files into pandas DataFrames.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_config, resolve_path


def load_nsl_kdd(config=None):
    """
    Loads KDDTrain+.txt (and KDDTest+.txt if present) using the column
    layout defined in config.yaml. If no test file exists, splits the
    training data 80/20 instead.

    Returns: (train_data, test_data) as pandas DataFrames, each containing
    the selected raw columns plus 'label' and 'difficulty_level'.
    """
    if config is None:
        config = load_config()

    train_path = resolve_path(config["paths"]["train_data"])
    test_path = resolve_path(config["paths"]["test_data"])
    columns = config["columns"]["names"]
    use_cols = config["columns"]["use_cols"]

    print(f"Loading training data from {train_path} ...")
    train_data = pd.read_csv(train_path, names=columns, header=None, usecols=use_cols)

    if os.path.exists(test_path):
        print(f"Found test file at {test_path}. Loading...")
        test_data = pd.read_csv(test_path, names=columns, header=None, usecols=use_cols)
    else:
        print("No test file found. Splitting training data 80/20 instead.")
        train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

    return train_data, test_data


def add_binary_target(df):
    """Maps the NSL-KDD 'label' column to a binary target: 0=normal, 1=threat."""
    df = df.copy()
    df["target"] = df["label"].apply(lambda x: 0 if str(x).strip() == "normal" else 1)
    return df
