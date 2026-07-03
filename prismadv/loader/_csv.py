import os
from typing import List

import pandas as pd


def load_csvs(dir_path: str) -> List[pd.DataFrame]:
    """Load a list of CSV files into a list of pandas DataFrames."""
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")

    # Filter for CSV files only
    csv_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.csv')]

    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {dir_path}")

    file_paths = [os.path.join(dir_path, file) for file in csv_files]

    dataframes = []
    for file_path in file_paths:
        try:
            df = load_csv(file_path)
            dataframes.append(df)
        except Exception as e:
            raise ValueError(f"Failed to load CSV file '{file_path}': {e}") from e

    return dataframes


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    try:
        return pd.read_csv(file_path, **kwargs)
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse CSV file '{file_path}': {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to read CSV file '{file_path}': {e}") from e
