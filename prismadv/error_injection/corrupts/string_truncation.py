import random

import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class StringTruncation(TabularCorruption):
    """Cut characters off sampled string cells.

    Parameters:
        truncation_length: number of characters to remove from each sampled
            cell. When omitted the cut is a share of the string's length
            drawn around `severity`. Either way at least one character goes,
            so a sampled cell is never returned unchanged: with the old
            ratio-only rule a 3% severity on strings shorter than 20
            characters truncated nothing at all.
    """

    def __init__(self, columns=None, severity=None, sampling=None, truncation_length=None,
                 random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling,
                         random_state=random_state, **kwargs)
        if truncation_length is not None and int(truncation_length) < 1:
            raise ValueError("truncation_length must be at least 1")
        self.truncation_length = None if truncation_length is None else int(truncation_length)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

    def identify_columns(self, dataframe: pd.DataFrame):
        str_cols = [
            c for c in dataframe.columns
            if pd.api.types.is_string_dtype(dataframe[c]) or pd.api.types.is_object_dtype(dataframe[c])
        ]
        if not str_cols:
            raise ValueError("No string-like columns found for TruncationCorruption.")
        self.columns = str_cols
        return str_cols

    def transform(self, dataframe: pd.DataFrame):
        self.validate_data(dataframe)
        if not self.columns:
            self.identify_columns(dataframe)
        if self.severity is None or self.severity <= 0:
            return dataframe

        df = dataframe.copy(deep=True)
        rows = self.sample_rows(df)

        for col in self.columns:
            series = df.loc[rows, col].copy()
            series = series.apply(self._truncate_string_safe)
            df.loc[rows, col] = series

        return df

    def _truncate_string_safe(self, x):
        if pd.isna(x):
            return x
        s = str(x)
        if len(s) <= 1:
            return s

        if self.truncation_length is not None:
            n_trunc = self.truncation_length
        else:
            trunc_ratio = min(max(self.severity * random.uniform(0.5, 1.5), 0.05), 0.95)
            n_trunc = int(len(s) * trunc_ratio)
        # Never remove the whole string, never remove nothing.
        n_trunc = min(max(n_trunc, 1), len(s) - 1)

        if random.random() < 0.8:
            return s[: len(s) - n_trunc]
        else:
            return s[n_trunc:]
