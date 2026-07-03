import random

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class StringTruncation(TabularCorruption):
    def __init__(self, columns=None, severity=None, sampling=None, random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

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

        trunc_ratio = min(max(self.severity * random.uniform(0.5, 1.5), 0.05), 0.95)
        n_trunc = int(len(s) * trunc_ratio)

        if random.random() < 0.8:
            return s[: max(1, len(s) - n_trunc)]
        else:
            return s[n_trunc:]
