import random

import numpy as np
import pandas as pd
from numpy import dtype

from prismadv.error_injection.abstract_corruption import TabularCorruption


class GaussianNoise(TabularCorruption):

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

    def identify_columns(self, dataframe):
        raise NotImplementedError("GaussianNoise corruption does not support column identification yet.")

    def transform(self, dataframe: pd.DataFrame):
        df = dataframe.copy(deep=True)
        for col in self.columns:
            self._transform_column(df, col)
        return df

    def _transform_column(self, dataframe, column):
        # Validate that column contains numeric data
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            raise ValueError(f"Column '{column}' is not numeric and cannot have Gaussian noise applied")

        stddev = np.std(dataframe[column])

        # Validate stddev is valid (not 0, NaN, or inf)
        if stddev == 0 or np.isnan(stddev) or np.isinf(stddev):
            # Skip transformation if stddev is invalid (e.g., all values are the same or column is empty)
            return dataframe

        scale = random.uniform(1, 5)

        if self.severity > 0:
            rows = self.sample_rows(dataframe)

            # Validate that rows is non-empty
            if len(rows) == 0:
                return dataframe

            noise = np.random.normal(0, scale * stddev, size=len(rows))
            if dataframe[column].dtype == dtype('int64'):
                noise = noise.astype(int)
            dataframe.loc[rows, column] += noise
        return dataframe
