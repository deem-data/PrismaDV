import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class MaskValues(TabularCorruption):
    """Replace sampled cells with a missing-value marker.

    Integer columns get a genuine missing value, every other column one of the
    textual markers. The integer column is left numeric (pandas upcasts it to
    float for the NaN) so a numeric corruption can still follow it in the same
    config; the manager casts it back to a nullable integer before saving, so
    the untouched rows are written as `0`, not `0.0`.
    """

    def identify_columns(self, dataframe):
        raise NotImplementedError("MaskValues corruption does not support column identification yet.")

    def transform(self, dataframe: pd.DataFrame):
        df = dataframe.copy(deep=True)
        for col in self._target_columns():
            if pd.api.types.is_numeric_dtype(dataframe[col]) and self.is_integer_column(dataframe[col]):
                mask_value = np.nan
                df[col] = df[col].astype(float)
            else:
                mask_value = np.random.choice(['?', 'NA', 'missing'])
                # A string marker needs an object column; pandas no longer
                # upcasts a float column for it implicitly.
                df[col] = df[col].astype(object)
            rows = self.sample_rows(dataframe)
            df.loc[rows, col] = mask_value
        return df
