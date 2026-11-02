import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class Scaling(TabularCorruption):
    """Multiply sampled numeric cells by a power of ten.

    Parameters:
        factor: the multiplier. When omitted one of 10, 100, 1000 is drawn.
    """

    def __init__(self, columns=None, severity=None, sampling=None, factor=None, random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling,
                         random_state=random_state, **kwargs)
        self.factor = factor

    def identify_columns(self, dataframe):
        raise NotImplementedError("Scaling corruption does not support column identification yet.")

    def transform(self, dataframe: pd.DataFrame):
        df = dataframe.copy(deep=True)

        columns = self._target_columns()
        non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            # `*=` on an object column repeats the strings instead of scaling.
            raise ValueError(f"Scaling needs numeric columns; {non_numeric} are not")

        scale_factor = self.factor if self.factor is not None else int(np.random.choice([10, 100, 1000]))

        rows = self.sample_rows(dataframe)
        if float(scale_factor) != int(scale_factor):
            # A fractional factor turns an integer column into floats; cast
            # first, pandas no longer upcasts on assignment.
            for column in columns:
                if pd.api.types.is_integer_dtype(df[column]):
                    df[column] = df[column].astype(float)
        df.loc[rows, columns] *= scale_factor

        return df
