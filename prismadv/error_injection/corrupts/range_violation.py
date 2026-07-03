import random

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class RangeViolation(TabularCorruption):
    """
    Inject values outside valid ranges.
    
    Parameters:
        min_value: Minimum valid value for the column(s)
        max_value: Maximum valid value for the column(s)
        strategy: How to violate the range
            - 'below_min': Generate values below minimum
            - 'above_max': Generate values above maximum
            - 'random': Randomly choose below_min or above_max
    
    Examples:
        - Negative ages
        - Percentages > 100 or < 0
        - Temperatures below absolute zero
        - Future dates (when not allowed)
    """

    def __init__(self, columns=None, severity=None, sampling=None,
                 min_value=None, max_value=None, strategy="random",
                 violation_factor=1.5, random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.strategy = strategy
        self.violation_factor = violation_factor  # How far outside range to go
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        valid_strategies = ["below_min", "above_max", "random"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

    def identify_columns(self, dataframe: pd.DataFrame):
        """Identify numerical columns suitable for range violations."""
        numerical_cols = []
        for col in dataframe.columns:
            if pd.api.types.is_numeric_dtype(dataframe[col]):
                numerical_cols.append(col)

        if not numerical_cols:
            raise ValueError("No numerical columns found for RangeViolation.")

        self.columns = numerical_cols
        return numerical_cols

    def transform(self, dataframe: pd.DataFrame):
        self.validate_data(dataframe)

        if not self.columns:
            self.identify_columns(dataframe)

        if self.severity is None or self.severity <= 0:
            return dataframe

        df = dataframe.copy(deep=True)
        rows = self.sample_rows(df)

        for col in self.columns:
            if col not in df.columns:
                continue

            # Determine range if not specified
            col_min = self.min_value if self.min_value is not None else df[col].min()
            col_max = self.max_value if self.max_value is not None else df[col].max()

            # Apply range violation to sampled rows
            df.loc[rows, col] = df.loc[rows, col].apply(
                lambda x: self._violate_range(x, col_min, col_max)
            )

        return df

    def _violate_range(self, value, col_min, col_max):
        """Generate a value that violates the range."""
        if pd.isna(value):
            return value

        if pd.isna(col_min) or pd.isna(col_max):
            return value

        range_span = col_max - col_min
        if range_span == 0:
            range_span = abs(col_max) if col_max != 0 else 1

        # Determine which side to violate
        if self.strategy == "below_min":
            violation_side = "below"
        elif self.strategy == "above_max":
            violation_side = "above"
        else:  # random
            violation_side = "below" if random.random() < 0.5 else "above"

        if violation_side == "below":
            # Generate value below minimum
            violation_amount = range_span * self.violation_factor * random.uniform(0.1, 1.0)
            violated_value = col_min - violation_amount
        else:
            # Generate value above maximum
            violation_amount = range_span * self.violation_factor * random.uniform(0.1, 1.0)
            violated_value = col_max + violation_amount

        return violated_value
