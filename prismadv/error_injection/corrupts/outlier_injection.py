import math
import random

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class OutlierInjection(TabularCorruption):
    """
    Inject statistical outliers into numerical columns.
    
    Strategies:
        - 'iqr_based': Values beyond Q3 + 1.5*IQR or below Q1 - 1.5*IQR
        - 'zscore_based': Values beyond mean ± 3*std
        - 'extreme_values': Values at min/max * factor
    """
    
    def __init__(self, columns=None, severity=None, sampling=None, 
                 strategy="iqr_based", factor=3.0, random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling,
                         random_state=random_state, **kwargs)
        self.strategy = strategy
        self.factor = factor

        valid_strategies = ["iqr_based", "zscore_based", "extreme_values"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
    
    def identify_columns(self, dataframe: pd.DataFrame):
        """Identify numerical columns suitable for outlier injection."""
        numerical_cols = []
        for col in dataframe.columns:
            if pd.api.types.is_numeric_dtype(dataframe[col]):
                # Exclude columns with very few unique values (likely categorical)
                if dataframe[col].nunique() > 10:
                    numerical_cols.append(col)
        
        if not numerical_cols:
            raise ValueError("No suitable numerical columns found for OutlierInjection.")
        
        self.columns = numerical_cols
        return numerical_cols
    
    def sample_mutable_rows(self, dataframe: pd.DataFrame):
        """Sample rows an outlier can actually land in.

        Every injector below returns NaN unchanged, but `sample_rows` draws from
        the whole frame, so on a sparse column the draw can miss every populated
        row and the corruption silently becomes a no-op. It bites harder the
        smaller the table: price_of_healthy_diet is 12% populated in some cost
        columns, where a 0.01 severity draw hit one or two live rows at 1379 rows
        and none at all at 537.

        The count still comes from the full row count, so `severity` keeps
        meaning the same share of the table; only the pool it is drawn from
        narrows. Sampling modes other than CAR order rows by value or by a
        companion column, which this cannot preserve, so they keep the inherited
        behaviour.
        """
        if not str(self.sampling).endswith("CAR"):
            return self.sample_rows(dataframe)
        columns = [column for column in self.columns if column in dataframe.columns]
        if not columns:
            return dataframe.index[:0]
        eligible = dataframe.index[dataframe[columns].notna().any(axis=1)]
        count = int(len(dataframe) * min(self.severity, 1.0))
        if len(eligible) <= count:
            return eligible
        return pd.Index(np.random.permutation(eligible)[:count])

    def transform(self, dataframe: pd.DataFrame):
        self.validate_data(dataframe)

        if not self.columns:
            self.identify_columns(dataframe)

        if self.severity is None or self.severity <= 0:
            return dataframe

        df = dataframe.copy(deep=True)
        rows = self.sample_mutable_rows(df)
        
        for col in self.columns:
            if col not in df.columns:
                raise ValueError(f"column '{col}' not in the data")

            if self.strategy == "iqr_based":
                inject = self._inject_iqr_outlier
            elif self.strategy == "zscore_based":
                inject = self._inject_zscore_outlier
            else:
                inject = self._inject_extreme_value

            # Statistics come from the clean column, not from the one being
            # rewritten, and integer columns receive integers rounded away
            # from the data so the column keeps its dtype on disk.
            reference = dataframe[col]
            integer_column = self.is_integer_column(reference)
            df.loc[rows, col] = df.loc[rows, col].apply(
                lambda x: self._as_column_type(inject(reference, x), x, integer_column)
            )

        return df

    @staticmethod
    def _as_column_type(outlier, original, integer_column):
        if not integer_column or pd.isna(outlier) or outlier == original:
            return outlier
        return math.ceil(outlier) if outlier > original else math.floor(outlier)
    
    def _inject_iqr_outlier(self, series: pd.Series, value):
        """Inject outlier based on IQR method."""
        if pd.isna(value):
            return value
        
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            return value
        
        # Randomly choose to create upper or lower outlier
        if random.random() < 0.5:
            # Upper outlier
            outlier = q3 + self.factor * iqr
        else:
            # Lower outlier
            outlier = q1 - self.factor * iqr
        
        return outlier
    
    def _inject_zscore_outlier(self, series: pd.Series, value):
        """Inject outlier based on z-score method."""
        if pd.isna(value):
            return value
        
        mean = series.mean()
        std = series.std()
        
        if std == 0 or pd.isna(std):
            return value
        
        # Randomly choose direction for outlier
        if random.random() < 0.5:
            # Positive outlier
            outlier = mean + self.factor * std
        else:
            # Negative outlier
            outlier = mean - self.factor * std
        
        return outlier
    
    def _inject_extreme_value(self, series: pd.Series, value):
        """Inject extreme outlier based on min/max values."""
        if pd.isna(value):
            return value
        
        min_val = series.min()
        max_val = series.max()
        
        if pd.isna(min_val) or pd.isna(max_val):
            return value
        
        # Randomly choose to create extreme high or low value
        if random.random() < 0.5:
            # Extreme high value
            outlier = max_val * self.factor
        else:
            # Extreme low value
            outlier = min_val * self.factor if min_val < 0 else min_val / self.factor
        
        return outlier

