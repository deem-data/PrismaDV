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
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.strategy = strategy
        self.factor = factor
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
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
            
            # Apply outlier injection to sampled rows
            if self.strategy == "iqr_based":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._inject_iqr_outlier(df[col], x)
                )
            elif self.strategy == "zscore_based":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._inject_zscore_outlier(df[col], x)
                )
            elif self.strategy == "extreme_values":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._inject_extreme_value(df[col], x)
                )
        
        return df
    
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

