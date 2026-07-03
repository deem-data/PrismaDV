import random

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class DistributionShift(TabularCorruption):
    """
    Shift the distribution of numerical columns.
    
    Strategies:
        - 'mean_shift': Shift the mean of the distribution
        - 'variance_change': Increase or decrease variance
        - 'skewness_alteration': Make distribution more skewed
        - 'scale_change': Multiply values by a factor
    
    Use cases:
        - Detecting distribution drift in ML pipelines
        - Testing data quality monitors
        - Simulating market shifts
        - Training set vs test set distribution differences
    """
    
    def __init__(self, columns=None, severity=None, sampling=None,
                 strategy='mean_shift', shift_factor=2.0, random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.strategy = strategy
        self.shift_factor = shift_factor
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        valid_strategies = ['mean_shift', 'variance_change', 'skewness_alteration', 'scale_change']
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
    
    def identify_columns(self, dataframe: pd.DataFrame):
        """Identify numerical columns."""
        numerical_cols = []
        for col in dataframe.columns:
            if pd.api.types.is_numeric_dtype(dataframe[col]):
                numerical_cols.append(col)
        
        if not numerical_cols:
            raise ValueError("No numerical columns found for DistributionShift.")
        
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
            
            if self.strategy == 'mean_shift':
                df = self._apply_mean_shift(df, col, rows)
            elif self.strategy == 'variance_change':
                df = self._apply_variance_change(df, col, rows)
            elif self.strategy == 'skewness_alteration':
                df = self._apply_skewness_alteration(df, col, rows)
            elif self.strategy == 'scale_change':
                df = self._apply_scale_change(df, col, rows)
        
        return df
    
    def _apply_mean_shift(self, df, col, rows):
        """Shift the mean of the distribution."""
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        if pd.isna(mean_val) or pd.isna(std_val):
            return df
        
        # Shift by shift_factor standard deviations
        shift = std_val * self.shift_factor
        
        for idx in rows:
            if idx in df.index and not pd.isna(df.at[idx, col]):
                df.at[idx, col] = df.at[idx, col] + shift
        
        return df
    
    def _apply_variance_change(self, df, col, rows):
        """Increase or decrease variance."""
        mean_val = df[col].mean()
        
        if pd.isna(mean_val):
            return df
        
        # Increase or decrease variance
        factor = self.shift_factor if random.random() < 0.5 else 1.0 / self.shift_factor
        
        for idx in rows:
            if idx in df.index and not pd.isna(df.at[idx, col]):
                # Move value further/closer to mean
                diff = df.at[idx, col] - mean_val
                df.at[idx, col] = mean_val + (diff * factor)
        
        return df
    
    def _apply_skewness_alteration(self, df, col, rows):
        """Make distribution more skewed."""
        median_val = df[col].median()
        
        if pd.isna(median_val):
            return df
        
        # Push values on one side further from median
        for idx in rows:
            if idx in df.index and not pd.isna(df.at[idx, col]):
                if df.at[idx, col] > median_val:
                    # Push higher values even higher
                    df.at[idx, col] = df.at[idx, col] * self.shift_factor
                else:
                    # Push lower values even lower
                    df.at[idx, col] = df.at[idx, col] / self.shift_factor
        
        return df
    
    def _apply_scale_change(self, df, col, rows):
        """Multiply values by a factor."""
        for idx in rows:
            if idx in df.index and not pd.isna(df.at[idx, col]):
                df.at[idx, col] = df.at[idx, col] * self.shift_factor
        
        return df

