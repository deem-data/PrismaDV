import random

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class SeasonalityAnomaly(TabularCorruption):
    """
    Inject values that violate expected seasonal patterns in time-series data.
    
    Parameters:
        time_column: Name of the time/date column
        value_column: Name of the column with values to corrupt
        anomaly_type: Type of anomaly to inject
            - 'spike': Sudden increase
            - 'drop': Sudden decrease
            - 'inversion': Invert the pattern
            - 'shift': Phase shift in pattern
    
    Use cases:
        - Unusual sales spikes in off-season
        - Temperature anomalies
        - Traffic pattern violations
        - Energy consumption anomalies
    """
    
    def __init__(self, columns=None, severity=None, sampling=None,
                 time_column=None, anomaly_type='spike',
                 anomaly_factor=3.0, random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.time_column = time_column
        self.anomaly_type = anomaly_type
        self.anomaly_factor = anomaly_factor
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        valid_types = ['spike', 'drop', 'inversion', 'shift']
        if self.anomaly_type not in valid_types:
            raise ValueError(f"anomaly_type must be one of {valid_types}")
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
    
    def identify_columns(self, dataframe: pd.DataFrame):
        """Identify numerical columns for seasonality corruption."""
        numerical_cols = []
        for col in dataframe.columns:
            if pd.api.types.is_numeric_dtype(dataframe[col]):
                # Exclude time-related columns
                if col != self.time_column:
                    numerical_cols.append(col)
        
        if not numerical_cols:
            raise ValueError("No numerical columns found for SeasonalityAnomaly.")
        
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
            
            # Get statistics for the column
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
                continue
            
            if self.anomaly_type == 'spike':
                # Create sudden spikes
                for idx in rows:
                    if idx in df.index and not pd.isna(df.at[idx, col]):
                        df.at[idx, col] = df.at[idx, col] + (std_val * self.anomaly_factor)
            
            elif self.anomaly_type == 'drop':
                # Create sudden drops
                for idx in rows:
                    if idx in df.index and not pd.isna(df.at[idx, col]):
                        df.at[idx, col] = df.at[idx, col] - (std_val * self.anomaly_factor)
            
            elif self.anomaly_type == 'inversion':
                # Invert values around mean
                for idx in rows:
                    if idx in df.index and not pd.isna(df.at[idx, col]):
                        diff = df.at[idx, col] - mean_val
                        df.at[idx, col] = mean_val - diff
            
            elif self.anomaly_type == 'shift':
                # Add/subtract mean to shift the pattern
                shift_amount = mean_val * random.choice([-1, 1])
                for idx in rows:
                    if idx in df.index and not pd.isna(df.at[idx, col]):
                        df.at[idx, col] = df.at[idx, col] + shift_amount
        
        return df

