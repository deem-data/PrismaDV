import random

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class TemporalGaps(TabularCorruption):
    """
    Introduce gaps in time-series data by removing consecutive records.
    
    Parameters:
        time_column: Name of the time/date column (if None, will try to identify)
        gap_size: How many consecutive records to remove (can be 'small', 'medium', 'large', or int)
    
    Use cases:
        - Missing sensor readings in IoT data
        - Dropped records in event logs
        - Service outages in monitoring data
        - Missing transactions in financial data
    """
    
    def __init__(self, columns=None, severity=None, sampling=None,
                 time_column=None, gap_size='medium', random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.time_column = time_column
        self.gap_size = gap_size
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        # Map gap size to number of rows
        self.gap_size_map = {
            'small': (2, 5),
            'medium': (5, 15),
            'large': (15, 50),
        }
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
    
    def identify_columns(self, dataframe: pd.DataFrame):
        """
        This corruption operates on the entire dataframe (removes rows),
        but we need to identify the time column.
        """
        if self.time_column and self.time_column in dataframe.columns:
            return [self.time_column]
        
        # Try to find time/date columns
        for col in dataframe.columns:
            if pd.api.types.is_datetime64_any_dtype(dataframe[col]):
                self.time_column = col
                return [col]
        
        raise ValueError(
            "No time column found. Please specify time_column parameter for TemporalGaps."
        )
    
    def transform(self, dataframe: pd.DataFrame):
        self.validate_data(dataframe)
        
        if not self.time_column:
            self.identify_columns(dataframe)
        
        if self.time_column not in dataframe.columns:
            raise ValueError(f"time_column '{self.time_column}' not found in dataframe")
        
        if self.severity is None or self.severity <= 0:
            return dataframe
        
        df = dataframe.copy(deep=True)
        
        # Sort by time column to ensure temporal order
        df = df.sort_values(by=self.time_column).reset_index(drop=True)
        
        # Calculate how many gaps to create based on severity
        total_rows = len(df)
        num_gaps = max(1, int(total_rows * self.severity * 0.1))
        
        # Create gaps
        rows_to_remove = set()
        
        for _ in range(num_gaps):
            # Determine gap size
            if isinstance(self.gap_size, int):
                gap = self.gap_size
            elif self.gap_size in self.gap_size_map:
                gap_min, gap_max = self.gap_size_map[self.gap_size]
                gap = random.randint(gap_min, gap_max)
            else:
                gap = random.randint(5, 15)  # Default medium
            
            # Select random starting position for gap
            if len(df) - gap > 0:
                start_pos = random.randint(0, len(df) - gap)
                # Add consecutive rows to removal set
                for i in range(start_pos, min(start_pos + gap, len(df))):
                    rows_to_remove.add(i)
        
        # Remove the rows
        df = df.drop(index=list(rows_to_remove)).reset_index(drop=True)
        
        return df

