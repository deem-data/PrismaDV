import random

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class TemporalOutOfOrder(TabularCorruption):
    """
    Shuffle timestamps to break temporal ordering.
    
    Strategies:
        - 'random_shuffle': Completely randomize the order of timestamps
        - 'local_swaps': Swap adjacent or nearby timestamps
        - 'block_reordering': Reorder entire blocks of time
        - 'reverse': Reverse the temporal order
    
    Use cases:
        - Time-series data where ordering matters
        - Event logs that should be chronological
        - Transaction histories
        - Sensor data streams
    """
    
    def __init__(self, columns=None, severity=None, sampling=None,
                 strategy="local_swaps", swap_distance=5, random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.strategy = strategy
        self.swap_distance = swap_distance  # For local_swaps strategy
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        valid_strategies = ["random_shuffle", "local_swaps", "block_reordering", "reverse"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
    
    def identify_columns(self, dataframe: pd.DataFrame):
        """Identify datetime/timestamp columns."""
        temporal_cols = []
        
        for col in dataframe.columns:
            # Check if column is datetime type
            if pd.api.types.is_datetime64_any_dtype(dataframe[col]):
                temporal_cols.append(col)
            # Check for numeric columns that might be timestamps
            elif pd.api.types.is_numeric_dtype(dataframe[col]):
                # Check if values look like Unix timestamps
                sample = dataframe[col].dropna().head(100)
                if len(sample) > 0:
                    # Unix timestamps are typically large integers
                    if sample.min() > 1000000000 and sample.max() < 9999999999:
                        temporal_cols.append(col)
        
        if not temporal_cols:
            raise ValueError("No temporal/datetime columns found for TemporalOutOfOrder.")
        
        self.columns = temporal_cols
        return temporal_cols
    
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
            
            if self.strategy == "random_shuffle":
                df = self._random_shuffle(df, col, rows)
            elif self.strategy == "local_swaps":
                df = self._local_swaps(df, col, rows)
            elif self.strategy == "block_reordering":
                df = self._block_reordering(df, col, rows)
            elif self.strategy == "reverse":
                df = self._reverse_order(df, col, rows)
        
        return df
    
    def _random_shuffle(self, df, col, rows):
        """Completely randomize the order of values in sampled rows."""
        # Get values from sampled rows
        values = df.loc[rows, col].copy()
        
        # Shuffle them
        shuffled_values = values.sample(frac=1.0, random_state=self.random_state).values
        
        # Assign back
        df.loc[rows, col] = shuffled_values
        
        return df
    
    def _local_swaps(self, df, col, rows):
        """Swap adjacent or nearby timestamps."""
        rows_list = list(rows)
        
        # For each sampled row, try to swap with a nearby row
        for i, idx in enumerate(rows_list):
            if idx not in df.index:
                continue
            
            # Find a nearby row to swap with
            # Look within swap_distance positions
            potential_swaps = []
            for j in range(max(0, i - self.swap_distance), 
                          min(len(rows_list), i + self.swap_distance + 1)):
                if j != i and rows_list[j] in df.index:
                    potential_swaps.append(rows_list[j])
            
            if potential_swaps:
                swap_with = random.choice(potential_swaps)
                # Swap values
                temp = df.at[idx, col]
                df.at[idx, col] = df.at[swap_with, col]
                df.at[swap_with, col] = temp
        
        return df
    
    def _block_reordering(self, df, col, rows):
        """Reorder entire blocks of consecutive rows."""
        rows_list = sorted(list(rows))
        
        if len(rows_list) < 2:
            return df
        
        # Divide rows into blocks
        block_size = max(2, len(rows_list) // 4)
        blocks = [rows_list[i:i+block_size] for i in range(0, len(rows_list), block_size)]
        
        # Shuffle blocks
        random.shuffle(blocks)
        
        # Flatten back to list
        reordered_rows = [idx for block in blocks for idx in block]
        
        # Get original values
        original_values = [df.at[idx, col] for idx in rows_list if idx in df.index]
        
        # Assign reordered values
        for i, idx in enumerate(reordered_rows):
            if idx in df.index and i < len(original_values):
                df.at[idx, col] = original_values[i]
        
        return df
    
    def _reverse_order(self, df, col, rows):
        """Reverse the temporal order of sampled rows."""
        rows_list = sorted(list(rows))
        
        # Get values in original order
        values = [df.at[idx, col] for idx in rows_list if idx in df.index]
        
        # Reverse values
        reversed_values = list(reversed(values))
        
        # Assign back
        valid_rows = [idx for idx in rows_list if idx in df.index]
        for idx, value in zip(valid_rows, reversed_values):
            df.at[idx, col] = value
        
        return df

