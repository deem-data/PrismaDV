import random

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class UniqueConstraintViolation(TabularCorruption):
    """
    Create duplicate values in columns that should be unique.
    
    Strategies:
        - 'exact_duplicates': Create exact copies of existing values
        - 'near_duplicates': Create values with small variations (e.g., ID_001 -> ID_001_copy)
        - 'high_frequency': Make certain values appear much more frequently
    
    Use cases:
        - Duplicate IDs in primary key columns
        - Duplicate usernames
        - Duplicate email addresses
        - Duplicate order numbers
    """
    
    def __init__(self, columns=None, severity=None, sampling=None,
                 strategy="exact_duplicates", random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.strategy = strategy
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        valid_strategies = ["exact_duplicates", "near_duplicates", "high_frequency"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
    
    def identify_columns(self, dataframe: pd.DataFrame):
        """
        Identify columns that appear to have unique constraints.
        Look for columns with high uniqueness ratio.
        """
        unique_cols = []
        
        for col in dataframe.columns:
            # Calculate uniqueness ratio
            total_non_null = dataframe[col].notna().sum()
            if total_non_null == 0:
                continue
            
            unique_count = dataframe[col].nunique()
            uniqueness_ratio = unique_count / total_non_null
            
            # Consider columns with >80% unique values as potentially unique
            if uniqueness_ratio > 0.8:
                unique_cols.append(col)
        
        if not unique_cols:
            # Fallback: use all columns
            unique_cols = list(dataframe.columns)
        
        self.columns = unique_cols
        return unique_cols
    
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
            
            if self.strategy == "exact_duplicates":
                df = self._create_exact_duplicates(df, col, rows)
            elif self.strategy == "near_duplicates":
                df = self._create_near_duplicates(df, col, rows)
            elif self.strategy == "high_frequency":
                df = self._create_high_frequency(df, col, rows)
        
        return df
    
    def _create_exact_duplicates(self, df, col, rows):
        """Create exact duplicate values."""
        # Select a subset of existing values to duplicate
        existing_values = df[col].dropna().unique()
        
        if len(existing_values) == 0:
            return df
        
        # Pick values to duplicate
        num_values_to_duplicate = max(1, int(len(existing_values) * 0.1))
        values_to_duplicate = np.random.choice(
            existing_values,
            size=min(num_values_to_duplicate, len(existing_values)),
            replace=False
        )
        
        # Assign these values to the sampled rows
        for idx in rows:
            if idx in df.index:
                df.at[idx, col] = random.choice(values_to_duplicate)
        
        return df
    
    def _create_near_duplicates(self, df, col, rows):
        """Create near-duplicate values with small variations."""
        for idx in rows:
            if idx not in df.index:
                continue
            
            original_value = df.at[idx, col]
            
            if pd.isna(original_value):
                continue
            
            # Create a near-duplicate
            near_dup = self._create_variation(original_value)
            df.at[idx, col] = near_dup
        
        return df
    
    def _create_variation(self, value):
        """Create a small variation of the value."""
        if pd.api.types.is_numeric_dtype(type(value)):
            # For numbers, add/subtract 1
            return value + random.choice([-1, 1])
        else:
            # For strings, add suffix or modify slightly
            str_value = str(value)
            variations = [
                str_value + "_copy",
                str_value + "_dup",
                str_value + "1",
                str_value + "_2",
                str_value.upper() if str_value.islower() else str_value.lower(),
                str_value + " ",  # Add trailing space
                " " + str_value,  # Add leading space
            ]
            return random.choice(variations)
    
    def _create_high_frequency(self, df, col, rows):
        """Make certain values appear much more frequently."""
        # Pick a few values to make highly frequent
        existing_values = df[col].dropna().unique()
        
        if len(existing_values) == 0:
            return df
        
        # Pick 1-3 values to make frequent
        num_frequent = min(3, len(existing_values))
        frequent_values = np.random.choice(
            existing_values,
            size=num_frequent,
            replace=False
        )
        
        # Assign these values to the sampled rows
        for idx in rows:
            if idx in df.index:
                df.at[idx, col] = random.choice(frequent_values)
        
        return df

