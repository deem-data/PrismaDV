import random

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class ForeignKeyViolation(TabularCorruption):
    """
    Inject values that violate foreign key relationships.
    
    Parameters:
        reference_values: Set/list of valid reference values (e.g., valid IDs from parent table)
                         Can be a dict: {col_name: reference_values}
        strategy: How to violate the foreign key
            - 'invalid_values': Use values not in reference set
            - 'null_values': Replace with NULL/NaN
            - 'out_of_range': Use values outside expected range
    
    Use cases:
        - Orphaned records in child tables
        - Invalid reference IDs
        - Broken relationships in relational data
        - Missing parent records
    """
    
    def __init__(self, columns=None, severity=None, sampling=None,
                 reference_values=None, strategy='invalid_values',
                 random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.reference_values = reference_values or {}
        self.strategy = strategy
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        valid_strategies = ['invalid_values', 'null_values', 'out_of_range']
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
    
    def identify_columns(self, dataframe: pd.DataFrame):
        """
        Cannot auto-identify foreign key relationships.
        User must specify columns and reference_values.
        """
        raise NotImplementedError(
            "ForeignKeyViolation requires explicit reference_values specification. "
            "Provide a dict mapping column names to valid reference values."
        )
    
    def transform(self, dataframe: pd.DataFrame):
        self.validate_data(dataframe)
        
        if not self.reference_values:
            raise ValueError("reference_values must be specified for ForeignKeyViolation")
        
        if self.severity is None or self.severity <= 0:
            return dataframe
        
        df = dataframe.copy(deep=True)
        rows = self.sample_rows(df)
        
        # Use columns from reference_values if columns not specified
        if not self.columns:
            self.columns = list(self.reference_values.keys())
        
        for col in self.columns:
            if col not in df.columns:
                continue
            
            if col not in self.reference_values:
                continue
            
            ref_values = self.reference_values[col]
            
            if self.strategy == 'invalid_values':
                df = self._inject_invalid_values(df, col, rows, ref_values)
            elif self.strategy == 'null_values':
                df = self._inject_null_values(df, col, rows)
            elif self.strategy == 'out_of_range':
                df = self._inject_out_of_range(df, col, rows, ref_values)
        
        return df
    
    def _inject_invalid_values(self, df, col, rows, ref_values):
        """Inject values not in the reference set."""
        ref_set = set(ref_values) if not isinstance(ref_values, set) else ref_values
        
        # Get existing values in the column
        existing_values = set(df[col].dropna().unique())
        
        # Find values that are not in reference set
        invalid_values = existing_values - ref_set
        
        # If no existing invalid values, generate some
        if len(invalid_values) == 0:
            # Generate invalid values based on type
            sample_val = next(iter(ref_set)) if ref_set else None
            
            if sample_val is not None:
                if pd.api.types.is_numeric_dtype(type(sample_val)):
                    # Generate numbers outside the reference range
                    max_ref = max(ref_set)
                    invalid_values = set([max_ref + i for i in range(1, 11)])
                else:
                    # Generate invalid string IDs
                    invalid_values = set([
                        f"INVALID_{i}" for i in range(10)
                    ])
        
        if len(invalid_values) == 0:
            return df
        
        # Replace sampled rows with invalid values
        invalid_list = list(invalid_values)
        for idx in rows:
            if idx in df.index:
                df.at[idx, col] = random.choice(invalid_list)
        
        return df
    
    def _inject_null_values(self, df, col, rows):
        """Replace with NULL/NaN values."""
        for idx in rows:
            if idx in df.index:
                df.at[idx, col] = np.nan
        
        return df
    
    def _inject_out_of_range(self, df, col, rows, ref_values):
        """Use values outside expected range."""
        if not ref_values:
            return df
        
        # Try to determine the range
        try:
            min_ref = min(ref_values)
            max_ref = max(ref_values)
            
            # Generate out-of-range values
            for idx in rows:
                if idx in df.index:
                    # Randomly choose below min or above max
                    if random.random() < 0.5:
                        # Below min
                        if pd.api.types.is_numeric_dtype(type(min_ref)):
                            df.at[idx, col] = min_ref - random.randint(1, 100)
                        else:
                            df.at[idx, col] = f"BEFORE_{min_ref}"
                    else:
                        # Above max
                        if pd.api.types.is_numeric_dtype(type(max_ref)):
                            df.at[idx, col] = max_ref + random.randint(1, 100)
                        else:
                            df.at[idx, col] = f"AFTER_{max_ref}"
        except (TypeError, ValueError):
            # Can't determine range, use invalid_values approach
            return self._inject_invalid_values(df, col, rows, ref_values)
        
        return df

