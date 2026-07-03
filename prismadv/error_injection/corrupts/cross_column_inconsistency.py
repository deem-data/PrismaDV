import random
from typing import List, Tuple, Callable

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class CrossColumnInconsistency(TabularCorruption):
    """
    Violate logical relationships between columns.
    
    This corruption requires specifying column pairs and their expected relationships.
    
    Parameters:
        column_pairs: List of tuples (col1, col2, relationship_type)
            Relationship types:
                - 'less_than': col1 < col2 (e.g., start_date < end_date)
                - 'greater_than': col1 > col2
                - 'less_equal': col1 <= col2
                - 'greater_equal': col1 >= col2
                - 'equal': col1 == col2
                - 'not_equal': col1 != col2
    
    Examples:
        - end_date < start_date
        - child_age > parent_age
        - withdrawal > balance
        - shipping_cost > total_cost
    """
    
    def __init__(self, columns=None, severity=None, sampling=None,
                 column_pairs=None, random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.column_pairs = column_pairs or []
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        # Validate column_pairs format
        if not isinstance(self.column_pairs, list):
            raise ValueError("column_pairs must be a list of tuples")
        
        for pair in self.column_pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 3:
                raise ValueError(
                    "Each column pair must be a tuple/list of (col1, col2, relationship_type)"
                )
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
    
    def identify_columns(self, dataframe: pd.DataFrame):
        """
        Cannot auto-identify columns for cross-column inconsistencies.
        User must specify column_pairs.
        """
        raise NotImplementedError(
            "CrossColumnInconsistency requires explicit column_pairs specification. "
            "Cannot auto-identify column relationships."
        )
    
    def transform(self, dataframe: pd.DataFrame):
        self.validate_data(dataframe)
        
        if not self.column_pairs:
            raise ValueError("column_pairs must be specified for CrossColumnInconsistency")
        
        if self.severity is None or self.severity <= 0:
            return dataframe
        
        df = dataframe.copy(deep=True)
        rows = self.sample_rows(df)
        
        for col1, col2, relationship in self.column_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            
            # Violate the relationship for sampled rows
            for idx in rows:
                if idx not in df.index:
                    continue
                
                val1 = df.at[idx, col1]
                val2 = df.at[idx, col2]
                
                # Skip if either value is NaN
                if pd.isna(val1) or pd.isna(val2):
                    continue
                
                # Violate the specified relationship
                df.at[idx, col2] = self._violate_relationship(val1, val2, relationship)
        
        return df
    
    def _violate_relationship(self, val1, val2, relationship):
        """Violate the specified relationship by modifying val2."""
        try:
            if relationship == 'less_than':
                # Ensure val1 >= val2 (violates val1 < val2)
                if pd.api.types.is_numeric_dtype(type(val1)):
                    return val1 - abs(val1) * random.uniform(0.1, 0.5)
                else:
                    # For dates or other comparable types
                    return val1
            
            elif relationship == 'greater_than':
                # Ensure val1 <= val2 (violates val1 > val2)
                if pd.api.types.is_numeric_dtype(type(val1)):
                    return val1 + abs(val1) * random.uniform(0.1, 0.5)
                else:
                    return val1
            
            elif relationship == 'less_equal':
                # Ensure val1 > val2 (violates val1 <= val2)
                if pd.api.types.is_numeric_dtype(type(val1)):
                    return val1 - abs(val1) * random.uniform(0.1, 0.5) - 1
                else:
                    return val1
            
            elif relationship == 'greater_equal':
                # Ensure val1 < val2 (violates val1 >= val2)
                if pd.api.types.is_numeric_dtype(type(val1)):
                    return val1 + abs(val1) * random.uniform(0.1, 0.5) + 1
                else:
                    return val1
            
            elif relationship == 'equal':
                # Ensure val1 != val2 (violates val1 == val2)
                if pd.api.types.is_numeric_dtype(type(val1)):
                    return val1 + random.choice([-1, 1]) * abs(val1) * random.uniform(0.1, 0.5)
                else:
                    return str(val2) + "_modified"
            
            elif relationship == 'not_equal':
                # Ensure val1 == val2 (violates val1 != val2)
                return val1
            
            else:
                return val2
        
        except Exception:
            # If any error occurs in transformation, return original value
            return val2

