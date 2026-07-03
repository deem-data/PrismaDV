import random

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class DataTypeViolation(TabularCorruption):
    """
    Inject values that violate expected data types.
    
    Strategies:
        - 'strings_in_numeric': Inject string values into numeric columns
        - 'numbers_in_string': Inject numeric values into string columns
        - 'mixed_types': Mix different data types in the same column
        - 'invalid_conversions': Inject values that can't be converted (e.g., '1.2.3', 'N/A')
    """
    
    def __init__(self, columns=None, severity=None, sampling=None,
                 strategy="strings_in_numeric", random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.strategy = strategy
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        valid_strategies = ["strings_in_numeric", "numbers_in_string", "mixed_types", "invalid_conversions"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
    
    def identify_columns(self, dataframe: pd.DataFrame):
        """Identify columns based on strategy."""
        if self.strategy == "strings_in_numeric":
            # Find numeric columns
            cols = [col for col in dataframe.columns 
                   if pd.api.types.is_numeric_dtype(dataframe[col])]
        elif self.strategy == "numbers_in_string":
            # Find string/object columns
            cols = [col for col in dataframe.columns
                   if pd.api.types.is_string_dtype(dataframe[col]) or 
                   pd.api.types.is_object_dtype(dataframe[col])]
        else:
            # For mixed_types and invalid_conversions, any column works
            cols = list(dataframe.columns)
        
        if not cols:
            raise ValueError(f"No suitable columns found for DataTypeViolation with strategy '{self.strategy}'.")
        
        self.columns = cols
        return cols
    
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
            
            # Convert column to object type to allow mixed types
            df[col] = df[col].astype('object')
            
            if self.strategy == "strings_in_numeric":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._inject_string_in_numeric(x)
                )
            elif self.strategy == "numbers_in_string":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._inject_number_in_string(x)
                )
            elif self.strategy == "mixed_types":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._inject_mixed_type(x)
                )
            elif self.strategy == "invalid_conversions":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._inject_invalid_conversion(x)
                )
        
        return df
    
    def _inject_string_in_numeric(self, value):
        """Inject string values into numeric columns."""
        if pd.isna(value):
            return value
        
        string_values = ["N/A", "null", "None", "nan", "error", "###", "—", 
                        "invalid", "text", "abc", "xyz", "unknown"]
        return random.choice(string_values)
    
    def _inject_number_in_string(self, value):
        """Inject numeric values into string columns."""
        if pd.isna(value):
            return value
        
        # Return a numeric value
        return random.choice([42, 0, -1, 3.14, 999, 100, 0.0, -99.9])
    
    def _inject_mixed_type(self, value):
        """Mix different data types."""
        if pd.isna(value):
            return value
        
        type_options = [
            lambda: random.randint(-100, 100),  # int
            lambda: random.uniform(-100, 100),   # float
            lambda: random.choice(["text", "string", "value", "data"]),  # string
            lambda: True if random.random() < 0.5 else False,  # boolean
            lambda: None,  # None
            lambda: ["list", "item"],  # list
            lambda: {"dict": "value"},  # dict
        ]
        
        return random.choice(type_options)()
    
    def _inject_invalid_conversion(self, value):
        """Inject values that can't be easily converted."""
        if pd.isna(value):
            return value
        
        invalid_values = [
            "1.2.3.4",      # Multiple decimals
            "12-34-56",     # Ambiguous format
            "1e2e3",        # Double exponent
            "0x1G",         # Invalid hex
            "++123",        # Double sign
            "--456",        # Double sign
            "12..34",       # Double decimal
            "NaN.",         # Invalid NaN format
            "inf/",         # Invalid inf format
            "1,2,3",        # Comma-separated (ambiguous)
            "1 234",        # Space separator
            "(123)",        # Parentheses
            "$123",         # Currency symbol
            "123%",         # Percentage symbol
            "N/A",          # Not available
            "#VALUE!",      # Excel error
            "#DIV/0!",      # Excel error
        ]
        
        return random.choice(invalid_values)

