import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class DateFormatCorruption(TabularCorruption):
    """
    Corrupt date/datetime formats and values.
    
    Strategies:
        - 'invalid_dates': Create impossible dates (Feb 30, month 13, etc.)
        - 'format_mixing': Mix different date formats (MM/DD/YYYY vs DD/MM/YYYY)
        - 'invalid_strings': Replace with invalid date strings
        - 'future_dates': For contexts where future dates shouldn't exist
        - 'past_dates': For contexts where old dates shouldn't exist
    """
    
    def __init__(self, columns=None, severity=None, sampling=None,
                 strategy="invalid_dates", random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.strategy = strategy
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        valid_strategies = ["invalid_dates", "format_mixing", "invalid_strings", 
                          "future_dates", "past_dates"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
    
    def identify_columns(self, dataframe: pd.DataFrame):
        """Identify date/datetime columns."""
        date_cols = []
        for col in dataframe.columns:
            # Check if column is datetime type
            if pd.api.types.is_datetime64_any_dtype(dataframe[col]):
                date_cols.append(col)
            # Also check if string column contains date-like values
            elif pd.api.types.is_string_dtype(dataframe[col]) or pd.api.types.is_object_dtype(dataframe[col]):
                # Sample a few non-null values
                sample = dataframe[col].dropna().head(10)
                if len(sample) > 0:
                    # Try to parse as dates
                    try:
                        pd.to_datetime(sample, errors='coerce')
                        if sample.apply(lambda x: self._looks_like_date(str(x))).mean() > 0.5:
                            date_cols.append(col)
                    except:
                        pass
        
        if not date_cols:
            raise ValueError("No date/datetime columns found for DateFormatCorruption.")
        
        self.columns = date_cols
        return date_cols
    
    def _looks_like_date(self, s):
        """Quick heuristic to check if string looks like a date."""
        date_indicators = ['/', '-', ':', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Monday', 'Tuesday']
        return any(indicator in s for indicator in date_indicators)
    
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
            
            # Convert to object type to allow string corruption
            df[col] = df[col].astype('object')
            
            if self.strategy == "invalid_dates":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._create_invalid_date(x)
                )
            elif self.strategy == "format_mixing":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._mix_format(x)
                )
            elif self.strategy == "invalid_strings":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._create_invalid_string(x)
                )
            elif self.strategy == "future_dates":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._create_future_date(x)
                )
            elif self.strategy == "past_dates":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._create_past_date(x)
                )
        
        return df
    
    def _create_invalid_date(self, value):
        """Create impossible dates."""
        if pd.isna(value):
            return value
        
        invalid_dates = [
            "2023-02-30",    # Feb 30 doesn't exist
            "2023-13-01",    # Month 13 doesn't exist
            "2023-04-31",    # April 31 doesn't exist
            "2023-06-31",    # June 31 doesn't exist
            "2023-09-31",    # September 31 doesn't exist
            "2023-11-31",    # November 31 doesn't exist
            "2023-00-15",    # Month 0 doesn't exist
            "2023-01-32",    # Day 32 doesn't exist
            "2023-01-00",    # Day 0 doesn't exist
            "2021-02-29",    # 2021 is not a leap year
            "9999-99-99",    # Completely invalid
        ]
        
        return random.choice(invalid_dates)
    
    def _mix_format(self, value):
        """Mix different date formats."""
        if pd.isna(value):
            return value
        
        # Try to parse the value
        try:
            if isinstance(value, (datetime, pd.Timestamp)):
                dt = pd.to_datetime(value)
            else:
                dt = pd.to_datetime(str(value), errors='coerce')
            
            if pd.isna(dt):
                return value
            
            # Generate in different formats
            formats = [
                dt.strftime("%m/%d/%Y"),     # MM/DD/YYYY
                dt.strftime("%d/%m/%Y"),     # DD/MM/YYYY
                dt.strftime("%Y-%m-%d"),     # ISO format
                dt.strftime("%d-%m-%Y"),     # European
                dt.strftime("%m-%d-%Y"),     # US with dashes
                dt.strftime("%Y/%m/%d"),     # ISO with slashes
                dt.strftime("%d.%m.%Y"),     # German style
                dt.strftime("%Y%m%d"),       # Compact
            ]
            
            return random.choice(formats)
        except:
            return value
    
    def _create_invalid_string(self, value):
        """Replace with invalid date strings."""
        if pd.isna(value):
            return value
        
        invalid_strings = [
            "not-a-date",
            "INVALID",
            "N/A",
            "null",
            "0000-00-00",
            "0000-00-00 00:00:00",
            "1900-01-01",     # Common placeholder
            "9999-12-31",     # Max date placeholder
            "01/01/0001",
            "TBD",
            "Unknown",
            "pending",
            "##########",
        ]
        
        return random.choice(invalid_strings)
    
    def _create_future_date(self, value):
        """Create dates in the far future."""
        if pd.isna(value):
            return value
        
        # Create dates 10-50 years in the future
        years_ahead = random.randint(10, 50)
        future_date = datetime.now() + timedelta(days=365 * years_ahead)
        
        return future_date.strftime("%Y-%m-%d")
    
    def _create_past_date(self, value):
        """Create dates in the far past."""
        if pd.isna(value):
            return value
        
        # Create dates 50-200 years in the past
        years_back = random.randint(50, 200)
        past_date = datetime.now() - timedelta(days=365 * years_back)
        
        return past_date.strftime("%Y-%m-%d")

