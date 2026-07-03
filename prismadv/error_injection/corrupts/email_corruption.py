import random
import re

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class EmailCorruption(TabularCorruption):
    """
    Corrupt email addresses with realistic errors.
    
    Strategies:
        - 'missing_at': Remove or corrupt the @ symbol
        - 'invalid_domain': Use invalid or non-existent domains
        - 'double_dots': Add consecutive dots in local or domain part
        - 'special_chars': Add invalid special characters
        - 'spaces': Add spaces in the email
        - 'mixed': Randomly apply various corruptions
    
    Examples:
        - user@example.com -> userexample.com (missing @)
        - user@example.com -> user@invalid (invalid domain)
        - user@example.com -> us..er@example.com (double dots)
        - user@example.com -> user name@example.com (spaces)
    """
    
    def __init__(self, columns=None, severity=None, sampling=None,
                 strategy="mixed", random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.strategy = strategy
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        valid_strategies = ["missing_at", "invalid_domain", "double_dots", 
                          "special_chars", "spaces", "mixed"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
    
    def identify_columns(self, dataframe: pd.DataFrame):
        """Identify columns that likely contain email addresses."""
        email_cols = []
        
        # Simple email regex pattern
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        for col in dataframe.columns:
            if not (pd.api.types.is_string_dtype(dataframe[col]) or 
                   pd.api.types.is_object_dtype(dataframe[col])):
                continue
            
            # Sample non-null values
            sample = dataframe[col].dropna().head(50)
            
            if len(sample) == 0:
                continue
            
            # Check how many look like emails
            email_like_count = sum(1 for val in sample if email_pattern.match(str(val)))
            
            # If >50% look like emails, consider it an email column
            if email_like_count / len(sample) > 0.5:
                email_cols.append(col)
        
        if not email_cols:
            raise ValueError("No email columns found for EmailCorruption.")
        
        self.columns = email_cols
        return email_cols
    
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
            
            # Ensure column is string type
            df[col] = df[col].astype('object')
            
            if self.strategy == "missing_at":
                df.loc[rows, col] = df.loc[rows, col].apply(self._corrupt_missing_at)
            elif self.strategy == "invalid_domain":
                df.loc[rows, col] = df.loc[rows, col].apply(self._corrupt_invalid_domain)
            elif self.strategy == "double_dots":
                df.loc[rows, col] = df.loc[rows, col].apply(self._corrupt_double_dots)
            elif self.strategy == "special_chars":
                df.loc[rows, col] = df.loc[rows, col].apply(self._corrupt_special_chars)
            elif self.strategy == "spaces":
                df.loc[rows, col] = df.loc[rows, col].apply(self._corrupt_spaces)
            elif self.strategy == "mixed":
                df.loc[rows, col] = df.loc[rows, col].apply(self._corrupt_mixed)
        
        return df
    
    def _corrupt_missing_at(self, email):
        """Remove or corrupt the @ symbol."""
        if pd.isna(email):
            return email
        
        email_str = str(email)
        
        if '@' not in email_str:
            return email
        
        corruption_options = [
            email_str.replace('@', ''),        # Remove @
            email_str.replace('@', ' '),       # Replace with space
            email_str.replace('@', '.'),       # Replace with dot
            email_str.replace('@', '@@'),      # Double @
            email_str.replace('@', '#'),       # Replace with #
        ]
        
        return random.choice(corruption_options)
    
    def _corrupt_invalid_domain(self, email):
        """Use invalid or non-existent domains."""
        if pd.isna(email):
            return email
        
        email_str = str(email)
        
        if '@' not in email_str:
            return email
        
        local, domain = email_str.rsplit('@', 1)
        
        invalid_domains = [
            "invalid",
            "test",
            "localhost",
            "example",
            "domain.invalid",
            "no-domain",
            "xxx",
            ".com",
            "com",
            "",
        ]
        
        new_domain = random.choice(invalid_domains)
        
        if new_domain:
            return f"{local}@{new_domain}"
        else:
            return f"{local}@"
    
    def _corrupt_double_dots(self, email):
        """Add consecutive dots."""
        if pd.isna(email):
            return email
        
        email_str = str(email)
        
        if '@' not in email_str:
            return email
        
        # Insert double dot at random position
        if len(email_str) > 2:
            pos = random.randint(1, len(email_str) - 1)
            return email_str[:pos] + '.' + email_str[pos:]
        
        return email_str
    
    def _corrupt_special_chars(self, email):
        """Add invalid special characters."""
        if pd.isna(email):
            return email
        
        email_str = str(email)
        
        if '@' not in email_str:
            return email
        
        invalid_chars = ['!', '#', '$', '%', '^', '&', '*', '(', ')', 
                        '[', ']', '{', '}', '|', '\\', '/', '?']
        
        # Insert invalid char at random position
        if len(email_str) > 2:
            pos = random.randint(1, len(email_str) - 1)
            char = random.choice(invalid_chars)
            return email_str[:pos] + char + email_str[pos:]
        
        return email_str
    
    def _corrupt_spaces(self, email):
        """Add spaces in the email."""
        if pd.isna(email):
            return email
        
        email_str = str(email)
        
        if '@' not in email_str:
            return email
        
        # Insert space at random position
        if len(email_str) > 2:
            pos = random.randint(1, len(email_str) - 1)
            return email_str[:pos] + ' ' + email_str[pos:]
        
        return email_str
    
    def _corrupt_mixed(self, email):
        """Apply random corruption strategy."""
        if pd.isna(email):
            return email
        
        strategies = [
            self._corrupt_missing_at,
            self._corrupt_invalid_domain,
            self._corrupt_double_dots,
            self._corrupt_special_chars,
            self._corrupt_spaces,
        ]
        
        strategy = random.choice(strategies)
        return strategy(email)

