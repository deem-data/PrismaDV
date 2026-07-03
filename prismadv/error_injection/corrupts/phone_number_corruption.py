import random
import re

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class PhoneNumberCorruption(TabularCorruption):
    """
    Corrupt phone numbers with common issues.
    
    Strategies:
        - 'wrong_length': Make numbers too short or too long
        - 'invalid_country_code': Use invalid country codes
        - 'mixed_formats': Mix different formatting styles
        - 'letters': Include letters in the number
        - 'missing_digits': Remove some digits
        - 'mixed': Randomly apply various corruptions
    
    Examples:
        - +1-555-1234 -> +1-555-123 (too short)
        - +1-555-1234 -> +999-555-1234 (invalid country code)
        - +1-555-1234 -> +1 555 12-34 (mixed format)
        - +1-555-1234 -> +1-5SS-1234 (letters)
    """
    
    def __init__(self, columns=None, severity=None, sampling=None,
                 strategy="mixed", random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.strategy = strategy
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        valid_strategies = ["wrong_length", "invalid_country_code", "mixed_formats",
                          "letters", "missing_digits", "mixed"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
    
    def identify_columns(self, dataframe: pd.DataFrame):
        """Identify columns that likely contain phone numbers."""
        phone_cols = []
        
        # Pattern for phone numbers (flexible)
        phone_pattern = re.compile(r'[\d\-\+\(\)\s\.]{7,}')
        
        for col in dataframe.columns:
            if not (pd.api.types.is_string_dtype(dataframe[col]) or 
                   pd.api.types.is_object_dtype(dataframe[col])):
                continue
            
            # Sample non-null values
            sample = dataframe[col].dropna().head(50)
            
            if len(sample) == 0:
                continue
            
            # Check how many look like phone numbers
            phone_like_count = sum(
                1 for val in sample 
                if phone_pattern.match(str(val)) and 
                sum(c.isdigit() for c in str(val)) >= 7
            )
            
            # If >50% look like phone numbers, consider it a phone column
            if phone_like_count / len(sample) > 0.5:
                phone_cols.append(col)
        
        if not phone_cols:
            raise ValueError("No phone number columns found for PhoneNumberCorruption.")
        
        self.columns = phone_cols
        return phone_cols
    
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
            
            if self.strategy == "wrong_length":
                df.loc[rows, col] = df.loc[rows, col].apply(self._corrupt_wrong_length)
            elif self.strategy == "invalid_country_code":
                df.loc[rows, col] = df.loc[rows, col].apply(self._corrupt_invalid_country)
            elif self.strategy == "mixed_formats":
                df.loc[rows, col] = df.loc[rows, col].apply(self._corrupt_mixed_formats)
            elif self.strategy == "letters":
                df.loc[rows, col] = df.loc[rows, col].apply(self._corrupt_letters)
            elif self.strategy == "missing_digits":
                df.loc[rows, col] = df.loc[rows, col].apply(self._corrupt_missing_digits)
            elif self.strategy == "mixed":
                df.loc[rows, col] = df.loc[rows, col].apply(self._corrupt_mixed)
        
        return df
    
    def _extract_digits(self, phone):
        """Extract just the digits from a phone number."""
        return ''.join(c for c in str(phone) if c.isdigit())
    
    def _corrupt_wrong_length(self, phone):
        """Make number too short or too long."""
        if pd.isna(phone):
            return phone
        
        digits = self._extract_digits(phone)
        
        if len(digits) < 3:
            return phone
        
        # Randomly make it too short or too long
        if random.random() < 0.5:
            # Too short - remove 1-3 digits
            remove_count = random.randint(1, min(3, len(digits) - 1))
            digits = digits[:-remove_count]
        else:
            # Too long - add extra digits
            add_count = random.randint(1, 5)
            digits += ''.join(str(random.randint(0, 9)) for _ in range(add_count))
        
        return digits
    
    def _corrupt_invalid_country(self, phone):
        """Use invalid country codes."""
        if pd.isna(phone):
            return phone
        
        phone_str = str(phone)
        
        invalid_codes = ['+999', '+000', '+1111', '+00', '+9', '+123456']
        code = random.choice(invalid_codes)
        
        # If phone starts with +, replace it
        if phone_str.startswith('+'):
            # Find where digits start after +
            match = re.match(r'\+\d+', phone_str)
            if match:
                rest = phone_str[match.end():]
                return code + rest
        
        # Otherwise, prepend the invalid code
        return code + '-' + phone_str
    
    def _corrupt_mixed_formats(self, phone):
        """Mix different formatting styles."""
        if pd.isna(phone):
            return phone
        
        digits = self._extract_digits(phone)
        
        if len(digits) < 7:
            return phone
        
        # Apply random formatting
        formats = [
            lambda d: f"{d[:3]}-{d[3:6]}-{d[6:]}",
            lambda d: f"({d[:3]}) {d[3:6]}-{d[6:]}",
            lambda d: f"{d[:3]}.{d[3:6]}.{d[6:]}",
            lambda d: f"{d[:3]} {d[3:6]} {d[6:]}",
            lambda d: f"+1 {d[:3]}-{d[3:6]} {d[6:]}",
            lambda d: f"{d}",  # No formatting
            lambda d: f"{d[:2]}-{d[2:5]} {d[5:8]}.{d[8:]}",  # Mixed
        ]
        
        formatter = random.choice(formats)
        return formatter(digits)
    
    def _corrupt_letters(self, phone):
        """Include letters in the number."""
        if pd.isna(phone):
            return phone
        
        phone_str = str(phone)
        
        # Replace 1-3 digits with letters
        result = list(phone_str)
        digit_positions = [i for i, c in enumerate(result) if c.isdigit()]
        
        if digit_positions:
            num_replacements = min(random.randint(1, 3), len(digit_positions))
            positions_to_replace = random.sample(digit_positions, num_replacements)
            
            for pos in positions_to_replace:
                result[pos] = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        return ''.join(result)
    
    def _corrupt_missing_digits(self, phone):
        """Remove some digits."""
        if pd.isna(phone):
            return phone
        
        phone_str = str(phone)
        digits = self._extract_digits(phone)
        
        if len(digits) < 5:
            return phone
        
        # Remove 1-4 digits
        remove_count = random.randint(1, min(4, len(digits) - 3))
        
        # Remove from random positions
        digit_list = list(digits)
        for _ in range(remove_count):
            if digit_list:
                pos = random.randint(0, len(digit_list) - 1)
                digit_list.pop(pos)
        
        return ''.join(digit_list)
    
    def _corrupt_mixed(self, phone):
        """Apply random corruption strategy."""
        if pd.isna(phone):
            return phone
        
        strategies = [
            self._corrupt_wrong_length,
            self._corrupt_invalid_country,
            self._corrupt_mixed_formats,
            self._corrupt_letters,
            self._corrupt_missing_digits,
        ]
        
        strategy = random.choice(strategies)
        return strategy(phone)

