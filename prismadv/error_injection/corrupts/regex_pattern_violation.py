import random
import re

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class RegexPatternViolation(TabularCorruption):
    """
    Generic pattern violation for any regex-constrained field.
    
    Parameters:
        pattern: Regex pattern that values should match (dict: {col_name: pattern})
        strategy: How to violate the pattern
            - 'character_insertion': Insert invalid characters
            - 'character_deletion': Delete required characters
            - 'format_breaking': Break the format structure
            - 'case_violation': Change case when pattern is case-sensitive
            - 'mixed': Random violations
    
    Examples:
        - ZIP codes: 12345 -> 1234A
        - SSN: 123-45-6789 -> 12345-6789
        - Product codes: ABC-123 -> ABC123
        - License plates: ABC1234 -> ABC-12-34
    """
    
    def __init__(self, columns=None, severity=None, sampling=None,
                 pattern=None, strategy="mixed", random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.pattern = pattern or {}  # Dict mapping column name to regex pattern
        self.strategy = strategy
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        valid_strategies = ["character_insertion", "character_deletion", 
                          "format_breaking", "case_violation", "mixed"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
    
    def identify_columns(self, dataframe: pd.DataFrame):
        """
        Cannot auto-identify regex patterns.
        User must specify pattern dict.
        """
        raise NotImplementedError(
            "RegexPatternViolation requires explicit pattern specification. "
            "Provide a dict mapping column names to regex patterns."
        )
    
    def transform(self, dataframe: pd.DataFrame):
        self.validate_data(dataframe)
        
        if not self.pattern:
            raise ValueError("pattern dict must be specified for RegexPatternViolation")
        
        if self.severity is None or self.severity <= 0:
            return dataframe
        
        df = dataframe.copy(deep=True)
        rows = self.sample_rows(df)
        
        # Use columns from pattern dict if columns not specified
        if not self.columns:
            self.columns = list(self.pattern.keys())
        
        for col in self.columns:
            if col not in df.columns:
                continue
            
            if col not in self.pattern:
                continue
            
            # Ensure column is string type
            df[col] = df[col].astype('object')
            
            pattern = self.pattern[col]
            
            if self.strategy == "character_insertion":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._violate_character_insertion(x, pattern)
                )
            elif self.strategy == "character_deletion":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._violate_character_deletion(x, pattern)
                )
            elif self.strategy == "format_breaking":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._violate_format_breaking(x, pattern)
                )
            elif self.strategy == "case_violation":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._violate_case(x, pattern)
                )
            elif self.strategy == "mixed":
                df.loc[rows, col] = df.loc[rows, col].apply(
                    lambda x: self._violate_mixed(x, pattern)
                )
        
        return df
    
    def _violate_character_insertion(self, value, pattern):
        """Insert invalid characters."""
        if pd.isna(value):
            return value
        
        value_str = str(value)
        
        if len(value_str) < 2:
            return value
        
        # Insert random characters
        invalid_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', 
                        '[', ']', '{', '}', '|', '\\', '?', '~']
        
        pos = random.randint(0, len(value_str))
        char = random.choice(invalid_chars)
        
        return value_str[:pos] + char + value_str[pos:]
    
    def _violate_character_deletion(self, value, pattern):
        """Delete required characters."""
        if pd.isna(value):
            return value
        
        value_str = str(value)
        
        if len(value_str) < 3:
            return value
        
        # Delete 1-2 characters
        num_to_delete = random.randint(1, min(2, len(value_str) - 1))
        
        result = list(value_str)
        for _ in range(num_to_delete):
            if result:
                pos = random.randint(0, len(result) - 1)
                result.pop(pos)
        
        return ''.join(result)
    
    def _violate_format_breaking(self, value, pattern):
        """Break the format structure (remove separators, etc.)."""
        if pd.isna(value):
            return value
        
        value_str = str(value)
        
        # Remove common separators
        separators = ['-', '_', '.', ' ', '/', '\\', ':', ',']
        
        for sep in separators:
            if sep in value_str:
                # Either remove all separators or move them
                if random.random() < 0.5:
                    value_str = value_str.replace(sep, '')
                else:
                    # Move separator to wrong position
                    value_str = value_str.replace(sep, '')
                    if len(value_str) > 2:
                        pos = random.randint(1, len(value_str) - 1)
                        value_str = value_str[:pos] + sep + value_str[pos:]
                break
        
        return value_str
    
    def _violate_case(self, value, pattern):
        """Change case when pattern is case-sensitive."""
        if pd.isna(value):
            return value
        
        value_str = str(value)
        
        # Randomly change case of alphabetic characters
        result = []
        for char in value_str:
            if char.isalpha():
                if random.random() < 0.5:
                    result.append(char.swapcase())
                else:
                    result.append(char)
            else:
                result.append(char)
        
        return ''.join(result)
    
    def _violate_mixed(self, value, pattern):
        """Apply random violation strategy."""
        if pd.isna(value):
            return value
        
        strategies = [
            self._violate_character_insertion,
            self._violate_character_deletion,
            self._violate_format_breaking,
            self._violate_case,
        ]
        
        strategy = random.choice(strategies)
        return strategy(value, pattern)

