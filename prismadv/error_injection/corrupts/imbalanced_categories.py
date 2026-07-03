import random

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class ImbalancedCategories(TabularCorruption):
    """
    Create severe class imbalance in categorical columns.
    
    Strategies:
        - 'over_represent': Make certain categories much more frequent
        - 'under_represent': Make certain categories very rare
        - 'eliminate': Completely remove some categories
        - 'majority_only': Replace most values with majority class
    
    Use cases:
        - Testing imbalanced classification handling
        - Rare event simulation
        - Dominant category scenarios
        - Class imbalance in training data
    """
    
    def __init__(self, columns=None, severity=None, sampling=None,
                 strategy='over_represent', target_categories=None,
                 imbalance_ratio=10.0, random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.strategy = strategy
        self.target_categories = target_categories  # Specific categories to manipulate
        self.imbalance_ratio = imbalance_ratio  # How severe the imbalance
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        valid_strategies = ['over_represent', 'under_represent', 'eliminate', 'majority_only']
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
    
    def identify_columns(self, dataframe: pd.DataFrame):
        """Identify categorical columns."""
        categorical_cols = []
        
        for col in dataframe.columns:
            # Check if column has limited unique values (likely categorical)
            num_unique = dataframe[col].nunique()
            total_rows = len(dataframe)
            
            # Consider categorical if < 30 unique values or < 5% unique
            if num_unique < 30 or (num_unique / total_rows) < 0.05:
                categorical_cols.append(col)
        
        if not categorical_cols:
            raise ValueError("No categorical columns found for ImbalancedCategories.")
        
        self.columns = categorical_cols
        return categorical_cols
    
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
            
            if self.strategy == 'over_represent':
                df = self._over_represent(df, col, rows)
            elif self.strategy == 'under_represent':
                df = self._under_represent(df, col, rows)
            elif self.strategy == 'eliminate':
                df = self._eliminate(df, col, rows)
            elif self.strategy == 'majority_only':
                df = self._majority_only(df, col, rows)
        
        return df
    
    def _over_represent(self, df, col, rows):
        """Make certain categories much more frequent."""
        # Get value counts
        value_counts = df[col].value_counts()
        
        if len(value_counts) == 0:
            return df
        
        # Pick categories to over-represent
        if self.target_categories:
            categories = [c for c in self.target_categories if c in value_counts.index]
        else:
            # Pick 1-2 random categories
            num_to_pick = min(2, len(value_counts))
            categories = np.random.choice(value_counts.index, size=num_to_pick, replace=False)
        
        if len(categories) == 0:
            return df
        
        # Replace sampled rows with these categories
        for idx in rows:
            if idx in df.index:
                df.at[idx, col] = random.choice(categories)
        
        return df
    
    def _under_represent(self, df, col, rows):
        """Make certain categories very rare."""
        value_counts = df[col].value_counts()
        
        if len(value_counts) == 0:
            return df
        
        # Pick categories to under-represent
        if self.target_categories:
            rare_categories = [c for c in self.target_categories if c in value_counts.index]
        else:
            # Pick least frequent categories
            num_to_pick = min(2, len(value_counts))
            rare_categories = value_counts.nsmallest(num_to_pick).index.tolist()
        
        if len(rare_categories) == 0:
            return df
        
        # Replace instances of these categories with other categories
        other_categories = [c for c in value_counts.index if c not in rare_categories]
        
        if len(other_categories) == 0:
            return df
        
        for idx in rows:
            if idx in df.index and df.at[idx, col] in rare_categories:
                df.at[idx, col] = random.choice(other_categories)
        
        return df
    
    def _eliminate(self, df, col, rows):
        """Completely remove some categories."""
        value_counts = df[col].value_counts()
        
        if len(value_counts) <= 1:
            return df
        
        # Pick categories to eliminate
        if self.target_categories:
            eliminate_categories = [c for c in self.target_categories if c in value_counts.index]
        else:
            # Pick 1-2 least frequent categories
            num_to_pick = min(2, len(value_counts) - 1)
            eliminate_categories = value_counts.nsmallest(num_to_pick).index.tolist()
        
        if len(eliminate_categories) == 0:
            return df
        
        # Get remaining categories
        remaining_categories = [c for c in value_counts.index if c not in eliminate_categories]
        
        if len(remaining_categories) == 0:
            return df
        
        # Replace eliminated categories
        for idx in df.index:
            if df.at[idx, col] in eliminate_categories:
                df.at[idx, col] = random.choice(remaining_categories)
        
        return df
    
    def _majority_only(self, df, col, rows):
        """Replace most values with majority class."""
        value_counts = df[col].value_counts()
        
        if len(value_counts) == 0:
            return df
        
        # Get majority class
        majority_class = value_counts.index[0]
        
        # Replace sampled rows with majority class
        for idx in rows:
            if idx in df.index:
                df.at[idx, col] = majority_class
        
        return df

