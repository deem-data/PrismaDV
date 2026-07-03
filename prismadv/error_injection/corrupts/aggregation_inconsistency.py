import random

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class AggregationInconsistency(TabularCorruption):
    """
    Violate aggregate constraints between columns.
    
    Parameters:
        aggregation_rules: List of tuples defining aggregation relationships
            Format: (agg_column, component_columns, agg_type, tolerance)
            agg_type can be: 'sum', 'average', 'count', 'max', 'min'
            
    Examples:
        - Line items don't sum to total
        - Average doesn't match individual values
        - Count doesn't match number of items
        - Subtotals don't add up to grand total
    
    Use cases:
        - Invoice validation
        - Financial reconciliation
        - Inventory consistency
        - Statistical report validation
    """
    
    def __init__(self, columns=None, severity=None, sampling=None,
                 aggregation_rules=None, violation_factor=0.2,
                 random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.aggregation_rules = aggregation_rules or []
        self.violation_factor = violation_factor
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        # Validate aggregation_rules format
        if not isinstance(self.aggregation_rules, list):
            raise ValueError("aggregation_rules must be a list of tuples")
        
        for rule in self.aggregation_rules:
            if not isinstance(rule, (tuple, list)) or len(rule) < 3:
                raise ValueError(
                    "Each aggregation rule must be a tuple/list of "
                    "(agg_column, component_columns, agg_type [, tolerance])"
                )
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
    
    def identify_columns(self, dataframe: pd.DataFrame):
        """
        Cannot auto-identify aggregation relationships.
        User must specify aggregation_rules.
        """
        raise NotImplementedError(
            "AggregationInconsistency requires explicit aggregation_rules specification. "
            "Provide rules as (agg_column, component_columns, agg_type)."
        )
    
    def transform(self, dataframe: pd.DataFrame):
        self.validate_data(dataframe)
        
        if not self.aggregation_rules:
            raise ValueError("aggregation_rules must be specified for AggregationInconsistency")
        
        if self.severity is None or self.severity <= 0:
            return dataframe
        
        df = dataframe.copy(deep=True)
        rows = self.sample_rows(df)
        
        for rule in self.aggregation_rules:
            if len(rule) == 3:
                agg_col, component_cols, agg_type = rule
                tolerance = 0
            else:
                agg_col, component_cols, agg_type, tolerance = rule
            
            if agg_col not in df.columns:
                continue
            
            # Check if all component columns exist
            if not all(col in df.columns for col in component_cols):
                continue
            
            # Violate the aggregation for sampled rows
            for idx in rows:
                if idx not in df.index:
                    continue
                
                if agg_type == 'sum':
                    df = self._violate_sum(df, idx, agg_col, component_cols)
                elif agg_type == 'average':
                    df = self._violate_average(df, idx, agg_col, component_cols)
                elif agg_type == 'count':
                    df = self._violate_count(df, idx, agg_col, component_cols)
                elif agg_type == 'max':
                    df = self._violate_max(df, idx, agg_col, component_cols)
                elif agg_type == 'min':
                    df = self._violate_min(df, idx, agg_col, component_cols)
        
        return df
    
    def _violate_sum(self, df, idx, agg_col, component_cols):
        """Make aggregate not equal to sum of components."""
        # Calculate correct sum
        component_values = [df.at[idx, col] for col in component_cols 
                           if not pd.isna(df.at[idx, col])]
        
        if not component_values:
            return df
        
        correct_sum = sum(component_values)
        
        # Violate by adding/subtracting a percentage
        violation = correct_sum * self.violation_factor * random.choice([-1, 1])
        df.at[idx, agg_col] = correct_sum + violation
        
        return df
    
    def _violate_average(self, df, idx, agg_col, component_cols):
        """Make aggregate not equal to average of components."""
        component_values = [df.at[idx, col] for col in component_cols 
                           if not pd.isna(df.at[idx, col])]
        
        if not component_values:
            return df
        
        correct_avg = sum(component_values) / len(component_values)
        
        # Violate by adding/subtracting a percentage
        violation = correct_avg * self.violation_factor * random.choice([-1, 1])
        df.at[idx, agg_col] = correct_avg + violation
        
        return df
    
    def _violate_count(self, df, idx, agg_col, component_cols):
        """Make count not match number of non-null components."""
        component_values = [df.at[idx, col] for col in component_cols 
                           if not pd.isna(df.at[idx, col])]
        
        correct_count = len(component_values)
        
        # Violate by adding/subtracting 1-3
        violation = random.randint(1, 3) * random.choice([-1, 1])
        df.at[idx, agg_col] = max(0, correct_count + violation)
        
        return df
    
    def _violate_max(self, df, idx, agg_col, component_cols):
        """Make aggregate not equal to maximum of components."""
        component_values = [df.at[idx, col] for col in component_cols 
                           if not pd.isna(df.at[idx, col])]
        
        if not component_values:
            return df
        
        correct_max = max(component_values)
        
        # Set to something less than the correct max
        violation = abs(correct_max) * self.violation_factor
        df.at[idx, agg_col] = correct_max - violation
        
        return df
    
    def _violate_min(self, df, idx, agg_col, component_cols):
        """Make aggregate not equal to minimum of components."""
        component_values = [df.at[idx, col] for col in component_cols 
                           if not pd.isna(df.at[idx, col])]
        
        if not component_values:
            return df
        
        correct_min = min(component_values)
        
        # Set to something more than the correct min
        violation = abs(correct_min) * self.violation_factor if correct_min != 0 else 1
        df.at[idx, agg_col] = correct_min + violation
        
        return df

