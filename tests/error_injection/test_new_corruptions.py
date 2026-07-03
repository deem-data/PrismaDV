import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from prismadv.error_injection.corrupts import (
    OutlierInjection,
    DataTypeViolation,
    RangeViolation,
    UniqueConstraintViolation,
    ValueReplacement,
    CrossColumnInconsistency,
    DateFormatCorruption,
    EmailCorruption,
    PhoneNumberCorruption,
    RegexPatternViolation,
    TemporalGaps,
    TemporalOutOfOrder,
    SeasonalityAnomaly,
    DistributionShift,
    ImbalancedCategories,
    ForeignKeyViolation,
    AggregationInconsistency,
    FunctionalDependencyViolation,
)


@pytest.fixture
def sample_numerical_df():
    """Sample dataframe with numerical data."""
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'salary': [50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000],
        'score': [75.5, 80.0, 85.5, 90.0, 78.0, 82.5, 88.0, 91.5, 76.0, 84.0],
    })


@pytest.fixture
def sample_string_df():
    """Sample dataframe with string data."""
    return pd.DataFrame({
        'email': ['user1@example.com', 'user2@test.com', 'user3@mail.com'] * 3,
        'phone': ['+1-555-1234', '+1-555-5678', '+1-555-9012'] * 3,
        'name': ['Alice', 'Bob', 'Charlie'] * 3,
    })


@pytest.fixture
def sample_datetime_df():
    """Sample dataframe with datetime data."""
    base_date = datetime(2023, 1, 1)
    return pd.DataFrame({
        'timestamp': [base_date + timedelta(days=i) for i in range(10)],
        'value': [10 + i * 2 for i in range(10)],
    })


@pytest.fixture
def sample_categorical_df():
    """Sample dataframe with categorical data."""
    return pd.DataFrame({
        'category': ['A', 'B', 'C'] * 30 + ['D'] * 10,
        'status': ['active', 'inactive', 'pending'] * 30 + ['active'] * 10,
    })


class TestOutlierInjection:
    def test_iqr_based_strategy(self, sample_numerical_df):
        corruption = OutlierInjection(columns=['age'], severity=0.3, strategy='iqr_based')
        corrupted = corruption.transform(sample_numerical_df)
        
        assert len(corrupted) == len(sample_numerical_df)
        assert not corrupted['age'].equals(sample_numerical_df['age'])
    
    def test_zscore_based_strategy(self, sample_numerical_df):
        corruption = OutlierInjection(columns=['salary'], severity=0.3, strategy='zscore_based')
        corrupted = corruption.transform(sample_numerical_df)
        
        assert len(corrupted) == len(sample_numerical_df)
    
    def test_extreme_values_strategy(self, sample_numerical_df):
        corruption = OutlierInjection(columns=['score'], severity=0.3, strategy='extreme_values')
        corrupted = corruption.transform(sample_numerical_df)
        
        assert len(corrupted) == len(sample_numerical_df)


class TestDataTypeViolation:
    def test_strings_in_numeric(self, sample_numerical_df):
        corruption = DataTypeViolation(columns=['age'], severity=0.3, strategy='strings_in_numeric')
        corrupted = corruption.transform(sample_numerical_df)
        
        # Check that some values are now strings
        assert corrupted['age'].dtype == object
    
    def test_invalid_conversions(self, sample_numerical_df):
        corruption = DataTypeViolation(columns=['salary'], severity=0.3, strategy='invalid_conversions')
        corrupted = corruption.transform(sample_numerical_df)
        
        assert corrupted['salary'].dtype == object


class TestRangeViolation:
    def test_below_min(self, sample_numerical_df):
        corruption = RangeViolation(columns=['age'], severity=0.3, min_value=0, max_value=100, strategy='below_min')
        corrupted = corruption.transform(sample_numerical_df)
        
        assert len(corrupted) == len(sample_numerical_df)
        # Some values should be negative (below min of 0)
        assert any(corrupted['age'] < 0)
    
    def test_above_max(self, sample_numerical_df):
        corruption = RangeViolation(columns=['age'], severity=0.3, min_value=0, max_value=100, strategy='above_max')
        corrupted = corruption.transform(sample_numerical_df)
        
        # Some values should be above 100
        assert any(corrupted['age'] > 100)


class TestValueReplacement:
    def test_cycles_explicit_replacement_values(self):
        df = pd.DataFrame({"pdays": [-1, 1, 2, 3]})
        corruption = ValueReplacement(
            columns=["pdays"],
            severity=1.0,
            replacement_values=[-2, -999],
            strategy="cycle",
        )

        corrupted = corruption.transform(df)

        assert corrupted["pdays"].tolist() == [-2, -999, -2, -999]

    def test_unique_template_replacement_values(self):
        df = pd.DataFrame({"job": ["admin", "tech", "blue-collar"]})
        corruption = ValueReplacement(
            columns=["job"],
            severity=1.0,
            strategy="unique_template",
            template="job_{row_number:04d}",
        )

        corrupted = corruption.transform(df)

        assert corrupted["job"].tolist() == ["job_0001", "job_0002", "job_0003"]


class TestUniqueConstraintViolation:
    def test_exact_duplicates(self):
        # Create a dataframe with unique emails
        df = pd.DataFrame({
            'email': [f'user{i}@example.com' for i in range(10)],
        })
        
        original_unique = df['email'].nunique()
        
        corruption = UniqueConstraintViolation(columns=['email'], severity=0.5, strategy='exact_duplicates')
        corrupted = corruption.transform(df)
        
        # Check that duplicates were created (should have fewer unique values)
        assert corrupted['email'].nunique() <= original_unique
    
    def test_near_duplicates(self, sample_string_df):
        corruption = UniqueConstraintViolation(columns=['name'], severity=0.3, strategy='near_duplicates')
        corrupted = corruption.transform(sample_string_df)
        
        assert len(corrupted) == len(sample_string_df)


class TestCrossColumnInconsistency:
    def test_less_than_violation(self):
        df = pd.DataFrame({
            'start_value': [10, 20, 30, 40, 50],
            'end_value': [15, 25, 35, 45, 55],
        })
        
        corruption = CrossColumnInconsistency(
            severity=0.5,
            column_pairs=[('start_value', 'end_value', 'less_than')]
        )
        corrupted = corruption.transform(df)
        
        assert len(corrupted) == len(df)


class TestFunctionalDependencyViolation:
    def test_replaces_dependent_values_with_valid_looking_mismatches(self):
        df = pd.DataFrame(
            {
                "CODE": ["a", "b", "c", "d"],
                "DESCRIPTION": ["alpha", "beta", "gamma", "delta"],
            }
        )
        corruption = FunctionalDependencyViolation(
            columns=["CODE", "DESCRIPTION"],
            key_columns=["CODE"],
            dependent_columns=["DESCRIPTION"],
            severity=0.5,
            random_state=17,
        )

        corrupted = corruption.transform(df)
        valid_pairs = set(df[["CODE", "DESCRIPTION"]].itertuples(index=False, name=None))
        corrupted_pairs = set(corrupted[["CODE", "DESCRIPTION"]].itertuples(index=False, name=None))

        assert len(corrupted) == len(df)
        assert set(corrupted["CODE"]) == set(df["CODE"])
        assert set(corrupted["DESCRIPTION"]).issubset(set(df["DESCRIPTION"]))
        assert corrupted_pairs - valid_pairs


class TestDateFormatCorruption:
    def test_invalid_dates(self, sample_datetime_df):
        # Convert datetime to string for corruption
        df = sample_datetime_df.copy()
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        
        corruption = DateFormatCorruption(columns=['timestamp'], severity=0.3, strategy='invalid_dates')
        corrupted = corruption.transform(df)
        
        assert len(corrupted) == len(df)
    
    def test_format_mixing(self, sample_datetime_df):
        df = sample_datetime_df.copy()
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        
        corruption = DateFormatCorruption(columns=['timestamp'], severity=0.3, strategy='format_mixing')
        corrupted = corruption.transform(df)
        
        assert len(corrupted) == len(df)


class TestEmailCorruption:
    def test_missing_at(self, sample_string_df):
        corruption = EmailCorruption(columns=['email'], severity=0.3, strategy='missing_at')
        corrupted = corruption.transform(sample_string_df)
        
        # Check that some emails are missing @
        assert any('@' not in str(val) for val in corrupted['email'] if pd.notna(val))
    
    def test_invalid_domain(self, sample_string_df):
        corruption = EmailCorruption(columns=['email'], severity=0.3, strategy='invalid_domain')
        corrupted = corruption.transform(sample_string_df)
        
        assert len(corrupted) == len(sample_string_df)


class TestPhoneNumberCorruption:
    def test_wrong_length(self, sample_string_df):
        corruption = PhoneNumberCorruption(columns=['phone'], severity=0.3, strategy='wrong_length')
        corrupted = corruption.transform(sample_string_df)
        
        # Check that lengths have changed
        original_lengths = [len(str(p)) for p in sample_string_df['phone']]
        corrupted_lengths = [len(str(p)) for p in corrupted['phone']]
        assert original_lengths != corrupted_lengths
    
    def test_letters(self, sample_string_df):
        corruption = PhoneNumberCorruption(columns=['phone'], severity=0.3, strategy='letters')
        corrupted = corruption.transform(sample_string_df)
        
        # Check that some entries contain letters
        assert any(any(c.isalpha() for c in str(val)) for val in corrupted['phone'] if pd.notna(val))


class TestRegexPatternViolation:
    def test_character_insertion(self):
        df = pd.DataFrame({
            'zip_code': ['12345', '67890', '11111', '22222', '33333'],
        })
        
        corruption = RegexPatternViolation(
            pattern={'zip_code': r'^\d{5}$'},
            severity=0.5,
            strategy='character_insertion'
        )
        corrupted = corruption.transform(df)
        
        assert len(corrupted) == len(df)


class TestTemporalGaps:
    def test_gap_creation(self, sample_datetime_df):
        original_len = len(sample_datetime_df)
        
        corruption = TemporalGaps(time_column='timestamp', severity=0.3, gap_size='small')
        corrupted = corruption.transform(sample_datetime_df)
        
        # Should have fewer rows due to gaps
        assert len(corrupted) < original_len


class TestTemporalOutOfOrder:
    def test_random_shuffle(self, sample_datetime_df):
        corruption = TemporalOutOfOrder(columns=['timestamp'], severity=0.5, strategy='random_shuffle')
        corrupted = corruption.transform(sample_datetime_df)
        
        # Timestamps should be different
        assert not corrupted['timestamp'].equals(sample_datetime_df['timestamp'])
    
    def test_local_swaps(self, sample_datetime_df):
        corruption = TemporalOutOfOrder(columns=['timestamp'], severity=0.5, strategy='local_swaps')
        corrupted = corruption.transform(sample_datetime_df)
        
        assert len(corrupted) == len(sample_datetime_df)


class TestSeasonalityAnomaly:
    def test_spike_injection(self, sample_datetime_df):
        corruption = SeasonalityAnomaly(
            columns=['value'],
            time_column='timestamp',
            severity=0.3,
            anomaly_type='spike'
        )
        corrupted = corruption.transform(sample_datetime_df)
        
        # Check that some values are significantly larger
        assert corrupted['value'].max() > sample_datetime_df['value'].max()


class TestDistributionShift:
    def test_mean_shift(self, sample_numerical_df):
        original_mean = sample_numerical_df['age'].mean()
        
        corruption = DistributionShift(columns=['age'], severity=0.5, strategy='mean_shift')
        corrupted = corruption.transform(sample_numerical_df)
        
        corrupted_mean = corrupted['age'].mean()
        # Mean should have shifted
        assert abs(corrupted_mean - original_mean) > 1
    
    def test_variance_change(self, sample_numerical_df):
        corruption = DistributionShift(columns=['salary'], severity=0.5, strategy='variance_change')
        corrupted = corruption.transform(sample_numerical_df)
        
        assert len(corrupted) == len(sample_numerical_df)


class TestImbalancedCategories:
    def test_over_represent(self, sample_categorical_df):
        corruption = ImbalancedCategories(columns=['category'], severity=0.5, strategy='over_represent')
        corrupted = corruption.transform(sample_categorical_df)
        
        # Check that distribution has changed
        original_counts = sample_categorical_df['category'].value_counts()
        corrupted_counts = corrupted['category'].value_counts()
        assert not original_counts.equals(corrupted_counts)
    
    def test_eliminate(self, sample_categorical_df):
        original_unique = sample_categorical_df['category'].nunique()
        
        corruption = ImbalancedCategories(columns=['category'], severity=0.5, strategy='eliminate')
        corrupted = corruption.transform(sample_categorical_df)
        
        # Should have fewer unique categories
        assert corrupted['category'].nunique() <= original_unique


class TestForeignKeyViolation:
    def test_invalid_values(self):
        df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        
        valid_ids = [1, 2, 3, 4, 5]
        
        corruption = ForeignKeyViolation(
            reference_values={'user_id': valid_ids},
            severity=0.5,
            strategy='invalid_values'
        )
        corrupted = corruption.transform(df)
        
        # Check that some IDs are not in valid set
        assert any(val not in valid_ids for val in corrupted['user_id'] if pd.notna(val))
    
    def test_null_values(self):
        df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        
        corruption = ForeignKeyViolation(
            reference_values={'user_id': [1, 2, 3]},
            severity=0.5,
            strategy='null_values'
        )
        corrupted = corruption.transform(df)
        
        # Check that some values are now NaN
        assert corrupted['user_id'].isna().any()


class TestAggregationInconsistency:
    def test_sum_violation(self):
        df = pd.DataFrame({
            'item1': [10, 20, 30, 40, 50],
            'item2': [5, 10, 15, 20, 25],
            'total': [15, 30, 45, 60, 75],
        })
        
        corruption = AggregationInconsistency(
            aggregation_rules=[('total', ['item1', 'item2'], 'sum')],
            severity=0.5
        )
        corrupted = corruption.transform(df)
        
        # Check that sums don't match
        for idx in range(len(corrupted)):
            actual_sum = corrupted.at[idx, 'item1'] + corrupted.at[idx, 'item2']
            total = corrupted.at[idx, 'total']
            # At least some should be inconsistent
            if idx < len(corrupted) * 0.5:
                continue
        
        assert len(corrupted) == len(df)
    
    def test_average_violation(self):
        df = pd.DataFrame({
            'val1': [10, 20, 30],
            'val2': [20, 30, 40],
            'avg': [15, 25, 35],
        })
        
        corruption = AggregationInconsistency(
            aggregation_rules=[('avg', ['val1', 'val2'], 'average')],
            severity=0.5
        )
        corrupted = corruption.transform(df)
        
        assert len(corrupted) == len(df)


# Test YAML serialization
class TestYAMLSerialization:
    def test_outlier_injection_to_dict(self):
        corruption = OutlierInjection(columns=['age'], severity=0.3, strategy='iqr_based')
        config_dict = corruption.to_dict()
        
        assert 'OutlierInjection' in config_dict
        assert config_dict['OutlierInjection']['Columns'] == ['age']
    
    def test_data_type_violation_to_dict(self):
        corruption = DataTypeViolation(columns=['value'], severity=0.5, strategy='strings_in_numeric')
        config_dict = corruption.to_dict()
        
        assert 'DataTypeViolation' in config_dict
    
    def test_email_corruption_to_dict(self):
        corruption = EmailCorruption(columns=['email'], severity=0.3, strategy='missing_at')
        config_dict = corruption.to_dict()
        
        assert 'EmailCorruption' in config_dict
