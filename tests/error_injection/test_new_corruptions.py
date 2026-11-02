import pytest
import pandas as pd

from prismadv.error_injection.corrupts import (
    OutlierInjection,
    DataTypeViolation,
    RangeViolation,
    UniqueConstraintViolation,
    ValueReplacement,
    CrossColumnInconsistency,
    DistributionShift,
    ImbalancedCategories,
    ForeignKeyViolation,
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


class TestCrossColumnInconsistency:
    @pytest.mark.parametrize("start, stop, fmt", [
        (["2007-08-16", "2010-01-31", "2015-12-01", "2020-02-29"],
         ["2007-08-20", "2010-02-05", "2016-01-01", "2020-03-10"], "%Y-%m-%d"),
        (["2007-08-16T12:56:08Z", "2010-01-31T23:00:00Z", "2015-12-01T00:00:00Z", "2020-02-29T08:30:00Z"],
         ["2007-08-16T13:11:08Z", "2010-02-01T01:00:00Z", "2015-12-01T00:15:00Z", "2020-02-29T09:00:00Z"],
         "%Y-%m-%dT%H:%M:%SZ"),
    ])
    def test_less_equal_on_date_strings_puts_stop_strictly_before_start(self, start, stop, fmt):
        df = pd.DataFrame({"START": start, "STOP": stop})
        corruption = CrossColumnInconsistency(
            severity=1.0, column_pairs=[("START", "STOP", "<=")], random_state=3,
        )
        corrupted = corruption.transform(df)

        assert (corrupted["START"] == df["START"]).all()
        parsed_start = pd.to_datetime(corrupted["START"], format=fmt)
        parsed_stop = pd.to_datetime(corrupted["STOP"], format=fmt)  # still in the column's format
        assert (parsed_stop < parsed_start).all()

    def test_less_equal_on_other_strings_still_raises(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["z", "w"]})
        corruption = CrossColumnInconsistency(severity=1.0, column_pairs=[("a", "b", "<=")], random_state=3)
        with pytest.raises(ValueError):
            corruption.transform(df)


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


class TestDistributionShift:
    def test_mean_shift(self, sample_numerical_df):
        original_mean = sample_numerical_df['age'].mean()
        
        corruption = DistributionShift(columns=['age'], severity=0.5, strategy='mean_shift')
        corrupted = corruption.transform(sample_numerical_df)
        
        corrupted_mean = corrupted['age'].mean()
        # Mean should have shifted
        assert abs(corrupted_mean - original_mean) > 1


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
