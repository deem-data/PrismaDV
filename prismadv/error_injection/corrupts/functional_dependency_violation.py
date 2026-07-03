import random

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class FunctionalDependencyViolation(TabularCorruption):
    """Break table-local key -> dependent value relationships with valid-looking values."""

    def __init__(
        self,
        columns=None,
        severity=None,
        sampling=None,
        key_columns=None,
        dependent_columns=None,
        random_state=None,
        **kwargs,
    ):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.key_columns = key_columns
        self.dependent_columns = dependent_columns
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def identify_columns(self, dataframe: pd.DataFrame):
        raise NotImplementedError(
            "FunctionalDependencyViolation requires explicit key_columns and dependent_columns."
        )

    def transform(self, dataframe: pd.DataFrame):
        self.validate_data(dataframe)
        if not self.key_columns or not self.dependent_columns:
            raise ValueError("key_columns and dependent_columns must be specified")

        missing_columns = [
            column
            for column in [*self.key_columns, *self.dependent_columns]
            if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(f"missing columns for functional dependency corruption: {missing_columns}")

        if self.severity is None or self.severity <= 0:
            return dataframe

        df = dataframe.copy(deep=True)
        sampled_rows = self.sample_rows(df)
        valid_tuples = self._valid_tuples(dataframe)
        donor_indices = list(dataframe.index)

        for idx in sampled_rows:
            if idx not in df.index:
                continue
            replacement = self._replacement_values(dataframe, idx, donor_indices, valid_tuples)
            if replacement is None:
                continue
            for column, value in replacement.items():
                df.at[idx, column] = value

        return df

    def _valid_tuples(self, dataframe: pd.DataFrame) -> set[tuple]:
        columns = [*self.key_columns, *self.dependent_columns]
        return set(dataframe[columns].dropna().itertuples(index=False, name=None))

    def _replacement_values(
        self,
        dataframe: pd.DataFrame,
        idx,
        donor_indices: list,
        valid_tuples: set[tuple],
    ) -> dict | None:
        key_tuple = tuple(dataframe.at[idx, column] for column in self.key_columns)
        original_dependent_tuple = tuple(dataframe.at[idx, column] for column in self.dependent_columns)
        candidate_indices = donor_indices.copy()
        random.shuffle(candidate_indices)

        for donor_idx in candidate_indices:
            if donor_idx == idx:
                continue
            donor_key_tuple = tuple(dataframe.at[donor_idx, column] for column in self.key_columns)
            if donor_key_tuple == key_tuple:
                continue
            donor_dependent_tuple = tuple(dataframe.at[donor_idx, column] for column in self.dependent_columns)
            if donor_dependent_tuple == original_dependent_tuple:
                continue
            candidate_tuple = (*key_tuple, *donor_dependent_tuple)
            if candidate_tuple in valid_tuples:
                continue
            return {
                column: value
                for column, value in zip(self.dependent_columns, donor_dependent_tuple, strict=True)
            }
        return None
