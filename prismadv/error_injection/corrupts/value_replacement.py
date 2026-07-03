import random
from collections.abc import Sequence

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class ValueReplacement(TabularCorruption):
    """Replace sampled cells with explicit configured values.

    This corruption is useful for semantic EIDBench-real errors where the bad value is
    meaningful, for example sentinel misuse or unseen categorical levels.
    """

    def __init__(
        self,
        columns=None,
        severity=None,
        sampling=None,
        replacement_values=None,
        strategy="cycle",
        template=None,
        random_state=None,
        **kwargs,
    ):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.replacement_values = replacement_values
        self.strategy = strategy
        self.template = template
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        valid_strategies = ["constant", "cycle", "random", "unique_template"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
        if self.strategy == "unique_template" and not self.template:
            raise ValueError("template is required for unique_template strategy")
        if self.strategy != "unique_template" and replacement_values is None:
            raise ValueError("replacement_values is required unless strategy is unique_template")

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

    def identify_columns(self, dataframe: pd.DataFrame):
        raise NotImplementedError("ValueReplacement requires explicit columns.")

    def transform(self, dataframe: pd.DataFrame):
        self.validate_data(dataframe)
        if not self.columns:
            raise ValueError("ValueReplacement requires explicit columns.")
        if self.severity is None or self.severity <= 0:
            return dataframe

        df = dataframe.copy(deep=True)
        rows = list(self.sample_rows(df))

        for col in self.columns:
            if col not in df.columns:
                continue
            df[col] = df[col].astype("object")
            for offset, idx in enumerate(rows):
                if idx not in df.index:
                    continue
                df.at[idx, col] = self._replacement_for(col, idx, offset, df.at[idx, col])

        return df

    def _replacement_for(self, column, row_index, offset, original_value):
        if self.strategy == "unique_template":
            return self.template.format(
                column=column,
                row_index=row_index,
                row_number=offset + 1,
                original=original_value,
            )

        values = self._values_for_column(column)
        if self.strategy == "constant":
            return values[0]
        if self.strategy == "cycle":
            return values[offset % len(values)]
        return random.choice(values)

    def _values_for_column(self, column):
        values = self.replacement_values
        if isinstance(values, dict):
            if column in values:
                values = values[column]
            elif "__default__" in values:
                values = values["__default__"]
            else:
                raise ValueError(f"replacement_values missing column: {column}")

        if isinstance(values, str) or not isinstance(values, Sequence):
            values = [values]
        values = list(values)
        if not values:
            raise ValueError("replacement_values must not be empty")
        return values
