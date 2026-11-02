# https://github.com/schelterlabs/jenga/blob/master/src/jenga/basis.py
import random
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class DataCorruption(ABC):
    def __init__(self, **kwargs):
        # Every parameter a config names must be consumed by some class in the
        # chain. Anything left over here was silently ignored before (a
        # `factor` on Scaling, `max_unique_num`, `truncation_length`, ...), so
        # the config described a corruption that never happened.
        if kwargs:
            raise TypeError(
                f"{type(self).__name__} does not accept parameter(s) {sorted(kwargs)}"
            )

    @abstractmethod
    def transform(self, data):
        pass


class TabularCorruption(DataCorruption):
    def __init__(self, columns=None, severity=None, sampling=None, random_state=None,
                 slice_column=None, slice_values=None, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.severity = 1.0 if severity is None else severity
        self.sampling = 'CAR' if sampling is None else sampling
        if (slice_column is None) != (slice_values is None):
            raise ValueError("slice_column and slice_values must be given together")
        self.slice_column = slice_column
        self.slice_values = slice_values
        self.seed(random_state)

    def seed(self, random_state):
        """Seed both RNGs the corruption classes draw from.

        Row sampling and every value generator use the module-level `random`
        and `np.random` state, so a corruption is reproducible only if that
        state is set right before it runs. Managers call this before each
        `transform`; the seed is also recorded on the instance so the saved
        config carries it.
        """
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def to_dict(self):
        params = {
            k: v for k, v in self.__dict__.items()
            if k != "columns" and not k.startswith("_") and v is not None
        }
        return {
            self.__class__.__name__: {
                "Columns": self.columns,
                "Params": params,
            }
        }

    @abstractmethod
    def transform(self, dataframe: pd.DataFrame):
        pass

    @abstractmethod
    def identify_columns(self, dataframe):
        pass

    @staticmethod
    def validate_data(dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

    @staticmethod
    def is_integer_column(series: pd.Series) -> bool:
        return pd.api.types.is_integer_dtype(series)

    def _target_columns(self):
        if self.columns is None:
            return []
        if isinstance(self.columns, str):
            return [self.columns]
        return list(self.columns)

    def population(self, data):
        """The rows a corruption may touch.

        The whole batch by default, or only the rows whose `slice_column`
        value is in `slice_values`. Severity is a share of this population:
        `severity: 0.5` with the slice "city in [Mumbai]" corrupts half of the
        Mumbai rows and nothing else. That is the shape of most production
        incidents (one region, one client, one experiment arm), whereas the
        default samples the batch completely at random.
        """
        if self.slice_column is None:
            return data
        if self.slice_column not in data.columns:
            raise ValueError(f"slice_column '{self.slice_column}' is not a column of the data")
        values = self.slice_values
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, set)):
            values = [values]
        population = data[data[self.slice_column].isin(list(values))]
        if population.empty:
            raise ValueError(f"slice {self.slice_column} in {list(values)} matches no row")
        return population

    def sample_rows(self, data):
        data = self.population(data)
        if self.severity == 1.0:
            rows = data.index
        # Completely At Random
        elif self.sampling.endswith('CAR'):
            rows = np.random.permutation(data.index)[:int(len(data) * self.severity)]
        elif self.sampling.endswith('NAR') or self.sampling.endswith('AR'):
            n_values_to_discard = int(len(data) * min(self.severity, 1.0))
            perc_lower_start = np.random.randint(0, len(data) - n_values_to_discard)
            perc_idx = range(perc_lower_start, perc_lower_start + n_values_to_discard)
            targets = self._target_columns()
            # Not At Random: rows are chosen by the corrupted column's own order
            if self.sampling.endswith('NAR'):
                if not targets:
                    raise ValueError("NAR sampling needs at least one target column")
                rows = data[targets[0]].sort_values().iloc[perc_idx].index
            # At Random: rows are chosen by the order of some other column
            else:
                candidates = sorted(set(data.columns) - set(targets))
                if not candidates:
                    raise ValueError("AR sampling needs a column other than the targets")
                depends_on_col = np.random.choice(candidates)
                rows = data[depends_on_col].sort_values().iloc[perc_idx].index

        else:
            raise ValueError(f"sampling type '{self.sampling}' not recognized")

        return rows
