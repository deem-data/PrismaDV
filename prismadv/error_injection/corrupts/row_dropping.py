import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class RowDropping(TabularCorruption):
    """Remove rows from the batch.

    Strategies:
        - 'random': drop `severity` of the rows (of the slice, when one is
          configured), chosen by the usual row sampler.
        - 'block': drop one contiguous run of rows in file order covering
          `severity` of the batch. This is the shape of a load that stopped
          part-way or a day of records that never arrived (Public Health
          England, October 2020). The start offset is seeded.

    The counterpart of DuplicatedRows: the batch keeps its columns and loses
    records, which nothing on the cell level can reveal.
    """

    def __init__(self, columns=None, severity=None, sampling=None, strategy="random",
                 random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling,
                         random_state=random_state, **kwargs)
        if strategy not in ("random", "block"):
            raise ValueError("strategy must be 'random' or 'block'")
        self.strategy = strategy

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

    def identify_columns(self, dataframe):
        raise NotImplementedError("RowDropping drops whole rows; it names no column.")

    def transform(self, dataframe: pd.DataFrame):
        self.validate_data(dataframe)
        df = dataframe.copy(deep=True)
        if self.strategy == "block":
            population = self.population(df)
            n = int(len(population) * min(self.severity, 1.0))
            if n == 0:
                return df
            start = int(np.random.randint(0, len(population) - n + 1))
            rows = population.index[start:start + n]
        else:
            rows = self.sample_rows(df)
        return df.drop(index=rows).reset_index(drop=True)
