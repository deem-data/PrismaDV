import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class DuplicatedRows(TabularCorruption):
    def __init__(self, columns=None, severity=None, sampling=None, keep_index=False, random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling, **kwargs)
        self.keep_index = keep_index
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

    def identify_columns(self, dataframe):
        raise NotImplementedError("DuplicatedRows does not require column identification.")

    def transform(self, dataframe: pd.DataFrame):
        self.validate_data(dataframe)
        if self.severity is None or self.severity <= 0:
            return dataframe

        df = dataframe.copy(deep=True)
        use_super = True
        if (self.sampling and (self.sampling.endswith('AR') or self.sampling.endswith('NAR'))) and not self.columns:
            use_super = False
        if use_super:
            rows = super().sample_rows(df)
        else:
            n = int(len(df) * min(self.severity, 1.0))
            rows = np.random.permutation(df.index)[:n]
        dup_block = df.loc[rows]
        out = pd.concat([df, dup_block], axis=0)
        if not self.keep_index:
            out = out.reset_index(drop=True)
        return out
