import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class FieldSwap(TabularCorruption):
    """Put valid values of one attribute under another attribute.

    Takes exactly two columns [A, B]. Strategies:
        - 'swap': A and B exchange their values (a wrong column order in one
                  export or form)
        - 'copy': A takes the value of B, and B keeps it (a list of site ids
                  given where app ids were expected: Atlassian, April 2022)

    Every value stays a valid value of the other column, so a check on single
    values finds nothing wrong. Rows are chosen by the usual sampler.
    """

    STRATEGIES = ("swap", "copy")

    def __init__(self, columns, severity=None, sampling=None, strategy="swap",
                 random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling,
                         random_state=random_state, **kwargs)
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {list(self.STRATEGIES)}")
        if not columns or len(columns) != 2:
            raise ValueError("FieldSwap needs exactly two columns")
        self.strategy = strategy

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

    def identify_columns(self, dataframe):
        raise NotImplementedError("FieldSwap needs the two columns to be named in the config.")

    def transform(self, dataframe: pd.DataFrame):
        self.validate_data(dataframe)
        a, b = self.columns
        for col in (a, b):
            if col not in dataframe.columns:
                raise ValueError(f"column '{col}' is not a column of the data")
        df = dataframe.copy(deep=True)
        rows = self.sample_rows(df)
        values_a = dataframe.loc[rows, a].to_numpy()
        values_b = dataframe.loc[rows, b].to_numpy()
        if df[a].dtype != df[b].dtype:
            df[a] = df[a].astype("object")
            if self.strategy == "swap":
                df[b] = df[b].astype("object")
        df.loc[rows, a] = values_b
        if self.strategy == "swap":
            df.loc[rows, b] = values_a
        return df
