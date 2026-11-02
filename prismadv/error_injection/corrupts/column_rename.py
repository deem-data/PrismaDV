import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class ColumnRename(TabularCorruption):
    """Give a column a new name and keep its values.

    The schema counterpart of FormatChange: nothing in the data is wrong, but
    a consumer that reads the column by its old name finds it missing, and a
    consumer that reads by position gets the right values under a name it
    does not check. Strategies:
        - 'explicit'   `new_names` gives the new name of each column in
                       `columns`, in the same order
        - 'add_prefix' prepend `prefix` (default 'new_') to each name
        - 'snake_case' lower-case the name and replace spaces with '_'
        - 'upper'      upper-case the name

    A new name must not collide with another column of the table.
    """

    STRATEGIES = ("explicit", "add_prefix", "snake_case", "upper")

    def __init__(self, columns, severity=1, strategy="explicit", new_names=None,
                 prefix=None, **kwargs):
        super().__init__(columns, severity=severity, **kwargs)
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {list(self.STRATEGIES)}")
        self.strategy = strategy
        self.new_names = new_names
        self.prefix = prefix
        targets = self._target_columns()
        if not targets:
            raise ValueError("ColumnRename needs at least one column")
        if strategy == "explicit":
            if isinstance(new_names, str):
                self.new_names = new_names = [new_names]
            if not new_names or len(new_names) != len(targets):
                raise ValueError("new_names must list one new name per column")
        if strategy == "add_prefix" and prefix is None:
            self.prefix = "new_"

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

    def identify_columns(self, dataframe):
        return self.columns

    def mapping(self):
        targets = self._target_columns()
        if self.strategy == "explicit":
            new = list(self.new_names)
        elif self.strategy == "add_prefix":
            new = [f"{self.prefix}{c}" for c in targets]
        elif self.strategy == "snake_case":
            new = [c.strip().lower().replace(" ", "_") for c in targets]
        else:
            new = [c.upper() for c in targets]
        return dict(zip(targets, new))

    def transform(self, dataframe: pd.DataFrame, **kwargs):
        self.validate_data(dataframe)
        mapping = self.mapping()
        for old, new in mapping.items():
            if old not in dataframe.columns:
                raise ValueError(f"column '{old}' is not a column of the data")
            if new == old:
                raise ValueError(f"column '{old}' would keep its name")
            if new in dataframe.columns and new not in mapping:
                raise ValueError(f"new name '{new}' is already a column of the data")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("two columns would get the same new name")
        return dataframe.copy(deep=True).rename(columns=mapping)
