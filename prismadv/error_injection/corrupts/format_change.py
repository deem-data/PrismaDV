import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class FormatChange(TabularCorruption):
    """Rewrite values in a new representation that keeps their meaning.

    The other format corruptions (StringNoise, RegexPatternViolation,
    DateFormatCorruption) make values invalid. This one makes them valid in a
    form the consumer was not built for: a path that arrives as a full URL
    (GOV.UK, February 2018), an IPv4 address that arrives as an IPv4-mapped
    IPv6 address (GitHub, January 2024), a date column that switches from
    ISO to day-first. Every touched value can be mapped back to the original
    one, so a check on single values finds nothing wrong.

    Strategies, each with the one parameter it reads:
        - 'prefix'        prepend `prefix` to the value
        - 'suffix'        append `suffix` to the value
        - 'date_format'   re-render a date with strftime `date_format`
        - 'case'          change letter case; `case` is 'upper', 'lower'
                          or 'title'
        - 'number_format' render a number with the Python format spec
                          `number_format`, for example ',.2f' or '.3e'

    Rows are chosen by the usual sampler; the natural shape is the whole
    column (`severity: 1`), and a slice when only one source changed its
    format. Null values are left as they are. The touched column becomes a
    string column.
    """

    STRATEGIES = ("prefix", "suffix", "date_format", "case", "number_format")
    CASES = ("upper", "lower", "title")

    def __init__(self, columns, severity=None, sampling=None, strategy=None,
                 prefix=None, suffix=None, date_format=None, case=None,
                 number_format=None, random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling,
                         random_state=random_state, **kwargs)
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {list(self.STRATEGIES)}")
        self.strategy = strategy
        self.prefix = prefix
        self.suffix = suffix
        self.date_format = date_format
        self.case = case
        self.number_format = number_format
        needed = {"prefix": prefix, "suffix": suffix, "date_format": date_format,
                  "case": case, "number_format": number_format}[strategy]
        if needed is None:
            raise ValueError(f"strategy '{strategy}' needs the parameter '{strategy}'")
        if strategy == "case" and case not in self.CASES:
            raise ValueError(f"case must be one of {list(self.CASES)}")
        if strategy == "number_format":
            format(1.0, number_format)  # raises ValueError on a bad spec

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

    def identify_columns(self, dataframe):
        raise NotImplementedError("FormatChange needs the columns to be named in the config.")

    def transform(self, dataframe: pd.DataFrame):
        self.validate_data(dataframe)
        df = dataframe.copy(deep=True)
        rows = self.sample_rows(df)
        for col in self._target_columns():
            if col not in df.columns:
                raise ValueError(f"column '{col}' is not a column of the data")
            df[col] = df[col].astype("object")
            df.loc[rows, col] = df.loc[rows, col].map(self._rewrite)
        return df

    def _rewrite(self, value):
        if pd.isna(value):
            return value
        if self.strategy == "prefix":
            return f"{self.prefix}{value}"
        if self.strategy == "suffix":
            return f"{value}{self.suffix}"
        if self.strategy == "case":
            return getattr(str(value), self.case)()
        if self.strategy == "number_format":
            return format(value, self.number_format)
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return value
        return parsed.strftime(self.date_format)
