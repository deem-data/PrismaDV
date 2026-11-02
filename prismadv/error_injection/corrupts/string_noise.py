import random

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class StringNoise(TabularCorruption):
    def __init__(
            self,
            columns=None,
            severity=None,
            sampling=None,
            mode="case",
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
            random_state=None,
            **kwargs
    ):
        super().__init__(columns=columns, severity=severity, sampling=sampling,
                         random_state=random_state, **kwargs)
        self.mode = mode
        self.alphabet = list(alphabet)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

    def identify_columns(self, dataframe: pd.DataFrame):
        str_cols = []
        for c in dataframe.columns:
            dt = dataframe[c].dtype
            if pd.api.types.is_string_dtype(dt) or pd.api.types.is_object_dtype(dt):
                str_cols.append(c)
        if not str_cols:
            raise ValueError("No string-like columns found for StringNoise.")
        self.columns = str_cols
        return str_cols

    def transform(self, dataframe: pd.DataFrame):
        self.validate_data(dataframe)
        if not self.columns:
            self.identify_columns(dataframe)

        if self.severity is None or self.severity <= 0:
            raise ValueError("severity must be larger than 0")

        df = dataframe.copy(deep=True)
        rows = self.sample_rows(df)

        for col in self.columns:
            series = df.loc[rows, col]
            series = series.apply(self._apply_noise_safe)
            df.loc[rows, col] = series
        return df

    def _apply_noise_safe(self, x):
        if pd.isna(x):
            return x
        s = str(x)
        p = float(np.clip(self.severity, 0.0, 1.0))
        if self.mode == "case":
            return self._flip_case(s, p)
        elif self.mode == "insert":
            return self._insert_letters(s, p)
        elif self.mode == "transpose":
            return self._transpose(x, s)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    @staticmethod
    def _flip_case(s: str, prob: float):
        out = []
        for ch in s:
            if ch.isalpha() and random.random() < prob:
                out.append(ch.swapcase())
            else:
                out.append(ch)
        # `prob` is the row-level severity reused per character, so at a few
        # percent almost every sampled cell used to come back unchanged. A
        # sampled cell must carry the error: flip at least one letter.
        alpha_positions = [i for i, ch in enumerate(s) if ch.isalpha()]
        if alpha_positions and out == list(s):
            i = random.choice(alpha_positions)
            out[i] = s[i].swapcase()
        return "".join(out)

    @staticmethod
    def _transpose(x, s: str):
        # Swap two adjacent, different characters: letters in a string, digits
        # in a number (never the sign or the decimal point). A number stays a
        # number of its own type: 142.5 -> 124.5.
        is_number = isinstance(x, (int, float, np.number)) and not isinstance(x, (bool, np.bool_))
        same_kind = str.isdigit if is_number else str.isalpha
        pairs = [i for i in range(len(s) - 1)
                 if same_kind(s[i]) and same_kind(s[i + 1]) and s[i] != s[i + 1]]
        if not pairs:
            return x
        i = random.choice(pairs)
        out = s[:i] + s[i + 1] + s[i] + s[i + 2:]
        if is_number:
            return int(out) if isinstance(x, (int, np.integer)) else float(out)
        return out

    def _insert_letters(self, s: str, rate: float):
        if len(s) == 0 or rate <= 0:
            return s

        base = max(1, int(np.ceil(len(s) * rate)))
        k = max(1, int(np.round(base * random.uniform(0.7, 1.3))))

        if len(s) >= 3:
            positions = list(range(1, len(s)))
        else:
            positions = list(range(0, len(s) + 1))

        insert_positions = sorted(random.choices(positions, k=k))
        chars = list(s)
        shift = 0
        for pos in insert_positions:
            ins = random.choice(self.alphabet)
            chars.insert(pos + shift, ins)
            shift += 1
        return "".join(chars)
