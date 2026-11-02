import math
import numbers
import random

import numpy as np
import pandas as pd

from prismadv.error_injection.abstract_corruption import TabularCorruption


class CrossColumnInconsistency(TabularCorruption):
    """
    Violate logical relationships between columns.

    This corruption requires specifying column pairs and their expected relationships.

    Parameters:
        column_pairs: List of tuples (col1, col2, relationship_type)
            Relationship types (the operator symbol is accepted as an alias):
                - 'less_than'     / '<'  : col1 < col2 (e.g., start_date < end_date)
                - 'greater_than'  / '>'  : col1 > col2
                - 'less_equal'    / '<=' : col1 <= col2
                - 'greater_equal' / '>=' : col1 >= col2
                - 'equal'         / '==' : col1 == col2
                - 'not_equal'     / '!=' : col1 != col2

    Examples:
        - end_date < start_date
        - child_age > parent_age
        - withdrawal > balance
        - shipping_cost > total_cost
    """

    RELATIONSHIP_ALIASES = {
        "<": "less_than",
        ">": "greater_than",
        "<=": "less_equal",
        ">=": "greater_equal",
        "==": "equal",
        "=": "equal",
        "!=": "not_equal",
    }
    RELATIONSHIPS = ("less_than", "greater_than", "less_equal", "greater_equal", "equal", "not_equal")

    def __init__(self, columns=None, severity=None, sampling=None,
                 column_pairs=None, random_state=None, **kwargs):
        super().__init__(columns=columns, severity=severity, sampling=sampling,
                         random_state=random_state, **kwargs)
        column_pairs = column_pairs or []

        # Validate column_pairs format
        if not isinstance(column_pairs, list):
            raise ValueError("column_pairs must be a list of tuples")

        self.column_pairs = []
        for pair in column_pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 3:
                raise ValueError(
                    "Each column pair must be a tuple/list of (col1, col2, relationship_type)"
                )
            col1, col2, relationship = pair
            relationship = self.RELATIONSHIP_ALIASES.get(relationship, relationship)
            if relationship not in self.RELATIONSHIPS:
                raise ValueError(
                    f"relationship '{pair[2]}' is not one of {self.RELATIONSHIPS} "
                    f"or their operator aliases {sorted(self.RELATIONSHIP_ALIASES)}"
                )
            self.column_pairs.append([col1, col2, relationship])

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

    def identify_columns(self, dataframe: pd.DataFrame):
        """
        Cannot auto-identify columns for cross-column inconsistencies.
        User must specify column_pairs.
        """
        raise NotImplementedError(
            "CrossColumnInconsistency requires explicit column_pairs specification. "
            "Cannot auto-identify column relationships."
        )

    def transform(self, dataframe: pd.DataFrame):
        self.validate_data(dataframe)

        if not self.column_pairs:
            raise ValueError("column_pairs must be specified for CrossColumnInconsistency")

        if self.severity is None or self.severity <= 0:
            return dataframe

        df = dataframe.copy(deep=True)
        rows = self.sample_rows(df)

        for col1, col2, relationship in self.column_pairs:
            missing = [c for c in (col1, col2) if c not in df.columns]
            if missing:
                raise ValueError(f"column(s) {missing} not in the data")

            integer_target = self.is_integer_column(df[col2])

            # Violate the relationship for sampled rows
            for idx in rows:
                if idx not in df.index:
                    continue

                val1 = df.at[idx, col1]
                val2 = df.at[idx, col2]

                # Skip if either value is NaN
                if pd.isna(val1) or pd.isna(val2):
                    continue

                new_value = self._violate_relationship(val1, val2, relationship, integer_target)
                df.at[idx, col2] = new_value

        return df

    @staticmethod
    def _is_number(value):
        # numpy scalars register with numbers.Number; bools are excluded because
        # True - 0.3 is not a meaningful relationship violation.
        return isinstance(value, numbers.Number) and not isinstance(value, (bool, np.bool_))

    DATE_FORMAT = "%Y-%m-%d"
    TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

    @classmethod
    def _shift_date_string(cls, value, earlier):
        """`value` moved strictly earlier (or later), written in its own format.

        A `yyyy-MM-dd` date moves by 1 to 30 days and a `yyyy-MM-ddTHH:mm:ssZ`
        timestamp by 1 to 72 hours, so the result is still a valid value of its
        column and only the relation to the other column breaks. Any other
        string gives None.
        """
        text = str(value)
        for fmt, unit, high in ((cls.DATE_FORMAT, "days", 30), (cls.TIMESTAMP_FORMAT, "hours", 72)):
            try:
                parsed = pd.to_datetime(text, format=fmt)
            except (ValueError, TypeError):
                continue
            offset = pd.Timedelta(**{unit: random.randint(1, high)})
            return (parsed - offset if earlier else parsed + offset).strftime(fmt)
        return None

    def _violate_relationship(self, val1, val2, relationship, integer_target=False):
        """Violate the specified relationship by modifying val2.

        On integer columns the returned value is an integer strictly on the
        wrong side of the relationship, so the column keeps its dtype and the
        untouched rows are written back unchanged.
        """
        numeric = self._is_number(val1)
        if not numeric:
            # For dates or other comparable types, collapsing col2 onto col1
            # violates the strict relations; the non-strict ones need a value
            # on the far side, which only date strings can give.
            if relationship in ('less_than', 'greater_than', 'not_equal'):
                return val1
            if relationship == 'equal':
                return str(val2) + "_modified"
            if relationship in ('less_equal', 'greater_equal'):
                shifted = self._shift_date_string(val1, earlier=relationship == 'less_equal')
                if shifted is not None:
                    return shifted
            raise ValueError(
                f"cannot violate '{relationship}' on non-numeric values ({val1!r}, {val2!r})"
            )

        magnitude = abs(val1) * random.uniform(0.1, 0.5)

        if relationship == 'less_than':
            # col1 < col2 is violated once col2 <= col1
            candidate = val1 - magnitude
            return math.floor(candidate) if integer_target else candidate
        if relationship == 'greater_than':
            # col1 > col2 is violated once col2 >= col1
            candidate = val1 + magnitude
            return math.ceil(candidate) if integer_target else candidate
        if relationship == 'less_equal':
            # col1 <= col2 is violated once col2 < col1
            candidate = val1 - magnitude - 1
            return math.floor(candidate) if integer_target else candidate
        if relationship == 'greater_equal':
            # col1 >= col2 is violated once col2 > col1
            candidate = val1 + magnitude + 1
            return math.ceil(candidate) if integer_target else candidate
        if relationship == 'equal':
            # col1 == col2 is violated by any other value
            candidate = val1 + random.choice([-1, 1]) * (magnitude + 1)
            return int(round(candidate)) if integer_target else candidate
        if relationship == 'not_equal':
            # col1 != col2 is violated once col2 == col1
            return val1
        raise ValueError(f"unknown relationship '{relationship}'")
