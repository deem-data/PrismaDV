from abc import ABC, abstractmethod

import oyaml as yaml
import pandas as pd

from prismadv.error_injection.abstract_corruption import DataCorruption
from prismadv.error_injection.corrupts import (
    # Existing
    MissingCategoricalValueCorruption,
    GaussianNoise,
    Scaling,
    ColumnInserting,
    MaskValues,
    ColumnDropping,
    ColumnRename,
    DuplicatedRows,
    RowDropping,
    StringNoise,
    StringTruncation,
    # Data Quality
    OutlierInjection,
    DataTypeViolation,
    RangeViolation,
    UniqueConstraintViolation,
    CrossColumnInconsistency,
    ValueReplacement,
    # Format & Pattern
    DateFormatCorruption,
    FormatChange,
    RegexPatternViolation,
    # Statistical & Distribution
    DistributionShift,
    ImbalancedCategories,
    # Referential & Relational
    ForeignKeyViolation,
    FunctionalDependencyViolation,
    FieldSwap,
)


class AbstractErrorInjectionManager(ABC):

    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    @abstractmethod
    def error_injection(self, corrupts: list[DataCorruption]):
        raise NotImplementedError

    CONFIG_FILE_NAME = "error_injection_config.yaml"

    def check_corrupts_existence(self, corrupts, processed_data_path):
        """Whether `processed_data_path` already holds exactly this corruption.

        The comparison ignores `random_state`: the saved config carries the
        seed the run was given, the benchmark config usually does not.
        """
        config_path = processed_data_path / self.CONFIG_FILE_NAME
        if not config_path.exists():
            return False
        existing_corrupts = self.load_error_injection_config(config_path)
        return self._comparable_config(existing_corrupts) == self._comparable_config(corrupts)

    @staticmethod
    def _comparable_config(corrupts):
        entries = []
        for corrupt in corrupts:
            entry = corrupt.to_dict()
            for attributes in entry.values():
                attributes["Params"] = {
                    k: v for k, v in attributes["Params"].items() if k != "random_state"
                }
            entries.append(entry)
        return yaml.dump(entries, default_flow_style=False)

    def corrupts_to_yaml(self, corrupts):
        return yaml.dump([corrupt.to_dict() for corrupt in corrupts], default_flow_style=False)

    @staticmethod
    def seed_corrupts(corrupts, base_seed):
        """Give every unseeded corruption a deterministic seed.

        `base_seed` is the processed-data label, so the same config injected
        at two labels draws different rows while each label stays
        reproducible. The seed lands on the instance and therefore in the
        saved config.
        """
        for position, corrupt in enumerate(corrupts):
            if getattr(corrupt, "random_state", None) is None:
                corrupt.seed(int(base_seed) * 100 + position)

    @staticmethod
    def apply_corrupts(clean_data, corrupts):
        """Run the corruptions in order and check that they did something.

        Each corruption re-seeds the global RNGs right before it runs, so the
        result does not depend on what ran earlier in the process. Integer
        columns that a corruption upcast to float (a NaN or a float written
        into an int64 column) are restored to a nullable integer dtype so the
        untouched rows are written back unchanged; two labels used to differ
        from their clean data in every row of the column for that reason
        alone. A result identical to the input is an error: the config
        described a corruption that never happened, and the label would be
        recorded as "safe" for a batch that carries no error.
        """
        corrupted = clean_data.copy(deep=True)
        for corrupt in corrupts:
            corrupt.seed(corrupt.random_state)
            corrupted = corrupt.transform(corrupted)
        corrupted = AbstractErrorInjectionManager._restore_integer_dtypes(clean_data, corrupted)
        if AbstractErrorInjectionManager._same_content(clean_data, corrupted):
            names = ", ".join(type(corrupt).__name__ for corrupt in corrupts)
            raise RuntimeError(f"corruption(s) {names} left the data unchanged")
        return corrupted

    @staticmethod
    def _restore_integer_dtypes(original, corrupted):
        for column in original.columns:
            if column not in corrupted.columns:
                continue
            if not pd.api.types.is_integer_dtype(original[column]):
                continue
            if not pd.api.types.is_float_dtype(corrupted[column]):
                continue
            values = corrupted[column].dropna()
            if len(values) and not (values == values.round()).all():
                continue
            corrupted[column] = corrupted[column].astype("Int64")
        return corrupted

    @staticmethod
    def _same_content(left, right):
        if left.shape != right.shape or list(left.columns) != list(right.columns):
            return False
        return bool((left.astype(str).values == right.astype(str).values).all())

    def _create_processed_data_path(self, processed_data_dir, processed_data_label):
        # The label is the config's own index (`general_task_7.yaml` -> `7`),
        # not a directory count: counting also saw non-label entries such as
        # `constraints/` and depended on the order configs were injected in.
        processed_data_path = processed_data_dir / str(processed_data_label)
        processed_data_path.mkdir(parents=True, exist_ok=True)
        return processed_data_path

    def save_error_injection_config(self, processed_data_path, corrupts):
        config_path = processed_data_path / self.CONFIG_FILE_NAME
        with open(config_path, "w") as f:
            f.write(self.corrupts_to_yaml(corrupts))

    @property
    def corruption_classes(self):
        return {
            # Existing
            "MissingCategoricalValueCorruption": MissingCategoricalValueCorruption,
            "GaussianNoise": GaussianNoise,
            "Scaling": Scaling,
            "ColumnInserting": ColumnInserting,
            "MaskValues": MaskValues,
            "ColumnDropping": ColumnDropping,
            "ColumnRename": ColumnRename,
            "DuplicatedRows": DuplicatedRows,
            "RowDropping": RowDropping,
            "StringNoise": StringNoise,
            "StringTruncation": StringTruncation,
            # Data Quality
            "OutlierInjection": OutlierInjection,
            "DataTypeViolation": DataTypeViolation,
            "RangeViolation": RangeViolation,
            "UniqueConstraintViolation": UniqueConstraintViolation,
            "CrossColumnInconsistency": CrossColumnInconsistency,
            "ValueReplacement": ValueReplacement,
            # Format & Pattern
            "DateFormatCorruption": DateFormatCorruption,
            "FormatChange": FormatChange,
            "RegexPatternViolation": RegexPatternViolation,
            # Statistical & Distribution
            "DistributionShift": DistributionShift,
            "ImbalancedCategories": ImbalancedCategories,
            # Referential & Relational
            "ForeignKeyViolation": ForeignKeyViolation,
            "FunctionalDependencyViolation": FunctionalDependencyViolation,
            "FieldSwap": FieldSwap,
        }

    def load_error_injection_config(self, error_injection_config_path):
        with open(error_injection_config_path, "r") as f:
            config_data = yaml.safe_load(f)

        corrupts = []
        for entry in config_data:
            for class_name, attributes in entry.items():
                if class_name in self.corruption_classes:
                    corrupt_class = self.corruption_classes[class_name]
                    corrupt_instance = corrupt_class(columns=attributes["Columns"], **attributes["Params"])
                    corrupts.append(corrupt_instance)
                else:
                    raise ValueError(f"Unknown corruption class: {class_name}")

        return corrupts
