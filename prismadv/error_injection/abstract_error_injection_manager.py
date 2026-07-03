from abc import ABC, abstractmethod

import oyaml as yaml

from prismadv.error_injection.abstract_corruption import DataCorruption
from prismadv.error_injection.corrupts import (
    # Existing
    MissingCategoricalValueCorruption,
    GaussianNoise,
    Scaling,
    ColumnInserting,
    MaskValues,
    ColumnDropping,
    DuplicatedRows,
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
    EmailCorruption,
    PhoneNumberCorruption,
    RegexPatternViolation,
    # Temporal & Sequential
    TemporalGaps,
    TemporalOutOfOrder,
    SeasonalityAnomaly,
    # Statistical & Distribution
    DistributionShift,
    ImbalancedCategories,
    # Referential & Relational
    ForeignKeyViolation,
    AggregationInconsistency,
    FunctionalDependencyViolation,
)


class AbstractErrorInjectionManager(ABC):

    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    @abstractmethod
    def error_injection(self, corrupts: list[DataCorruption]):
        raise NotImplementedError

    def _load_existing_corrupts_yaml(self, processed_data_dir):
        existing_corrupts_yaml_list = []
        for processed_data_root in processed_data_dir.iterdir():
            config_path = processed_data_root / "error_injection_config.yaml"
            if config_path.exists():
                existing_corrupts = self.load_error_injection_config(config_path)
                existing_corrupts_yaml = self.corrupts_to_yaml(existing_corrupts)
                existing_corrupts_yaml_list.append(existing_corrupts_yaml)
        return existing_corrupts_yaml_list

    def check_corrupts_existence(self, corrupts, processed_data_dir):
        new_corrupts_yaml = self.corrupts_to_yaml(corrupts)
        existing_corrupts_yaml_list = self._load_existing_corrupts_yaml(processed_data_dir)
        for existing_corrupts_yaml in existing_corrupts_yaml_list:
            if new_corrupts_yaml == existing_corrupts_yaml:
                return True
        return False

    def corrupts_to_yaml(self, corrupts):
        return yaml.dump([corrupt.to_dict() for corrupt in corrupts], default_flow_style=False)

    def _create_processed_data_path(self, processed_data_dir):
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        processed_data_label = len(list(processed_data_dir.iterdir()))
        processed_data_path = processed_data_dir / f"{processed_data_label}"
        processed_data_path.mkdir(parents=True, exist_ok=True)
        return processed_data_path

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
            "DuplicatedRows": DuplicatedRows,
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
            "EmailCorruption": EmailCorruption,
            "PhoneNumberCorruption": PhoneNumberCorruption,
            "RegexPatternViolation": RegexPatternViolation,
            # Temporal & Sequential
            "TemporalGaps": TemporalGaps,
            "TemporalOutOfOrder": TemporalOutOfOrder,
            "SeasonalityAnomaly": SeasonalityAnomaly,
            # Statistical & Distribution
            "DistributionShift": DistributionShift,
            "ImbalancedCategories": ImbalancedCategories,
            # Referential & Relational
            "ForeignKeyViolation": ForeignKeyViolation,
            "AggregationInconsistency": AggregationInconsistency,
            "FunctionalDependencyViolation": FunctionalDependencyViolation,
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
