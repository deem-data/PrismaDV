from pathlib import Path
from typing import List, Optional

import oyaml as yaml

from prismadv.data_models.deequ_schema import DeequSchema
from prismadv.utils import get_project_root


class DeequFunctionManager:
    _instance = None
    _initialized = False

    def __new__(cls, info_path: Optional[Path] = None):
        # If custom path provided, don't use singleton pattern
        if info_path is not None:
            instance = super(DeequFunctionManager, cls).__new__(cls)
            return instance
        
        # For default path, use singleton pattern
        if cls._instance is None:
            cls._instance = super(DeequFunctionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, info_path: Optional[Path] = None):
        # If custom path provided, initialize with that path
        if info_path is not None:
            self.info_path = Path(info_path)
            self.info = self.get_info()
            return
        
        # For default path, use singleton pattern (only initialize once)
        if self.__class__._initialized:
            return
        self.info_path = get_project_root() / "prismadv" / "ir_translator" / "deequ_constraints" / "info.yaml"
        self.info = self.get_info()
        self.__class__._initialized = True

    def get_info(self):
        """
        Reads the info.yaml file and returns its content.
        """
        with open(self.info_path, 'r') as file:
            info = yaml.safe_load(file)
        return [DeequSchema.from_dict({k: v}) for k, v in info.items()]

    def get_constraint(self, name: str):
        """
        Returns the constraint schema for the given constraint type.
        """
        for schema in self.info:
            if schema.Name == name:
                return schema
        raise ValueError(f"Constraint type '{name}' not found in info.")

    @property
    def constraint_names(self):
        """
        Returns a list of all constraint names.
        """
        return [schema.Name for schema in self.info]

    def get_relevant_constriants_from_code(self, deequ_code: str) -> List[DeequSchema]:
        relevant_names = [name for name in self.constraint_names if name in deequ_code]
        relevant_constraints_schema = [self.get_constraint(name) for name in relevant_names]
        return relevant_constraints_schema

    def get_constraints(
            self,
            is_row_level: bool = None,
            can_be_used_for_multiple_columns: bool = None
    ):
        constraints = self.info
        if is_row_level is not None:
            constraints = [
                schema for schema in constraints
                if schema.CanUseSatisfies == is_row_level
            ]
        if can_be_used_for_multiple_columns is not None:
            constraints = [
                schema for schema in constraints
                if schema.canBeUsedForMultipleColumns == can_be_used_for_multiple_columns
            ]
        return [schema.to_string() for schema in constraints]


if __name__ == "__main__":
    deequ_function_manager = DeequFunctionManager()
    info = deequ_function_manager.get_info()
    res = deequ_function_manager.get_constraints()
    print("All Constraints:")
    for constraint in res:
        print(constraint)
