from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass
class DeequParameter:
    description: str = ""
    type: str = None


@dataclass
class DeequSchema:
    Name: str
    Description: str
    RequiredArgs: Dict[str, DeequParameter]
    OptionalArgs: Dict[str, DeequParameter] = None
    CanUseSatisfies: bool = None
    canBeUsedForMultipleColumns: bool = None
    Examples: Union[str, List[str], None] = None

    @staticmethod
    def from_dict(data: Dict):
        assert len(data) == 1, "Data dictionary must contain exactly one key representing the function name."
        data_value = list(data.values())[0]
        required_args = {i: DeequParameter() for i in data_value['required']}
        optional_args = {i: DeequParameter() for i in data_value['optional']} if data_value['optional'] else {}
        can_use_satisfies = data_value.get('canUseSatisfies', None)
        can_be_used_for_multiple_columns = data_value.get('canBeUsedForMultipleColumns', None)
        
        # Read examples - can be either 'example' (single string) or 'examples' (list of strings)
        examples = None
        if 'examples' in data_value:
            examples = data_value['examples']
        elif 'example' in data_value:
            examples = data_value['example']

        return DeequSchema(
            Name=list(data.keys())[0],
            Description=data_value.get("description", "").strip(),
            RequiredArgs=required_args,
            OptionalArgs=optional_args,
            CanUseSatisfies=can_use_satisfies,
            canBeUsedForMultipleColumns=can_be_used_for_multiple_columns,
            Examples=examples
        )

    def to_string(self):
        """
        Converts the schema to a string representation.
        """
        result = f"Name: {self.Name}, Description: {self.Description}\n RequiredArgs: {list(self.RequiredArgs.keys())}" \
                 f"\n OptionalArgs: {list(self.OptionalArgs.keys())}"
        
        # Add examples if they exist
        if self.Examples:
            if isinstance(self.Examples, list):
                examples_str = "\n".join([f"  - {ex}" for ex in self.Examples])
                result += f"\n Examples:\n{examples_str}"
            else:
                result += f"\n Example: {self.Examples}"
        
        return result
