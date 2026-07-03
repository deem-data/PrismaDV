from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict


@dataclass(frozen=True)
class LLMConfig:
    model_name: str
    temperature: float = 0.6
    max_tokens: Optional[int] = None
    seed: Optional[int] = None


@dataclass(frozen=True)
class IOConfig:
    overwrite: bool = False


@dataclass(frozen=True)
class ModelConfig:
    use_async: bool = True
    use_dataflow: bool = True
    correlation_detection: bool = False
    with_assumptions: bool = True
    downstream_task_description: Optional[str] = None


@dataclass(frozen=True)
class PrismaDVConfig:
    model: ModelConfig
    llm: LLMConfig
    io: IOConfig

    @staticmethod
    def from_dict(config_dict: Dict) -> 'PrismaDVConfig':
        model_config = ModelConfig(**config_dict['model'])
        llm_config = LLMConfig(**config_dict['llm'])
        io_config = IOConfig(**config_dict.get('io', {}))
        return PrismaDVConfig(
            model=model_config,
            llm=llm_config,
            io=io_config,
        )

    def to_dict(self) -> Dict:
        return {
            'model': self.model.__dict__,
            'llm': self.llm.__dict__,
            'io': self.io.__dict__,
        }

    def make_output_filename(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"prismadv--{self.llm.model_name}--{timestamp}.yaml"

    def __eq__(self, other) -> bool:
        # io is not considered for equality check
        if not isinstance(other, PrismaDVConfig):
            return False
        return self.model == other.model and self.llm == other.llm
