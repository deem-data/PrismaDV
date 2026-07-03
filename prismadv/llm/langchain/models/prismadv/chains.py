from langchain_core.output_parsers import CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prismadv.llm.langchain.prompts.prismadv.assumptions_generation.multi_column_assumption import \
    MULTI_COLUMN_ASSUMPTION_GENERATION_PROMPT
from prismadv.llm.langchain.prompts.prismadv.assumptions_generation.single_column_assumption import \
    SINGLE_COLUMN_ASSUMPTION_GENERATION_PROMPT
from prismadv.llm.langchain.prompts.prismadv.constraint_scope_detection.column_access_detection import \
    COLUMN_ACCESS_DETECTION_PROMPT
from prismadv.llm.langchain.prompts.prismadv.constraint_scope_detection.column_correlation_discovery import \
    COLUMN_CORRELATION_DISCOVERY_PROMPT
from prismadv.llm.langchain.prompts.prismadv.ir_generation.multi_column_constraint import MULTI_COLUMN_CONSTRAINT_PROMPT
from prismadv.llm.langchain.prompts.prismadv.ir_generation.sql_as_ir import IR_GENERATION_PROMPT
from prismadv.llm.tasks import PrismaDVTasks

PROMPT_MAP = {
    PrismaDVTasks.COLUMN_ACCESS_DETECTION: COLUMN_ACCESS_DETECTION_PROMPT,
    PrismaDVTasks.COLUMN_CORRELATION_DISCOVERY: COLUMN_CORRELATION_DISCOVERY_PROMPT,
    PrismaDVTasks.SINGLE_COLUMN_ASSUMPTION_GENERATION: SINGLE_COLUMN_ASSUMPTION_GENERATION_PROMPT,
    PrismaDVTasks.MULTI_COLUMN_ASSUMPTION_GENERATION: MULTI_COLUMN_ASSUMPTION_GENERATION_PROMPT,
    PrismaDVTasks.CODE_GENERATION: IR_GENERATION_PROMPT,
    PrismaDVTasks.MULTI_COLUMN_CODE_GENERATION: MULTI_COLUMN_CONSTRAINT_PROMPT,
}

PARSER_MAP = {
    PrismaDVTasks.COLUMN_ACCESS_DETECTION: CommaSeparatedListOutputParser,
    PrismaDVTasks.COLUMN_CORRELATION_DISCOVERY: JsonOutputParser,
    PrismaDVTasks.SINGLE_COLUMN_ASSUMPTION_GENERATION: JsonOutputParser,
    PrismaDVTasks.MULTI_COLUMN_ASSUMPTION_GENERATION: JsonOutputParser,
    PrismaDVTasks.CODE_GENERATION: JsonOutputParser,
    PrismaDVTasks.MULTI_COLUMN_CODE_GENERATION: JsonOutputParser,
}


def create_chain(task: PrismaDVTasks, model, downstream_task_description: str):
    prompt = ChatPromptTemplate(
        [("human", PROMPT_MAP[task])],
        partial_variables={"downstream_task_description": downstream_task_description},
    )
    parser = PARSER_MAP[task]()
    return prompt | model | parser


def build_chains(model, downstream_task_description: str):
    return {task: create_chain(task, model, downstream_task_description) for task in PROMPT_MAP}
