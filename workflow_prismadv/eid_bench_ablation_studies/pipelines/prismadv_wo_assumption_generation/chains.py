from langchain_core.output_parsers import CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prismadv.llm.langchain.prompts.prismadv.constraint_scope_detection.column_access_detection import \
    COLUMN_ACCESS_DETECTION_PROMPT
from prismadv.llm.langchain.prompts.prismadv.constraint_scope_detection.column_correlation_discovery import \
    COLUMN_CORRELATION_DISCOVERY_PROMPT
from workflow_prismadv.eid_bench_ablation_studies.pipelines.prismadv_wo_assumption_generation.prompt.direct_code_generation import \
    DIRECT_CODE_GENERATION_PROMPT
from workflow_prismadv.eid_bench_ablation_studies.pipelines.prismadv_wo_assumption_generation.tasks import PrismaDVTasks

PROMPT_MAP = {
    PrismaDVTasks.COLUMN_ACCESS_DETECTION: COLUMN_ACCESS_DETECTION_PROMPT,
    PrismaDVTasks.COLUMN_CORRELATION_DISCOVERY: COLUMN_CORRELATION_DISCOVERY_PROMPT,
    PrismaDVTasks.SINGLE_DIRECT_CODE_GENERATION: DIRECT_CODE_GENERATION_PROMPT,
    PrismaDVTasks.MULTI_DIRECT_CODE_GENERATION: DIRECT_CODE_GENERATION_PROMPT,
}

PARSER_MAP = {
    PrismaDVTasks.COLUMN_ACCESS_DETECTION: CommaSeparatedListOutputParser,
    PrismaDVTasks.COLUMN_CORRELATION_DISCOVERY: JsonOutputParser,
    PrismaDVTasks.SINGLE_DIRECT_CODE_GENERATION: JsonOutputParser,
    PrismaDVTasks.MULTI_DIRECT_CODE_GENERATION: JsonOutputParser,
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
