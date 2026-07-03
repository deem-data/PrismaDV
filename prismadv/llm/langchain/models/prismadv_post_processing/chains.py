from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prismadv.llm.langchain.prompts.prismadv_post_processing.code_consolidation.general_code_consolidation import \
    CODE_CONSOLIDATION_PROMPT
from prismadv.llm.langchain.prompts.prismadv_post_processing.code_fixing.general_code_fixing import \
    CODE_FIXING_PROMPT
from prismadv.llm.tasks import PrismaDVTasks

PROMPT_MAP = {
    PrismaDVTasks.CODE_FIXING: CODE_FIXING_PROMPT,
    PrismaDVTasks.CODE_CONSOLIDATION: CODE_CONSOLIDATION_PROMPT,
}

PARSER_MAP = {
    PrismaDVTasks.CODE_FIXING: JsonOutputParser,
    PrismaDVTasks.CODE_CONSOLIDATION: JsonOutputParser,
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
