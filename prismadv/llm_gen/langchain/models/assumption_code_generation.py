import json

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prismadv.llm_backend.entry import get_langchain_model
from prismadv.llm_gen.langchain.examples import TABLE_PROFILING_EXAMPLE
from prismadv.llm_gen.langchain.prompts.data_generation import (
    TABLE_GENERATION_PROMPT,
    ASSUMPTION_GENERATION_PROMPT,
    CODE_SYNTHESIS_PROMPT,
    SYSTEM_PROMPT
)
from prismadv.utils import get_project_root

model_name = "gpt-5"

model = get_langchain_model(model_name)

parser = JsonOutputParser()
table_generation_prompt = ChatPromptTemplate(
    [
        ("system", SYSTEM_PROMPT),
        ("human", TABLE_GENERATION_PROMPT)],
)
assumption_generation_prompt = ChatPromptTemplate(
    [
        ("system", SYSTEM_PROMPT),
        ("human", ASSUMPTION_GENERATION_PROMPT)],
)
code_generation_prompt = ChatPromptTemplate(
    [
        ("system", SYSTEM_PROMPT),
        ("human", CODE_SYNTHESIS_PROMPT)],
)

table_generation_chain = table_generation_prompt | model | parser
assumption_generation_chain = assumption_generation_prompt | model | parser
code_generation_chain = code_generation_prompt | model | parser

ft_root = get_project_root() / "data_ft"

existing_datasets = [d.name for d in ft_root.iterdir() if d.is_dir()]
already_generated_domains = [existing_dataset.split("_", 1)[1] for existing_dataset in existing_datasets]

num_datasets_to_generate = 1
for i in range(num_datasets_to_generate):
    print(f"Generating dataset {i + 1}/{num_datasets_to_generate}...")
    table_generation_input = {
        "table_profile_example": TABLE_PROFILING_EXAMPLE,
        "already_generated_domains": "\n".join(f"- {d}" for d in already_generated_domains),
    }
    table_response = table_generation_chain.invoke(input=table_generation_input)

    table_save_path = ft_root / f"dataset_{table_response['table'].replace('.csv', '')}"
    table_save_path.mkdir(parents=True, exist_ok=True)

    with open(table_save_path / "table_metadata.json", "w") as f:
        json.dump(table_response, f, indent=2)

    already_generated_domains.append(table_response["domain"])
