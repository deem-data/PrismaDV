import asyncio
import json
import warnings
from uuid import uuid4

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prismadv.dq_manager import DeequDataQualityManager
from prismadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from prismadv.llm_backend.entry import get_langchain_model
from prismadv.llm_gen.langchain.prompts.data_generation import (
    ASSUMPTION_GENERATION_PROMPT,
    CODE_SYNTHESIS_PROMPT,
    SYSTEM_PROMPT
)
from prismadv.llm_gen.langchain.prompts.table_summarization import TABLE_SUMMARIZATION_PROMPT
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyspark")

model_name = "gemini-2.5-pro"
num_records_per_dataset = 30
dataset_name = "sleep_health"
model = get_langchain_model(model_name)

parser = JsonOutputParser()

table_generation_prompt = ChatPromptTemplate(
    ("human", TABLE_SUMMARIZATION_PROMPT)
)

assumption_generation_prompt = ChatPromptTemplate(
    [
        ("system", SYSTEM_PROMPT),
        ("human", ASSUMPTION_GENERATION_PROMPT)],
)
code_synthesis_prompt = ChatPromptTemplate(
    [
        ("system", SYSTEM_PROMPT),
        ("human", CODE_SYNTHESIS_PROMPT)],
)

table_summarization_chain = table_generation_prompt | model | parser
assumption_generation_chain = assumption_generation_prompt | model | parser
code_synthesis_chain = code_synthesis_prompt | model | parser

gen_root = get_project_root() / "data_gen"

project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
raw_file_path = project_manager.files_root
target_table_name = project_manager.raw_data_info["target_table_name"]
input_data_list = [raw_file_path / f"{target_table_name}.csv"]
for i, input_data in enumerate(input_data_list):
    table_save_path = gen_root / f"dataset_{input_data.name.replace('.csv', '')}"
    if not table_save_path.exists():
        table_save_path.mkdir(parents=True, exist_ok=True)
        try:
            pd_data = pd.read_csv(input_data)
            deequ_inspector_manager = DeequInspectorManager()
            data_quality_manager = DeequDataQualityManager()
            spark_df, spark = data_quality_manager.spark_df_from_pandas_df(pd_data)
            column_desc = deequ_inspector_manager.spark_df_to_column_desc(spark_df, spark)

            table_summarization_inputs = {
                "filename": input_data.name,
                "column_profiles": column_desc,
                "example_rows": pd_data.sample(3).to_dict(orient="records"),
            }
            response = table_summarization_chain.invoke(input=table_summarization_inputs)
            table_response = {
                "table": input_data.name,
                "domain": response["domain"],
                "description": response["description"],
                "profile": column_desc,
                "example_rows": pd_data.sample(3).to_dict(orient="records"),
            }
            with open(table_save_path / "table_metadata.json", "w") as f:
                json.dump(table_response, f, indent=2)
        except Exception as e:
            print(f"Error processing dataset {i + 1}: {e}")
            continue
    else:
        table_response = json.load(open(table_save_path / "table_metadata.json"))

    import re


    def infer_start_idx(dir_path, max_n: int) -> int:
        """Return the smallest idx in [0, max_n) that has no generated file."""
        existing = set()
        for p in dir_path.glob("record_*.json"):
            m = re.match(r"record_(\d+)_", p.name)
            if m:
                existing.add(int(m.group(1)))
        for i in range(max_n):
            if i not in existing:
                return i
        return max_n


    async def generate_record(record_no: int):
        try:
            print(f"  Generating record {record_no + 1}/{num_records_per_dataset}...")
            assumption_generation_input = {
                "table_name": table_response["table"].replace('.csv', ''),
                "table_profile": table_response["profile"],
                "example_rows": pd.DataFrame(table_response['example_rows']).to_string(),
                "task_description": table_response["description"]
            }
            assumption_response = await assumption_generation_chain.ainvoke(input=assumption_generation_input)

            assumption_strs = []
            for group in assumption_response["assumptions_on_column_groups"]:
                group_lines = [f"Column group: {group['target_column_group']}"]
                for idx, a in enumerate(group["assumptions"], 1):
                    group_lines.append(f"  {idx}. {a['assumption']} (source: {a['source']})")
                assumption_strs.append("\n".join(group_lines))
            assumptions_text = "\n".join(assumption_strs)

            code_synthesis_input = {
                "table_name": table_response["table"].replace('.csv', ''),
                "table_profile": table_response["profile"],
                "task_description": assumption_response["task_description"],
                "example_rows": pd.DataFrame(table_response['example_rows']).to_string(),
                "assumptions": assumptions_text,
            }

            code_response = await code_synthesis_chain.ainvoke(input=code_synthesis_input)

            final_assumptions = assumption_response["assumptions_on_column_groups"]
            dataset_record = {
                "table": table_response["table"],
                "domain": table_response["domain"],
                "profile": table_response["profile"],
                "example_rows": table_response["example_rows"],
                "task_description": assumption_response["task_description"],
                "assumptions": final_assumptions,
                "code": code_response["result"]["code"],
            }
            generation_output = {
                "table_metadata": table_response,
                "assumptions_metadata": assumption_response,
                "code_metadata": code_response,
                "dataset_record": dataset_record,
            }

            # write with unique filename to avoid collisions under concurrency
            dataset_record_path = table_save_path / f"record_{record_no}_{uuid4().hex[:8]}.json"
            with open(dataset_record_path, "w") as f:
                json.dump(generation_output, f, indent=2)
        except Exception as e:
            print(f"Error generating record for dataset {i + 1}: {e}")


    async def _run_all_records(start_idx: int = 0):
        start = max(0, min(start_idx, num_records_per_dataset))
        await asyncio.gather(*(generate_record(i) for i in range(start, num_records_per_dataset + start)))


    start_idx = infer_start_idx(table_save_path, num_records_per_dataset)
    print(f"Starting from idx {start_idx} out of {num_records_per_dataset}")
    asyncio.run(_run_all_records(start_idx))
