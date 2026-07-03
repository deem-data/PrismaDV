import asyncio
import glob
import json
import os

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prismadv.data_models.code_container import CodeContainer
from prismadv.llm_backend.entry import get_langchain_model
from prismadv.llm_gen.langchain.prompts.data_verification import (
    SYSTEM_PROMPT,
    LOOSE_OR_TIGHT_DETECTION_PROMPT,
    BAD_ASSERTIONS_REMOVING_PROMPT
)
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.runtime_environments import PythonExecutor
from prismadv.utils import get_project_root
from workflow.e2e_generation.utils import process_bad_assertions

loose_or_tight_detection_prompt = ChatPromptTemplate(
    [
        ("system", SYSTEM_PROMPT),
        ("human", LOOSE_OR_TIGHT_DETECTION_PROMPT)
    ],
)

bad_assertions_removing_prompt = ChatPromptTemplate(
    [
        ("system", SYSTEM_PROMPT),
        ("human", BAD_ASSERTIONS_REMOVING_PROMPT)
    ],
)


def _process_one(dataset_record_path: str):
    with open(dataset_record_path, "r") as f:
        data = json.load(f)

    code_with_assertions = CodeContainer(data['code_metadata']['result']['code'])
    code_wo_assertions, assertions = code_with_assertions.extract_assertions()

    updated_assertions = assertions
    num_removed_history, num_modified_history, code_editing_history = [], [], []

    while True:
        code_with_indexed_assertions = code_wo_assertions.insert_assertions(updated_assertions, with_index=True)

        # IMPORTANT: if invoke() is not thread-safe, create a fresh chain instance here.
        bad_assertions = loose_or_tight_detection_chain.invoke(
            {"code_with_indexed_assertions": str(code_with_indexed_assertions)}
        )

        updated_assertions, num_removed, num_modified = process_bad_assertions(bad_assertions, updated_assertions)
        num_removed_history.append(num_removed)
        num_modified_history.append(num_modified)

        edited_code = code_wo_assertions.insert_assertions(updated_assertions, with_index=False)
        code_editing_history.append(str(edited_code))

        if ((num_removed == 0 and num_modified == 0) or
                (len(num_removed_history) >= 3 and num_removed_history[-3:] == [0, 0, 0])):
            break

    assertion_editing_path = dataset_record_path.replace(".json", "_edited.json")
    editing_res = {
        "code_editing_history": code_editing_history,
        "num_removed_history": num_removed_history,
        "num_modified_history": num_modified_history,
    }
    with open(assertion_editing_path, "w", encoding="utf-8") as f:
        json.dump(editing_res, f, ensure_ascii=False, indent=2)

    return assertion_editing_path


# 2) Async wrapper that runs the worker in a thread
async def process_one_async(path: str, sem: asyncio.Semaphore):
    async with sem:
        return await asyncio.to_thread(_process_one, path)


# 3) Driver
async def main_async(paths, max_concurrency: int = 16):
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [asyncio.create_task(process_one_async(p, sem)) for p in paths]
    results = []
    errors = []
    for t in asyncio.as_completed(tasks):
        try:
            results.append(await t)
        except Exception as e:
            errors.append(repr(e))
    return results, errors


model_name = "gemini-2.5-pro"
dataset_name_list = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]

model = get_langchain_model(model_name)
parser = JsonOutputParser()
executor = PythonExecutor()
for dataset_name in dataset_name_list:
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)

    loose_or_tight_detection_chain = loose_or_tight_detection_prompt | model | parser
    bad_assertions_removing_chain = bad_assertions_removing_prompt | model | parser

    gen_root = get_project_root() / "data_gen"
    raw_file_path = project_manager.files_root
    target_table_name = project_manager.raw_data_info["target_table_name"]
    input_data = raw_file_path / f"{target_table_name}.csv"
    print(input_data)
    table_save_path = gen_root / f"dataset_{input_data.name.replace('.csv', '')}"
    if not table_save_path.exists():
        raise Exception("Require to generate the assumptions and codes first.")
    executable_record_path_list = []
    edited_files = {
        p.replace("_edited.json", ".json")
        for p in glob.glob(os.path.join(table_save_path, "record_*_edited.json"))
    }
    for dataset_record_path in glob.glob(os.path.join(table_save_path, "record_*.json")):
        if dataset_record_path.endswith("_edited.json"):
            continue  # skip already edited files
        if dataset_record_path in edited_files:
            continue  # skip if an edited version exists
        print(dataset_record_path)
        with open(dataset_record_path, "r") as f:
            data = json.load(f)
        code_with_assertions = CodeContainer(data['code_metadata']['result']['code'])
        code_wo_assertions, assertions = code_with_assertions.extract_assertions()
        response = executor.run_script(
            project_name=dataset_name,
            input_path=project_manager.get_new_data_path("general_task", "0", True),
            script_context=code_wo_assertions,
            output_path=None,
        )
        if response.startswith("Error"):
            continue  # skip script that cannot executable
        executable_record_path_list.append(dataset_record_path)

    print(executable_record_path_list)
    results, errors = asyncio.run(main_async(executable_record_path_list, max_concurrency=8))
    print(f"wrote {len(results)} files")
    if errors: print("errors:", *errors, sep="\n- ")
