import time
from pathlib import Path

import oyaml as yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prismadv.data_models import Constraints
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from prismadv.llm.langchain.models.prismadv.costs import run_with_cost
from prismadv.llm_backend.entry import get_langchain_model
from prismadv.llm_backend.llm_pricing_info import price_tokens
from prismadv.loader import FileLoader
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow_prismadv.eid_bench_baselines.pipelines.run_codex import (
    append_failed_attempts,
    cost_all_attempts,
    failed_attempts_path,
    normalize_raw_constraints,
    read_failed_attempts,
)
from workflow_prismadv.utils.spark_cleanup import close_spark_session
from workflow_prismadv.utils.replicates import (
    artifact_filename,
    artifact_stem,
    existing_artifacts,
    normalize_replicate,
    select_scripts,
)

METHOD = "single_shot"

SINGLE_PROMPT = """
Write a pydeequ data unit test for a dataset consumed by the following program.

Here is sample of data that can be processed by the program:

{data_sample}

Here is the data profiling results:
{columns_desc}

Here is the program:

{code}

Focus on the python code for the Check object from pydeequ only.

Here is an example output:

{{
    "constraints": [
        {{
            'column_name': 'some_column',
            'code_for_constraint': ".isComplete('some_column')",
        }},
        {{
            'column_name': 'another_column',
            'code_for_constraint': ".isPositive('another_column')",
        }}
    ]
}}

The output must be a valid JSON object with a single key "constraints" that maps to a list of constraints. No other text should be included outside the JSON object.
"""


def run_single_shot(dataset_name, subtask_name, processed_data_label, model_name, replicate=None,
                    script_stems=None):
    return run_prompt_baseline(METHOD, SINGLE_PROMPT, dataset_name, subtask_name, processed_data_label,
                               model_name, replicate, script_stems)


def run_prompt_baseline(method, prompt_template, dataset_name, subtask_name, processed_data_label, model_name,
                        replicate=None, script_stems=None) -> list[str]:
    """One LLM call per script from `prompt_template` (the single-shot and few-shot rows).

    Returns the stems of scripts that failed. A failure's bill waits in a log beside
    the artifacts and is added to the artifact of the run that later succeeds.
    """
    max_retries = 3
    replicate = normalize_replicate(replicate)
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)

    dq_manager = DeequDataQualityManager()
    train_data = FileLoader.load_csv(
        project_manager.get_observed_data_path(subtask_name, processed_data_label)
    )
    spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data)
    example_rows = train_data.sample(3).to_dict(orient="records")

    script_path_list = project_manager.get_available_script_path_list_for_subtask(subtask_name=subtask_name)
    failed_scripts = []
    for script_path in select_scripts(script_path_list, script_stems):
        constraints_path = project_manager.get_source_constraints_path(subtask_name, script_path.stem)
        constraints_path.mkdir(parents=True, exist_ok=True)
        constraint_output_path = constraints_path / artifact_filename(method, model_name, replicate)
        existing = existing_artifacts(constraints_path, method, model_name, replicate)
        if existing:
            print(f"Constraints file {existing[-1]} already exists (replicate {replicate}). Skipping...")
            continue
        failure_log = failure_log_path(constraints_path, model_name, replicate, method)
        started = time.perf_counter()
        cost_summary = {}
        try:
            constraints, normalization = _generate_for_script(
                method, prompt_template, script_path, model_name, max_retries, dq_manager, spark_train, spark_train_data,
                example_rows, list(train_data.columns), cost_summary)
        except Exception as exc:
            record = failure_record(cost_summary.get("constraint_generation", {}), model_name, exc,
                                    time.perf_counter() - started)
            append_failed_attempts(failure_log, [record])
            print(f"FAILED {dataset_name}/{script_path.stem}: {exc!r}. "
                  f"Cost ${record['usd'] or 0:.4f} logged in {failure_log}")
            failed_scripts.append(script_path.stem)
            continue
        accepted = cost_summary["constraint_generation"]
        no_cache = usd_without_cache(accepted, model_name)
        prior = read_failed_attempts(failure_log)
        constraints.cost_summary = cost_summary
        constraints.save_to_yaml(constraint_output_path, extra={
            "replicate": replicate,
            "wall_clock_seconds": round(time.perf_counter() - started, 3),
            "cost_without_cache_usd": no_cache,
            "constraint_normalization": normalization,
            "failed_attempts": prior,
            "cost_all_attempts": cost_all_attempts(accepted["cost"], no_cache, prior),
        })
        failure_log.unlink(missing_ok=True)
    close_spark_session(spark_train)
    return failed_scripts


def _generate_for_script(method, prompt_template, script_path, model_name, max_retries, dq_manager, spark_train, spark_train_data,
                         example_rows, known_columns, cost_summary):
    """One script's constraints; the LLM usage lands in `cost_summary` even when this raises."""
    source_code, assertions = FileLoader.load_py_file(script_path).extract_assertions()
    assert "# ASSERTION_START" not in str(source_code)

    column_desc_dict = DeequInspectorManager().spark_df_to_column_desc_dict(spark_train, spark_train_data)
    column_desc = yaml.dump(column_desc_dict, default_flow_style=False, sort_keys=False)
    input_variables = {
        "code": source_code,
        "columns_desc": column_desc,
        "data_sample": example_rows,
    }
    print(f"Running {method} for script: {script_path.stem}, model: {model_name}")

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = get_langchain_model(model_name, temperature=0.6)
    parser = JsonOutputParser()

    chain = prompt | llm | parser

    def _invoke_with_retries():
        for attempt in range(max_retries):
            try:
                raw = chain.invoke(input_variables)['constraints']
                if not isinstance(raw, list):
                    raise ValueError(f"'constraints' is a {type(raw).__name__}, not a list")
                return raw
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error during LLM invocation: {e}. Retrying ({attempt + 1}/{max_retries})...")
                    continue
                else:
                    raise e

    raw_constraints = run_with_cost("constraint_generation", _invoke_with_retries, cost_summary)
    # The same formats the codex runner accepts: plain strings, `suggestion`, a missing column.
    raw_constraints, normalization = normalize_raw_constraints(raw_constraints, known_columns)
    code_list_for_constraints = [item["code_for_constraint"] for item in raw_constraints]
    code_list_for_constraints_valid = dq_manager.filter_valid_constraints_on_spark(
        code_list_for_constraints, spark_train, spark_train_data)
    constraints = Constraints.from_deequ_output(raw_constraints, code_list_for_constraints_valid)
    return constraints, normalization


def usd_without_cache(cost_entry: dict, model_name: str) -> float | None:
    """The call's tokens with every prompt token at the fresh-input rate."""
    usd = price_tokens(model_name, cost_entry.get("prompt_tokens") or 0,
                       cost_entry.get("completion_tokens") or 0, 0)
    return None if usd is None else round(usd, 6)


def failure_record(cost_entry: dict, model_name: str, exc: BaseException, seconds: float) -> dict:
    """What a failed script billed, in the shape the codex runner logs failed attempts."""
    return {
        "outcome": f"{type(exc).__name__}: {exc}"[:500],
        "wall_clock_seconds": round(seconds, 3),
        "llm_calls": cost_entry.get("requests", 0),
        "input_tokens": cost_entry.get("prompt_tokens", 0),
        "cached_input_tokens": cost_entry.get("cached_prompt_tokens", 0),
        "output_tokens": cost_entry.get("completion_tokens", 0),
        "reasoning_tokens": cost_entry.get("reasoning_tokens", 0),
        "usd": cost_entry.get("cost"),
        "usd_without_cache": usd_without_cache(cost_entry, model_name) if cost_entry else None,
    }


def failure_log_path(constraints_path, model_name, replicate, method=METHOD) -> Path:
    """One log per script, method, model and replicate; the artifact name carries a timestamp, so it cannot key the log."""
    return failed_attempts_path(constraints_path / f"{artifact_stem(method, model_name, replicate, timestamp=False)}.yaml")


def make_output_filename(model_name, replicate=None) -> str:
    return artifact_filename(METHOD, model_name, replicate)
