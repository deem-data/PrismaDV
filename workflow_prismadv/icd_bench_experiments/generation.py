"""Generate constraints with PrismaDV on ICDBench cases.

The ICDBench PrismaDV configuration: the target column is given, there is no
column-correlation step and no task description, and the calls are synchronous.
The artifact records this configuration, the replicate, the cost summary (with
the LLM requests of every stage) and the wall-clock time of the case.

    python -m workflow_prismadv.icd_bench_experiments.generation --model_name gpt-6-sol-medium
"""
import argparse
import os
import sys
import time
import traceback

import oyaml as yaml
import pandas as pd

from prismadv.data_models.code_container import CodeContainer
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from prismadv.llm.langchain.models.prismadv import PrismaLangChainDV
from workflow_prismadv.icd_bench_experiments import find_constraints_output_path
from workflow_prismadv.icd_bench_experiments.codex import case_id, select_cases
from workflow_prismadv.utils.replicates import add_replicate_argument, normalize_replicate

MODEL_NAME_OPTIONS = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1", "gpt-5-mini", "gpt-5", "gemini-2.5-flash",
                      "gemini-2.5-pro", "gpt-6-sol-medium", "gemini-3.7-flash", "gemini-3.8-flash"]
TEMPERATURE = 0.6
TASK_DESCRIPTION = ""


def prismadv_config(model_name, target_column) -> dict:
    """What identifies the run; written into the artifact."""
    return {
        "model": {
            "use_async": False,
            "use_dataflow": True,
            "correlation_detection": False,
            "downstream_task_description": TASK_DESCRIPTION,
            "columns_to_consider": [target_column],
        },
        "llm": {"model_name": model_name, "temperature": TEMPERATURE},
    }


def run_case(evaluation_case, model_name, replicate, dq_manager) -> dict:
    started = time.perf_counter()
    data_sample = pd.DataFrame(evaluation_case.sample_data())
    data_sample_df, spark = dq_manager.spark_df_from_pandas_df(data_sample)

    column_desc_dict = DeequInspectorManager().spark_df_to_column_desc_dict(spark, data_sample_df)
    column_desc = yaml.dump(column_desc_dict, default_flow_style=False, sort_keys=False)
    code = CodeContainer(evaluation_case.downstream_code())

    input_variables = {
        "code_script": code,
        "downstream_task_description": TASK_DESCRIPTION,
        "column_desc_dict": column_desc_dict,
        "columns_desc": column_desc,
        "code_snippet": code,
        "cfg_use_dataflow": True,
        "columns_to_consider": [evaluation_case.target_column()],
        "spark": spark,
        "data_sample": data_sample_df,
    }
    prismadv = PrismaLangChainDV(model_name=model_name, downstream_task_description=TASK_DESCRIPTION,
                                 temperature=TEMPERATURE)
    constraints_with_sources, column_data_flow_locations, cost_summary = prismadv.invoke(input_variables)
    return {
        "prismadv_config": prismadv_config(model_name, evaluation_case.target_column()),
        "replicate": normalize_replicate(replicate),
        "case": case_id(evaluation_case),
        "constraints": constraints_with_sources.to_dict()["constraints"],
        "column_data_flow_locations": {
            col: loc.to_dict() for col, loc in column_data_flow_locations.items()
        },
        "cost_summary": cost_summary,
        # Profiling of the sample plus every PrismaDV stage, the span of the
        # other rows' wall_clock_seconds.
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", choices=MODEL_NAME_OPTIONS, required=True)
    parser.add_argument("--case", action="append", default=None,
                        help="Run only this case (repeatable): module name, class name or <module>.<Class>.")
    parser.add_argument("--overwrite", action="store_true")
    add_replicate_argument(parser)
    args = parser.parse_args(argv)

    dq_manager = DeequDataQualityManager()
    failed = []
    for evaluation_case in select_cases(args.case):
        constraints_output_path = find_constraints_output_path(evaluation_case, args.model_name, args.replicate)
        if constraints_output_path.exists() and not args.overwrite:
            print(f"Constraints already exist at {constraints_output_path}. Skipping generation.")
            continue
        print(f"Running PrismaDV for ICDBench case {case_id(evaluation_case)} (target column "
              f"{evaluation_case.target_column()!r}), model {args.model_name}, "
              f"replicate {normalize_replicate(args.replicate)}", flush=True)
        try:
            res_dict = run_case(evaluation_case, args.model_name, args.replicate, dq_manager)
        except Exception:  # noqa: BLE001
            # No artifact, so a rerun picks the case up again.
            traceback.print_exc()
            failed.append(case_id(evaluation_case))
            continue
        with open(constraints_output_path, "w") as f:
            yaml.dump(res_dict, f, default_flow_style=False, sort_keys=False)
        print(f"... constraints saved to {constraints_output_path}", flush=True)
    if failed:
        print(f"{len(failed)} case(s) failed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # py4j's threads keep the interpreter alive otherwise.
    os._exit(code)
