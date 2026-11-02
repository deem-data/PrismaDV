"""Generation and evaluation shared by the ICDBench single-prompt (zero-shot) and few-shot LLM baselines.

One LLM call per case, retried up to `MAX_RETRIES` times inside one cost record,
as in the EIDBench single-shot pipeline. A case whose calls all fail gets no
artifact, so a rerun picks it up; what it billed waits in a failed-attempts log
and is folded into the artifact of the run that succeeds.

Scoring uses every constraint the model returned, unfiltered (see `scoring.py`).
"""
import argparse
import os
import signal
import sys
import time
import warnings

import oyaml as yaml
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prismadv.llm.langchain.models.prismadv.costs import run_with_cost
from prismadv.llm_backend.entry import get_langchain_model
from workflow_prismadv.eid_bench_baselines.pipelines.run_codex import (
    append_failed_attempts,
    cost_all_attempts,
    failed_attempts_path,
    read_failed_attempts,
)
from workflow_prismadv.eid_bench_baselines.pipelines.run_single_shot import failure_record, usd_without_cache
from workflow_prismadv.icd_bench_experiments import ALL_EVALUATION_CASES
from workflow_prismadv.icd_bench_experiments.codex import case_id, constraint_codes, select_cases
from workflow_prismadv.utils.replicates import add_replicate_argument, normalize_replicate

MODEL_NAME_OPTIONS = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1", "gpt-5-mini", "gpt-5", "gemini-2.5-flash",
                      "gemini-2.5-pro", "gpt-6-sol-medium", "gemini-3.7-flash", "gemini-3.8-flash"]
TEMPERATURE = 0.6
MAX_RETRIES = 3
PROMPT_SAMPLE_ROWS = 20


def generate_case(prompt_template, evaluation_case, model_name, cost_summary) -> list:
    """The constraints of one case; the LLM usage lands in `cost_summary` even when this raises."""
    chain = ChatPromptTemplate.from_template(prompt_template) \
        | get_langchain_model(model_name=model_name, temperature=TEMPERATURE) | JsonOutputParser()
    input_variables = {
        "data_sample": pd.DataFrame(evaluation_case.sample_data()).head(PROMPT_SAMPLE_ROWS).to_string(index=False),
        "code": evaluation_case.downstream_code(),
        "target_column": evaluation_case.target_column(),
    }

    def invoke_with_retries():
        for attempt in range(MAX_RETRIES):
            try:
                raw = chain.invoke(input_variables)["constraints"]
                if not isinstance(raw, list):
                    raise ValueError(f"'constraints' is a {type(raw).__name__}, not a list")
                return raw
            except Exception as exc:  # noqa: BLE001
                if attempt == MAX_RETRIES - 1:
                    raise
                print(f"Error during LLM invocation: {exc}. Retrying ({attempt + 1}/{MAX_RETRIES})...")

    return run_with_cost("constraint_generation", invoke_with_retries, cost_summary)


def run_generation(method, prompt_template, find_output_path, argv=None) -> int:
    """CLI: generate one artifact per case for `--model_name` and `--replicate`."""
    parser = argparse.ArgumentParser(description=f"ICDBench {method} constraint generation.")
    parser.add_argument("--model_name", choices=MODEL_NAME_OPTIONS, required=True)
    parser.add_argument("--case", action="append", default=None,
                        help="Run only this case (repeatable): module name, class name or <module>.<Class>.")
    parser.add_argument("--overwrite", action="store_true")
    add_replicate_argument(parser)
    args = parser.parse_args(argv)

    failed = []
    for evaluation_case in select_cases(args.case):
        output_path = find_output_path(evaluation_case, args.model_name, args.replicate)
        if output_path.exists() and not args.overwrite:
            print(f"Constraints already exist at {output_path}. Skipping generation.")
            continue
        print(f"Running {method} for ICDBench case {case_id(evaluation_case)}, model {args.model_name}, "
              f"replicate {normalize_replicate(args.replicate)}", flush=True)
        failed_log = failed_attempts_path(output_path)
        started = time.perf_counter()
        cost_summary = {}
        try:
            raw = generate_case(prompt_template, evaluation_case, args.model_name, cost_summary)
        except Exception as exc:  # noqa: BLE001
            record = failure_record(cost_summary.get("constraint_generation", {}), args.model_name, exc,
                                    time.perf_counter() - started)
            append_failed_attempts(failed_log, [record])
            print(f"FAILED {case_id(evaluation_case)}: {exc!r}. Cost ${record['usd'] or 0:.4f} logged in {failed_log}")
            failed.append(case_id(evaluation_case))
            continue
        accepted = cost_summary["constraint_generation"]
        no_cache = usd_without_cache(accepted, args.model_name)
        prior = read_failed_attempts(failed_log)
        result = {
            "method": method,
            "model_name": args.model_name,
            "temperature": TEMPERATURE,
            "replicate": normalize_replicate(args.replicate),
            "case": case_id(evaluation_case),
            "constraints": constraint_codes(raw),
            "raw_constraints": raw,
            "cost_summary": cost_summary,
            "wall_clock_seconds": round(time.perf_counter() - started, 3),
            "cost_without_cache_usd": no_cache,
            "failed_attempts": prior,
            "cost_all_attempts": cost_all_attempts(accepted["cost"], no_cache, prior),
        }
        with open(output_path, "w") as f:
            yaml.dump(result, f, default_flow_style=False, sort_keys=False)
        failed_log.unlink(missing_ok=True)
        print(f"... constraints saved to {output_path}", flush=True)
    if failed:
        print(f"{len(failed)} case(s) failed: {failed}")
        return 1
    return 0


def run_evaluation(method, find_output_path, argv=None) -> None:
    """CLI: score every case's artifact for `--model_name` and `--replicate`, then exit."""
    from prismadv.dq_manager import DeequDataQualityManager
    from workflow_prismadv.icd_bench_experiments.scoring import report, score_case

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser = argparse.ArgumentParser(description=f"Evaluate the ICDBench {method} baseline.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--out", default=None, help="Write the per-case results to this TSV.")
    args = parser.parse_args(argv)

    dq_manager = DeequDataQualityManager()
    rows, missing = [], []
    spark = None
    for evaluation_case in ALL_EVALUATION_CASES:
        path = find_output_path(evaluation_case, args.model_name, args.replicate)
        if not path.exists():
            missing.append(case_id(evaluation_case))
            continue
        with open(path, "r") as f:
            artifact = yaml.safe_load(f)
        constraints = constraint_codes(artifact.get("constraints"))
        scores, spark = score_case(dq_manager, evaluation_case, constraints)
        # Artifacts written before cost tracking carry no cost summary.
        accepted = (artifact.get("cost_summary") or {}).get("constraint_generation") or {}
        failed = artifact.get("failed_attempts") or []
        costs = artifact.get("cost_all_attempts") or {}
        rows.append({
            "case": case_id(evaluation_case),
            **scores,
            "usd": costs.get("usd"),
            "usd_without_cache": costs.get("usd_without_cache"),
            "attempts": 1 + len(failed),
            "llm_calls": (accepted.get("requests") or 0) + sum(f.get("llm_calls") or 0 for f in failed),
            "llm_calls_exact": bool(accepted),
            "wall_clock_seconds": round((artifact.get("wall_clock_seconds") or 0)
                                        + sum(f.get("wall_clock_seconds") or 0 for f in failed), 3),
        })
        print(f"{case_id(evaluation_case)}: {len(constraints)} constraints, exact={scores['exact_match']}, "
              f"rejects={scores['rejects_reject_batch']}, passes={scores['passes_pass_batch']}")

    report(f"ICDBench {method}, model {args.model_name}, replicate {args.replicate}", rows, missing, args.out)
    if spark is not None:
        spark.stop()
    # The signal below ends the process without flushing, so piped output would be lost.
    sys.stdout.flush()
    # py4j keeps the process alive otherwise.
    os.kill(os.getpid(), signal.SIGTERM)
