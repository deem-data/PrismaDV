"""Stage 2 of the SIFTA pipeline: a
program writes the constraints of every task of a dataset as EIDBench-synth artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import dspy
import oyaml as yaml

from prismadv.llm.dspy.models.column_wise_module import (
    ConstraintGenerationModule,
    HumanDesignedConstraintGenerationModule,
)
from prismadv.llm_backend.model_routes import litellm_route
from prismadv.utils import get_project_root, load_dotenv
from workflow_prismadv.results.benchmarks import synth
from workflow_prismadv.utils.replicates import parse_replicate, replicate_part
from workflow_sifta import hard_exit
from workflow_sifta.baselines.gepa.dataset_preparation import prepare_script_level_dataset
from workflow_sifta.baselines.gepa.script_level_module import ScriptLevelConstraintGenerationModule
from workflow_sifta.optimization_fns import OptimizationWorkflow
from workflow_sifta._0_splits import SUBTASK, load_split, scored_scripts
from workflow_sifta.runtime.usage import (TASK_LM, replicate_rollout_id, reset_usage, set_rollout_id, tracked_lm,
                                          usage_summary)

load_dotenv()

LLM_TEMPERATURE = 0.6
LLM_MAX_TOKENS = 32000
RUN_ARMS = {"run_": "sifta", "gepa_column_run_": "gepa_column", "baseline_gepa_run_": "gepa_script"}
MAIN_SETTINGS = {
    "sifta": {"batch_sampler_strategy": "worst_first", "feedback_level": "full", "feedback_constraints": 5,
              "silent_as_safe": 0.0, "num_batches": None},
    "gepa_column": {"metric_type": "f1_score"},
    "gepa_script": {},
}
SUFFIXES = {
    ("batch_sampler_strategy", "random"): "random",
    ("batch_sampler_strategy", "best_first"): "best_first",
    ("feedback_level", "score"): "fb_score",
    ("feedback_level", "stats"): "fb_stats",
    ("feedback_level", "assumptions"): "fb_assumptions",
    ("feedback_constraints", 1): "fc1",
    ("feedback_constraints", 3): "fc3",
    ("metric_type", "fail_precision"): "fp",
    ("silent_as_safe", 0.03): "silent3",
    ("silent_as_safe", 0.05): "silent5",
    ("num_batches", 6): "batches6",
    ("num_batches", 12): "batches12",
}
OPTIONAL_SETTINGS = ("feedback_constraints", "silent_as_safe", "num_batches")
BASE_PROGRAMS = ("base_column", "base_script")
PROGRAMS = ["base_column", "sifta", "gepa_column", "base_script", "gepa_script"]     # main rows, table order
ABLATION_PROGRAMS = ["sifta_random", "sifta_best_first", "sifta_fb_score", "sifta_fb_stats", "sifta_fb_assumptions",
                     "sifta_fc1", "sifta_fc3", "gepa_column_fp", "sifta_silent3",
                     "sifta_silent5", "sifta_batches6", "sifta_batches12"]
SCRIPT_LEVEL = ("gepa_script", "base_script")
# Programs optimized with SIFTA's fail precision. Their artifacts record the run's metric_rev, and stage 4 scores only
# those of the current version (workflow_sifta.metrics.optimization_metrics.METRIC_REV).
FAIL_PRECISION_PROGRAMS = ("sifta", "sifta_random", "sifta_best_first", "sifta_fb_score", "sifta_fb_stats",
                           "sifta_fb_assumptions", "sifta_fc1", "sifta_fc3",
                           "gepa_column_fp",
                           "sifta_silent3", "sifta_silent5",
                           "sifta_batches6", "sifta_batches12")
COLUMN_MODULES = {"base": ConstraintGenerationModule, "human": HumanDesignedConstraintGenerationModule}


def apply_instructions_to_module(module: dspy.Module, instructions: dict) -> None:
    """Set each predictor's instructions. The signature is copied first: DSPy signatures are class attributes, so
    changing one in place would change every instance of the module."""
    for name, predictor in module.named_predictors():
        if name in instructions:
            predictor.signature = predictor.signature.with_instructions(instructions[name])


def arm_of_run(run_dir: Path) -> str:
    for prefix, arm in RUN_ARMS.items():
        if run_dir.name.startswith(prefix):
            return arm
    raise ValueError(f"{run_dir.name}: not a SIFTA, GEPA-column or GEPA run directory")


def program_of_run(arm: str, summary: dict) -> tuple[str, dict]:
    """The program name of a run (the arm, plus a suffix for each setting that differs from the main arm) and the
    settings it was named from."""
    # A setting newer than the run (silent_as_safe) has the main value there.
    settings = {key: summary.get(key, MAIN_SETTINGS[arm][key] if key in OPTIONAL_SETTINGS else None)
                for key in MAIN_SETTINGS[arm]}
    suffixes = []
    for key, value in settings.items():
        if value == MAIN_SETTINGS[arm][key]:
            continue
        if (key, value) not in SUFFIXES:
            raise ValueError(f"{arm} run with {key}={value!r}: no program name for it; add it to SUFFIXES")
        suffixes.append(SUFFIXES[(key, value)])
    program = "_".join([arm, *suffixes])
    if program not in PROGRAMS + ABLATION_PROGRAMS:
        raise ValueError(f"program {program!r} is not in PROGRAMS or ABLATION_PROGRAMS; add it there")
    return program, settings


def load_program(program: str, arm: str | None, run_dir: Path | None, settings: dict) -> tuple[dspy.Module, dict]:
    """The module, and the provenance the artifact records."""
    if run_dir is None:
        module = ScriptLevelConstraintGenerationModule() if program == "base_script" else ConstraintGenerationModule()
        return module, {"program": program, "run_dir": None}
    summary = json.loads((run_dir / "summary.json").read_text())
    final = json.loads((run_dir / "final_program.json").read_text())
    if arm == "gepa_script":
        module = ScriptLevelConstraintGenerationModule()
    else:
        module = COLUMN_MODULES[summary.get("initial_program_type") or "base"]()
    apply_instructions_to_module(module, final["instructions"])
    return module, {"program": program, "run_dir": str(run_dir.relative_to(get_project_root())),
                    "run_settings": settings, "metric_rev": summary.get("metric_rev"),
                    "final_program_round": final.get("round"),
                    "final_program_val_score": final.get("score"), "instructions": final["instructions"]}


def existing_artifacts(task_dir: Path, program: str, model: str, replicate: int) -> list[Path]:
    return [p for p in task_dir.glob(f"{program}--{model}--*.yaml") if parse_replicate(p) == replicate]


def generate_column_level(workflow: OptimizationWorkflow, module, dataset: str, task: str, num_threads: int):
    generated = workflow.generate_test_constraints(module, [task], num_threads)
    if not generated:
        return None, {}
    entry = generated[0]
    failed = [c for c in entry["columns_to_consider"] if c not in entry["single_column_results"]]
    return entry["constraints_with_sources"], {"columns_considered": list(entry["columns_to_consider"]),
                                               "columns_failed": failed}


def generate_script_level(workflow: OptimizationWorkflow, module, dataset: str, task: str):
    examples = prepare_script_level_dataset(dataset_subtasks={dataset: [SUBTASK]}, script_name_list=[task],
                                            processed_data_label="0", project_root=get_project_root())
    if not examples:
        return None, {}
    ex = examples[0]
    constraints = module(code_script=ex["code_script"], columns_desc=ex["columns_desc"],
                         downstream_task_description=ex["downstream_task_description"])
    constraints = workflow.validate_constraints_on_training_data(dataset_name=dataset, subtask_name=SUBTASK,
                                                                 constraints_with_sources=constraints)
    return constraints, {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, help="optimization run whose final program generates")
    ap.add_argument("--program", choices=BASE_PROGRAMS, help="an unoptimized starting program instead of a run")
    ap.add_argument("--dataset", choices=synth.DATASETS, help="with --program (a run states its own)")
    ap.add_argument("--replicate", type=int, help="with --program (a run uses its split seed)")
    ap.add_argument("--llm", default="gemini-3.8-flash", help="with --program (a run states its own)")
    ap.add_argument("--task", action="append", help="only these tasks (repeatable); default: every task")
    ap.add_argument("--num_threads", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="write a new artifact even where one exists")
    a = ap.parse_args(argv)

    root = get_project_root()
    if (a.run_dir is None) == (a.program is None):
        ap.error("give either --run_dir or --program")
    if a.run_dir is not None:
        run_dir = a.run_dir if a.run_dir.is_absolute() else root / a.run_dir
        summary = json.loads((run_dir / "summary.json").read_text())
        if summary.get("split_seed") is None:
            ap.error(f"{run_dir} has no split_seed: it predates the split files")
        arm = arm_of_run(run_dir)
        if summary.get("num_batches") is not None and summary["train_label_list"] != summary["eval_label_list"]:
            ap.error(f"{run_dir.name} drew disjoint train and eval batches, an earlier design; "
                     "its artifacts would carry the name of the shared-batch runs")
        program, settings = program_of_run(arm, summary)
        dataset, replicate, llm = summary["train_dataset_name"], summary["split_seed"], summary["llm_name"]
        split_seed = replicate
        rollout_id = replicate_rollout_id(replicate, summary.get("replicate_seed", 0))   # the run's own task LM
    else:
        if a.dataset is None or a.replicate is None:
            ap.error("--program needs --dataset and --replicate")
        run_dir, program, dataset, replicate, llm, split_seed = None, a.program, a.dataset, a.replicate, a.llm, None
        arm, settings = None, {}
        rollout_id = replicate

    tasks = synth.tasks(dataset, root)
    if split_seed is not None:
        scored = scored_scripts(load_split(dataset, split_seed, root))
        tasks = [t for t in tasks if t in scored]
    if a.task:
        unknown = sorted(set(a.task) - set(tasks))
        if unknown:
            ap.error(f"unknown tasks for {dataset}: {unknown}")
        tasks = [t for t in tasks if t in a.task]
    constraints_dir = synth.general(dataset, root) / "constraints"
    todo = [t for t in tasks if a.force or not existing_artifacts(constraints_dir / t, program, llm, replicate)]
    print(f"{program} {llm} r{replicate} on {dataset}: {len(todo)} of {len(tasks)} tasks to generate")
    if not todo:
        return 0

    set_rollout_id(rollout_id)  # before the LM is made: replicate N reads and writes its own cached answers
    dspy.configure(lm=tracked_lm(litellm_route(llm), temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS,
                                 usage_role=TASK_LM))
    module, provenance = load_program(program, arm, run_dir, settings)
    workflow = OptimizationWorkflow(train_dataset_subtasks={dataset: [SUBTASK]}, train_processed_data_label_list=[],
                                    project_root=root)
    failed = []
    for task in todo:
        reset_usage()
        started = time.time()
        try:
            if program in SCRIPT_LEVEL or arm == "gepa_script":
                constraints, extra = generate_script_level(workflow, module, dataset, task)
            else:
                constraints, extra = generate_column_level(workflow, module, dataset, task, a.num_threads)
        except Exception as e:  # noqa: BLE001 -- reported below, and the run exits with status 1
            print(f"[{task}] generation failed: {e!r}")
            constraints = None
        if constraints is None:
            failed.append(task)
            continue
        wall_clock = round(time.time() - started, 3)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rep = replicate_part(replicate)
        path = constraints_dir / task / f"{program}--{llm}{'--' + rep if rep else ''}--{stamp}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "sifta_generation": {**provenance,
                                 "llm": {"model_name": llm, "temperature": LLM_TEMPERATURE,
                                         "max_tokens": LLM_MAX_TOKENS}},
            "replicate": replicate,
            "split_seed": split_seed,
            **extra,
            **constraints.to_dict(),
            "llm_usage": usage_summary(),
            "wall_clock_seconds": wall_clock,
        }
        path.write_text(yaml.dump(artifact, sort_keys=False, allow_unicode=True))
        usage = artifact["llm_usage"]["total"]
        print(f"[{task}] {path.relative_to(root)}: ${usage['cost']:.3f} (billed ${usage['cost_spent']:.3f}), "
              f"{wall_clock:.0f} s")
    workflow.cleanup_spark_sessions()
    if failed:
        print(f"generation failed for {len(failed)} task(s): {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    hard_exit(main())
