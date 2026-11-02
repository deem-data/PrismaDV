"""Entry point of the SIFTA experiments: runs a stage here, or with --submit as a Slurm job on its own node.

  optimize <arm>   stage 1: sifta | gepa | gepa-column on one dataset and split, with a budget of validation passes
  generate         stage 2: a program writes the constraints of every task (workflow_sifta._2_generation)
  validate         stage 3: validates programs' constraints on every label (workflow_sifta._3_validation);
                   with --submit, the labels are split over --jobs jobs, each with every program
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from prismadv.utils import get_project_root
from workflow_prismadv.results.benchmarks import synth

SCRATCH = Path(os.environ.get("SIFTA_SCRATCH", Path.home() / "scratch/prismadv-runs"))
CACHE_STORE = SCRATCH / "dspy-cache"
LOG_DIR = SCRATCH / "sifta-logs"
NODE = ["--partition=compute", "--account=sc-users", "--cpus-per-task=8", "--mem=48G", "--time=14:00:00"]
MERGE_NODE = ["--partition=compute", "--account=sc-users", "--cpus-per-task=1", "--mem=8G", "--time=1:00:00"]
ENV = {"SPARK_VERSION": "3.5"}
LLM_STAGES = ("optimize", "generate")

LLM = "gemini-3.8-flash"
VALIDATION_PASSES = 15
SIFTA_ROUNDS = 3
PROGRAMS = ["base_column", "sifta", "gepa_column", "base_script", "gepa_script",   # as workflow_sifta._2_generation
            "sifta_random", "sifta_best_first", "sifta_fb_score", "sifta_fb_stats", "sifta_fb_assumptions",
            "sifta_fc1", "sifta_fc3", "gepa_column_fp", "sifta_silent3",
            "sifta_silent5", "sifta_batches6", "sifta_batches12"]
# Main-arm settings of `optimize`; another value makes an ablation, which stage 2 names with a suffix.
SIFTA_STRATEGY, SIFTA_FEEDBACK, GEPA_COLUMN_METRIC = "worst_first", "full", "f1"
SIFTA_FEEDBACK_CONSTRAINTS = 5
ABLATION_SUFFIX = {"random": "random", "best_first": "best_first", "score": "fb_score", "stats": "fb_stats",
                   "assumptions": "fb_assumptions", "fail_precision": "fp"}


def optimize_command(a) -> list[str]:
    common = ["--split_seed", str(a.split), "--validation_passes", str(a.passes)]
    if a.arm == "sifta":
        return ["workflow_sifta._1_optimization", "--dataset", a.dataset, *common, "--strategy", a.strategy,
                "--feedback_level", a.feedback_level, "--feedback_constraints", str(a.feedback_constraints),
                "--silent_as_safe", str(a.silent_as_safe), "--replicate_seed", str(a.replicate_seed),
                "--initial_program", "base", "--max_rounds", str(SIFTA_ROUNDS),
                "--llm", a.llm, "--reflection_llm", a.llm,
                *(["--num_batches", str(a.num_batches)] if a.num_batches is not None else [])]
    module = {"gepa": "workflow_sifta.baselines.gepa.optimization",
              "gepa-column": "workflow_sifta.baselines.gepa_column.optimization"}[a.arm]
    extra = ["--metric", a.metric] if a.arm == "gepa-column" else []
    return [module, "--dataset_name", a.dataset, *common, "--max_rounds", "1", *extra,
            "--llm_name", a.llm, "--reflection_llm_name", a.llm]


def generate_command(a) -> list[str]:
    out = ["workflow_sifta._2_generation"]
    if a.run_dir:
        out += ["--run_dir", str(a.run_dir)]
    else:
        out += ["--program", a.program, "--dataset", a.dataset, "--replicate", str(a.replicate), "--llm", a.llm]
    for t in a.task or []:
        out += ["--task", t]
    return out + (["--force"] if a.force else [])


def validate_command(a) -> list[str]:
    out = ["workflow_sifta._3_validation", "--dataset", a.dataset, "--llm", a.llm]
    for p in a.program:
        out += ["--program", p]
    for n in a.label or []:
        out += ["--label", n]
    for t in a.task or []:
        out += ["--task", t]
    return out


def job_name(a) -> str:
    if a.stage == "optimize":
        changed = [ABLATION_SUFFIX[v] for v, main in [(a.strategy, SIFTA_STRATEGY), (a.feedback_level, SIFTA_FEEDBACK),
                                                       (a.metric, GEPA_COLUMN_METRIC)] if v != main]
        if a.feedback_constraints != SIFTA_FEEDBACK_CONSTRAINTS:
            changed.append(f"fc{a.feedback_constraints}")
        if a.silent_as_safe:
            changed.append(f"silent{round(a.silent_as_safe * 100)}")
        if a.replicate_seed:
            changed.append(f"seed{a.replicate_seed}")
        if a.num_batches is not None:
            changed.append(f"batches{a.num_batches}")
        return f"sifta-opt-{'_'.join([a.arm, *changed])}-{a.dataset}-s{a.split}"
    if a.stage == "validate":
        return f"sifta-val-{a.dataset}"
    what = Path(a.run_dir).name if a.run_dir else f"{a.program}-{a.dataset}-r{a.replicate}"
    return f"sifta-gen-{what}"


def submit_merge() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = get_project_root()
    wrap = f"cd {shlex.quote(str(root))} && uv run python -m workflow_sifta.runtime.cache_store merge --store {CACHE_STORE}"
    subprocess.run(["sbatch", "--parsable", "-J", "dspy-cache-merge", "--dependency=singleton", *MERGE_NODE,
                    f"--output={LOG_DIR}/%x_%j.out", "--wrap", wrap], check=True)


def run_here(stage: str, module_args: list[str]) -> int:
    """Run the stage in this process's environment. An LLM stage works on the cache store and, inside a Slurm job,
    submits a merge when it ends."""
    env = {**os.environ, **ENV}
    if stage in LLM_STAGES:
        env["DSPY_CACHE_STORE"] = str(CACHE_STORE)
    rc = subprocess.run([sys.executable, "-u", "-m", *module_args], env=env, cwd=get_project_root()).returncode
    if stage in LLM_STAGES and os.environ.get("SLURM_JOB_ID"):
        submit_merge()
    return rc


def submit(name: str, argv: list[str]) -> int:
    """Submit this command, without --submit, as a Slurm job."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = get_project_root()
    inner = " ".join(shlex.quote(x) for x in argv if x != "--submit")
    wrap = f"cd {shlex.quote(str(root))} && uv run python -m workflow_sifta.run {inner}"
    out = subprocess.run(["sbatch", "--parsable", "-J", name, *NODE, f"--output={LOG_DIR}/%x_%j.out",
                          "--wrap", wrap], check=True, capture_output=True, text=True)
    print(f"{name}: job {out.stdout.strip()}, log {LOG_DIR}/{name}_{out.stdout.strip()}.out")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    opt = sub.add_parser("optimize")
    opt.add_argument("arm", choices=["sifta", "gepa", "gepa-column"])
    opt.add_argument("--dataset", required=True)
    opt.add_argument("--split", type=int, required=True, help="split seed = replicate")
    opt.add_argument("--passes", type=int, default=VALIDATION_PASSES, help="budget in validation passes")
    opt.add_argument("--llm", default=LLM)
    opt.add_argument("--strategy", default=SIFTA_STRATEGY, choices=["worst_first", "random", "best_first"],
                     help="sifta only; another value than the default is an ablation")
    opt.add_argument("--feedback_level", default=SIFTA_FEEDBACK, choices=["full", "score", "stats", "assumptions"],
                     help="sifta only; another value than the default is an ablation")
    opt.add_argument("--feedback_constraints", type=int, default=SIFTA_FEEDBACK_CONSTRAINTS, choices=[1, 3, 5],
                     help="sifta only, with the main settings: how many constraints each feedback shows in detail; "
                          "another value than the default is a hyperparameter run (sifta_fc1 / sifta_fc3)")
    opt.add_argument("--metric", default=GEPA_COLUMN_METRIC, choices=["f1", "fail_precision"],
                     help="gepa-column only; another value than the default is an ablation")
    opt.add_argument("--silent_as_safe", type=float, default=0.0, choices=[0.0, 0.03, 0.05],
                     help="sifta only, with the main settings: this fraction of the silent train and eval units is "
                          "labelled safe for the optimization (ablation sifta_silent3 / sifta_silent5)")
    opt.add_argument("--replicate_seed", type=int, default=0, choices=[0, 1],
                     help="sifta only: 1 reruns the split with fresh LLM samples and minibatch draws (same program "
                          "name as seed 0); 0 is the original run")
    opt.add_argument("--num_batches", type=int, default=None, choices=[6, 12],
                     help="sifta only, with the main settings: the optimization observes this many train+eval data "
                          "batches, shared by train and eval as in the main arm (sifta_batches<N>); default: every "
                          "train+eval label")

    gen = sub.add_parser("generate")
    gen.add_argument("--run_dir", type=Path)
    gen.add_argument("--program", choices=["base_column", "base_script"])
    gen.add_argument("--dataset")
    gen.add_argument("--replicate", type=int)
    gen.add_argument("--llm", default=LLM)
    gen.add_argument("--task", action="append")
    gen.add_argument("--force", action="store_true")

    val = sub.add_parser("validate")
    val.add_argument("--dataset", required=True)
    val.add_argument("--program", action="append", required=True, choices=PROGRAMS, help="repeatable")
    val.add_argument("--llm", default=LLM)
    val.add_argument("--label", action="append")
    val.add_argument("--task", action="append")
    val.add_argument("--jobs", type=int, default=8, help="with --submit: the labels are split over this many jobs")

    sub.add_parser("merge-cache")

    for p in (opt, gen, val):
        p.add_argument("--submit", action="store_true", help="run as a Slurm job on its own node")
    a = ap.parse_args(argv)

    if a.stage == "merge-cache":
        from workflow_sifta.runtime.cache_store import merge
        merge(CACHE_STORE)
        return 0
    if a.stage == "generate" and (a.run_dir is None) == (a.program is None):
        gen.error("give either --run_dir or --program")
    if a.stage == "optimize":
        if a.arm != "sifta" and (a.strategy, a.feedback_level) != (SIFTA_STRATEGY, SIFTA_FEEDBACK):
            opt.error("--strategy and --feedback_level are SIFTA settings")
        if a.arm != "gepa-column" and a.metric != GEPA_COLUMN_METRIC:
            opt.error("--metric is a gepa-column setting")
        if a.silent_as_safe and (a.arm != "sifta" or (a.strategy, a.feedback_level) != (SIFTA_STRATEGY, SIFTA_FEEDBACK)):
            opt.error("--silent_as_safe is planned for sifta with the main settings only")
        main_settings = (a.strategy, a.feedback_level, a.silent_as_safe) == (SIFTA_STRATEGY, SIFTA_FEEDBACK, 0.0)
        if a.feedback_constraints != SIFTA_FEEDBACK_CONSTRAINTS and (a.arm != "sifta" or not main_settings):
            opt.error("--feedback_constraints is planned for sifta with the main settings only")
        if a.replicate_seed and a.arm != "sifta":
            opt.error("--replicate_seed is a sifta setting")
        if a.num_batches is not None and (a.arm != "sifta" or not main_settings
                                          or a.feedback_constraints != SIFTA_FEEDBACK_CONSTRAINTS):
            opt.error("--num_batches is planned for sifta with the main settings only")
    if a.stage == "validate":
        if a.submit:
            # The validator starts Spark once per label for all tasks and programs, so a job takes every program
            # and a share of the labels.
            labels = a.label or synth.labels(a.dataset)
            k = min(a.jobs, len(labels))
            for i in range(k):
                a1 = argparse.Namespace(**{**vars(a), "label": labels[i::k]})
                submit(f"{job_name(a1)}-{i + 1}of{k}", ["validate", *validate_command(a1)[1:]])
            return 0
        return run_here(a.stage, validate_command(a))
    if a.submit:
        return submit(job_name(a), argv)
    return run_here(a.stage, optimize_command(a) if a.stage == "optimize" else generate_command(a))


if __name__ == "__main__":
    raise SystemExit(main())
