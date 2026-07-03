"""AutoTest (Chen et al., Auto-Test, SIGMOD 2025) as a task-agnostic baseline."""
import os
import json
import uuid
import tempfile
import subprocess
from pathlib import Path

import oyaml as yaml
import pandas as pd

from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root

AUTOTEST_REPO = os.environ.get("AUTOTEST_REPO")
AUTOTEST_ENV = os.environ.get("AUTOTEST_ENV")
AUTOTEST_CONDA_SH = os.environ.get("AUTOTEST_CONDA_SH")
AUTOTEST_HF_HOME = os.environ.get("AUTOTEST_HF_HOME")
AUTOTEST_STAGING = Path(os.environ.get("AUTOTEST_STAGING", Path(tempfile.gettempdir()) / "autotest_runs"))
NUM_WORKERS = int(os.environ.get("AUTOTEST_NUM_WORKERS", "1"))
NUM_GPUS = int(os.environ.get("AUTOTEST_NUM_GPUS", "1"))

OUT_COLS = ["header", "outlier", "conf", "dist_val", "SDC"]


def _sdc_path(sdc_name):
    return Path(AUTOTEST_REPO) / "results" / "SDC" / f"{sdc_name}_selected_sdc.csv"


def _require_autotest():
    if not AUTOTEST_REPO or not AUTOTEST_ENV:
        raise RuntimeError(
            "AutoTest is an external dependency and is not configured.\n"
            "Set AUTOTEST_REPO and AUTOTEST_ENV, or implement "
            "autotest_detect_csvs() against your own AutoTest API using the documented "
            "CSV-in / CSV-out contract."
        )


def _run_batch_detect(manifest_path, sdc_path, out_dir, gpu):
    """Invoke AutoTest's batch_detect.py in its own conda env."""
    activate = (f"source {AUTOTEST_CONDA_SH} && conda activate {AUTOTEST_ENV}"
                if AUTOTEST_CONDA_SH else f"conda activate {AUTOTEST_ENV}")
    hf = f"HF_HOME={AUTOTEST_HF_HOME} " if AUTOTEST_HF_HOME else ""
    cmd = (f"{activate} && export {hf}TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES={gpu} && "
           f"python batch_detect.py {manifest_path} {sdc_path} {out_dir}")
    return subprocess.Popen(["bash", "-lc", cmd], cwd=str(AUTOTEST_REPO),
                            stdout=open(out_dir.parent / f"worker_gpu{gpu}_{uuid.uuid4().hex[:6]}.log", "w"),
                            stderr=subprocess.STDOUT)


def autotest_detect_csvs(id_to_csv, sdc_name="rt_train", run_name=None,
                         num_workers=None, num_gpus=None, timeout=None):
    """Detect outliers in a set of CSV tables (each column = a table column)."""
    _require_autotest()
    num_workers = num_workers or NUM_WORKERS
    num_gpus = max(1, num_gpus or NUM_GPUS)

    run_dir = AUTOTEST_STAGING / (run_name if run_name else f"run_{uuid.uuid4().hex[:10]}")
    out_dir = run_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    sdc_path = _sdc_path(sdc_name)

    manifest = [{"id": i, "csv": str(p)} for i, p in id_to_csv.items()]
    n = min(num_workers, max(1, len(manifest)))
    procs = []
    for w in range(n):
        chunk = manifest[w::n]
        if not chunk:
            continue
        mpath = run_dir / f"manifest_{w}.json"
        mpath.write_text(json.dumps(chunk))
        procs.append(_run_batch_detect(mpath, sdc_path, out_dir, gpu=w % num_gpus))
    print(f"AutoTest: {len(manifest)} tables across {len(procs)} worker(s) on "
          f"{num_gpus} GPU(s); staging={run_dir}", flush=True)
    for p in procs:
        p.wait(timeout=timeout)

    results = {}
    for i in id_to_csv:
        out_csv = out_dir / f"{i}.csv"
        results[i] = pd.read_csv(out_csv, sep="\t") if out_csv.exists() else pd.DataFrame(columns=OUT_COLS)
    return results


def autotest_detect_tables(id_to_df, sdc_name="rt_train", run_name=None,
                           num_workers=None, num_gpus=None, timeout=None):
    """Like autotest_detect_csvs but for in-memory DataFrames (writes temp CSVs)."""
    run_dir = AUTOTEST_STAGING / (run_name if run_name else f"tbl_{uuid.uuid4().hex[:10]}")
    in_dir = run_dir / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    id_to_csv = {}
    for i, df in id_to_df.items():
        p = in_dir / f"{i}.csv"
        if not p.exists():
            df.to_csv(p, index=False)
        id_to_csv[i] = p
    return autotest_detect_csvs(id_to_csv, sdc_name=sdc_name, run_name=run_name,
                                num_workers=num_workers, num_gpus=num_gpus, timeout=timeout)


def run_autotest(dataset_name, subtask_name, processed_data_label, sdc_name="rt_train"):
    """EIDBench 'inference' step for AutoTest."""
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    constraint_output_path = project_manager.get_task_agnostic_constraint_path(
        subtask_name, processed_data_label) / "autotest_constraints.yaml"
    constraint_output_path.parent.mkdir(parents=True, exist_ok=True)
    marker = {
        "method": "autotest",
        "reference": "Chen et al., Auto-Test, SIGMOD 2025 (arXiv:2504.10762)",
        "sdc_set": sdc_name,
        "note": ("Task- and dataset-agnostic: Semantic-Domain Constraints learned offline "
                 "from the RT-train corpus, applied directly to detect outlier cells at "
                 "validation time. AutoTest is run as an external API."),
    }
    with open(constraint_output_path, "w") as f:
        yaml.dump(marker, f, sort_keys=False)
    print(f"Wrote AutoTest constraint marker: {constraint_output_path}")
