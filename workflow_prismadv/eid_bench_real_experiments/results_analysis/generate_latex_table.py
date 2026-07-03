"""Build a standalone LaTeX PDF table for EIDBench-real detection performance.

Mirrors the format used for EIDBench-synth Table 3:
  Method | Exec. | Non-exec. | Data to Pass (Passed↑, False alarm↓) |
         | Data to Reject (Rejected↑, Missed↓) | Precision↑ | Recall↑ | F1↑

Convention matches the notebook (safe = positive class):
  TP = Passed   (safe predicted safe)
  FN = False alarm
  TN = Rejected (unsafe predicted unsafe)
  FP = Missed

Repeated-runs mode: every (dataset, method, model) cell may contain N
generation artifacts (timestamped). Counts are averaged across runs first, and
the displayed precision/recall/F1 are computed from those same averaged counts.
The ± values report per-run metric std. Best F1 per block is bolded,
second-best underlined.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import oyaml as yaml
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ID = "project"

DATASET_ORDER = (
    "bank_marketing_analysis",
    "healthy_diet_dashboard",
    "omop_cdm_databricks",
)
DATASET_DISPLAY = {
    "bank_marketing_analysis": "Bank Marketing",
    "healthy_diet_dashboard": "Healthy Diet Dashboard",
    "omop_cdm_databricks": "OMOP CDM (Databricks)",
}

# (display label, artifact prefix, model name)
METHOD_ROWS: tuple[tuple[str, str, str], ...] = (
    # Model order within each block follows eid_bench (model_order in
    # eid_bench_experiments/results_analysis/result_full_approach.ipynb):
    # gemini-2.5-flash, gpt-4.1, gpt-4o, gpt-5-mini, gemini-2.5-pro, gpt-5.
    ("prismadv [gemini-2.5-flash]",  "prismadv_real_etl",   "gemini-2.5-flash"),
    ("prismadv [gpt-4.1]",           "prismadv_real_etl",   "gpt-4.1"),
    ("prismadv [gpt-5-mini]",        "prismadv_real_etl",   "gpt-5-mini"),
    ("prismadv [gemini-2.5-pro]",    "prismadv_real_etl",   "gemini-2.5-pro"),
    ("prismadv [gpt-5]",             "prismadv_real_etl",   "gpt-5"),
    ("zero_shot [gemini-2.5-flash]", "single_shot_real_etl", "gemini-2.5-flash"),
    ("zero_shot [gpt-4.1]",          "single_shot_real_etl", "gpt-4.1"),
    ("zero_shot [gpt-5-mini]",       "single_shot_real_etl", "gpt-5-mini"),
    ("zero_shot [gemini-2.5-pro]",   "single_shot_real_etl", "gemini-2.5-pro"),
    ("zero_shot [gpt-5]",            "single_shot_real_etl", "gpt-5"),
    ("few_shot [gemini-2.5-flash]",  "few_shot_real_etl",   "gemini-2.5-flash"),
    ("few_shot [gpt-4.1]",           "few_shot_real_etl",   "gpt-4.1"),
    ("few_shot [gpt-5-mini]",        "few_shot_real_etl",   "gpt-5-mini"),
    ("few_shot [gemini-2.5-pro]",    "few_shot_real_etl",   "gemini-2.5-pro"),
    ("few_shot [gpt-5]",             "few_shot_real_etl",   "gpt-5"),
    ("swe_agent [gemini-2.5-flash]", "swe_agent_real_etl",  "gemini-2.5-flash"),
    ("swe_agent [gpt-5]",            "swe_agent_real_etl",  "gpt-5"),
    ("deequ",                        "deequ",               "deequ"),
    ("autotest",                     "autotest",            "autotest"),
    ("tfdv",                         "tfdv",                "tfdv"),
    ("pocketflow",                   "pocketflow",          "gpt-5"),
    ("one-class-svm",                "one_class_svm",       "one_class_svm"),
    ("isolation-forest",             "isolation_forest",    "isolation_forest"),
    ("stats-novelty",                "stats_novelty",       "stats_novelty"),
)
TS_RE = re.compile(r"--(\d{8}_\d{6})$")

# Method blocks (display order) — drives both row order and best/second highlighting scope.
METHOD_BLOCKS = [
    ("deequ", [
        "deequ",
    ]),
    ("autotest", [
        "autotest",
    ]),
    ("tfdv", [
        "tfdv",
    ]),
    ("one-class-svm", [
        "one-class-svm",
    ]),
    ("isolation-forest", [
        "isolation-forest",
    ]),
    ("stats-novelty", [
        "stats-novelty",
    ]),
    ("zero-shot", [
        "zero_shot [gemini-2.5-flash]",
        "zero_shot [gpt-4.1]",
        "zero_shot [gpt-5-mini]",
        "zero_shot [gemini-2.5-pro]",
        "zero_shot [gpt-5]",
    ]),
    ("few-shot", [
        "few_shot [gemini-2.5-flash]",
        "few_shot [gpt-4.1]",
        "few_shot [gpt-5-mini]",
        "few_shot [gemini-2.5-pro]",
        "few_shot [gpt-5]",
    ]),
    ("swe-agent", [
        "swe_agent [gemini-2.5-flash]",
        "swe_agent [gpt-5]",
    ]),
    ("prismaDV", [
        "prismadv [gemini-2.5-flash]",
        "prismadv [gpt-4.1]",
        "prismadv [gpt-5-mini]",
        "prismadv [gemini-2.5-pro]",
        "prismadv [gpt-5]",
    ]),
]
ROW_ORDER = [method for _, methods in METHOD_BLOCKS for method in methods]

METHOD_DISPLAY = {
    "deequ":                       r"\texttt{deequ}",
    "autotest":                    r"\texttt{autotest}",
    "tfdv":                        r"\texttt{tfdv}",
    "one-class-svm":               r"\texttt{one-class-svm}",
    "isolation-forest":            r"\texttt{isolation-forest}",
    "stats-novelty":               r"\texttt{stats-novelty}",
    "prismadv [gemini-2.5-flash]": r"\texttt{prismaDV [gemini-2.5-flash]}",
    "prismadv [gpt-4.1]":          r"\texttt{prismaDV [gpt-4.1]}",
    "prismadv [gemini-2.5-pro]":   r"\texttt{prismaDV [gemini-2.5-pro]}",
    "prismadv [gpt-5-mini]":       r"\texttt{prismaDV [gpt-5-mini]}",
    "prismadv [gpt-5]":            r"\texttt{prismaDV [gpt-5]}",
    "zero_shot [gemini-2.5-flash]": r"\texttt{zero-shot [gemini-2.5-flash]}",
    "zero_shot [gpt-4.1]":          r"\texttt{zero-shot [gpt-4.1]}",
    "zero_shot [gpt-5-mini]":      r"\texttt{zero-shot [gpt-5-mini]}",
    "zero_shot [gemini-2.5-pro]":   r"\texttt{zero-shot [gemini-2.5-pro]}",
    "zero_shot [gpt-5]":           r"\texttt{zero-shot [gpt-5]}",
    "few_shot [gemini-2.5-flash]":  r"\texttt{few-shot [gemini-2.5-flash]}",
    "few_shot [gpt-4.1]":           r"\texttt{few-shot [gpt-4.1]}",
    "few_shot [gpt-5-mini]":       r"\texttt{few-shot [gpt-5-mini]}",
    "few_shot [gemini-2.5-pro]":    r"\texttt{few-shot [gemini-2.5-pro]}",
    "few_shot [gpt-5]":            r"\texttt{few-shot [gpt-5]}",
    "swe_agent [gemini-2.5-flash]": r"\texttt{swe-agent [gemini-2.5-flash]}",
    "swe_agent [gpt-5]":           r"\texttt{swe-agent [gpt-5]}",
}


def constraints_dir(example_id: str) -> Path:
    return PROJECT_ROOT / "data_processed" / "eid_bench_real" / example_id / "constraints" / SCRIPT_ID


def discover_runs(example_id: str) -> dict[str, list[str]]:
    base = constraints_dir(example_id)
    runs: dict[str, list[str]] = {}
    for label, prefix, model_name in METHOD_ROWS:
        stems = sorted(
            (p.stem for p in base.glob(f"{prefix}--{model_name}--*.yaml")),
            key=lambda s: (TS_RE.search(s).group(1) if TS_RE.search(s) else s, s),
        )
        if stems:
            runs[label] = stems
    return runs


def validation_path(example_id: str, variant: str, stem: str, label: str | None = None) -> Path:
    base = PROJECT_ROOT / "data_processed" / "eid_bench_real" / example_id / "constraints_validation" / SCRIPT_ID
    if variant == "clean":
        return base / "clean" / f"validation_results__{stem}.yaml"
    return base / "corrupted" / label / f"validation_results__{stem}.yaml"


def load_corruption_outcomes(example_id: str) -> dict[str, str]:
    errors_dir = PROJECT_ROOT / "benchmarks" / "EIDBench-real" / example_id / "errors"
    outcomes = {}
    for path in sorted(errors_dir.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        outcomes[path.stem] = cfg.get("expected_current_outcome", "unsafe")
    return outcomes


def summarize_validation(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    s = data["summary"]
    total = s["passed_warning"] + s["failed_warning"] + s["passed_error"] + s["failed_error"]
    return {
        "num_non_compilable": s["non_compilable"],
        "total_constraints": total,
        "predicted_as_safe": s["failed_error"] == 0,
    }


def collect_rows() -> pd.DataFrame:
    rows = []
    for example_id in DATASET_ORDER:
        outcomes = load_corruption_outcomes(example_id)
        for llm, stems in discover_runs(example_id).items():
            # We evaluate only on the error-injected (corrupted) batches, not the
            # unmodified clean batch. Assign run_idx only after confirming the
            # stem has at least one validation file; otherwise incomplete marker
            # artifacts can split pooled overall metrics into artificial runs.
            corr_root = (
                PROJECT_ROOT / "data_processed" / "eid_bench_real" / example_id
                / "constraints_validation" / SCRIPT_ID / "corrupted"
            )
            if not corr_root.exists():
                continue
            valid_stems = [
                stem for stem in stems
                if any(corr_root.glob(f"*/validation_results__{stem}.yaml"))
            ]
            for run_idx, stem in enumerate(valid_stems):
                for label_dir in sorted(corr_root.iterdir()):
                    if not label_dir.is_dir():
                        continue
                    vp = label_dir / f"validation_results__{stem}.yaml"
                    if not vp.exists():
                        continue
                    rec = summarize_validation(vp)
                    rec.update({
                        "dataset_name": example_id,
                        "llm_name": llm,
                        "run_idx": run_idx,
                        "is_safe": outcomes.get(label_dir.name, "unsafe") == "safe",
                    })
                    rows.append(rec)
    return pd.DataFrame(rows)


def compute_run_metrics(group: pd.DataFrame) -> dict:
    y_true = group["is_safe"].astype(bool)
    y_pred = group["predicted_as_safe"].astype(bool)
    tp = int((y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )
    return {
        "exec": group["total_constraints"].mean() - group["num_non_compilable"].mean(),
        "non_exec": group["num_non_compilable"].mean(),
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
        "precision": precision, "recall": recall, "f1": f1,
    }


def aggregate_runs(per_run: list[dict]) -> dict:
    """Mean counts across runs; derive displayed metrics from those counts."""
    if not per_run:
        return {
            "avg_exec": float("nan"), "avg_non_exec": float("nan"),
            "TP": float("nan"), "FN": float("nan"), "TN": float("nan"), "FP": float("nan"),
            "precision_mean": float("nan"), "precision_std": 0.0,
            "recall_mean":    float("nan"), "recall_std":    0.0,
            "f1_mean":        float("nan"), "f1_std":        0.0,
            "n_runs": 0,
        }
    df_runs = pd.DataFrame(per_run)

    def finite_std(series: pd.Series) -> float:
        finite = series[np.isfinite(series)]
        return finite.std(ddof=1) if len(finite) > 1 else 0.0

    tp = df_runs["TP"].mean()
    fn = df_runs["FN"].mean()
    tn = df_runs["TN"].mean()
    fp = df_runs["FP"].mean()
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )

    return {
        "avg_exec":      df_runs["exec"].mean(),
        "avg_non_exec":  df_runs["non_exec"].mean(),
        "TP":            tp,
        "FN":            fn,
        "TN":            tn,
        "FP":            fp,
        "precision_mean": precision,
        "precision_std":  finite_std(df_runs["precision"]),
        "recall_mean":    recall,
        "recall_std":     finite_std(df_runs["recall"]),
        "f1_mean":        f1,
        "f1_std":         finite_std(df_runs["f1"]),
        "n_runs":         int(len(df_runs)),
    }


def per_run_metrics_for_overall(df: pd.DataFrame) -> dict[str, list[dict]]:
    """Per (method, run_idx) pool counts across all datasets, then compute F1.

    Pairs runs across datasets by run_idx (i.e. by sorted timestamp). If a cell
    has fewer runs than its peers, that paired index simply pools fewer rows.
    """
    out: dict[str, list[dict]] = {}
    for llm, sub in df.groupby("llm_name", sort=False):
        per_run = []
        for run_idx, run_sub in sub.groupby("run_idx", sort=True):
            per_run.append(compute_run_metrics(run_sub))
        out[llm] = per_run
    return out


def rank_for_highlight(values: list[float]) -> tuple[set[int], set[int]]:
    indexed = [(i, round(v, 6)) for i, v in enumerate(values) if np.isfinite(v)]
    if not indexed:
        return set(), set()
    sorted_vals = sorted({v for _, v in indexed}, reverse=True)
    best_val = sorted_vals[0]
    second_val = sorted_vals[1] if len(sorted_vals) >= 2 else None
    best = {i for i, v in indexed if v == best_val}
    second = {i for i, v in indexed if v == second_val} if second_val is not None else set()
    return best, second


def format_pct(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "--"
    return f"{mean * 100:.1f}\\% $\\pm$ {std * 100:.1f}\\%"


def emit_block(rows: list[tuple[str, dict]], lines: list[str], best_idx: set[int], second_idx: set[int]) -> None:
    for i, (method_key, m) in enumerate(rows):
        method_label = METHOD_DISPLAY[method_key]
        prec = format_pct(m["precision_mean"], m["precision_std"])
        rec = format_pct(m["recall_mean"], m["recall_std"])
        f1 = format_pct(m["f1_mean"], m["f1_std"])
        if i in best_idx:
            f1 = f"\\textbf{{{f1}}}"
        elif i in second_idx:
            f1 = f"\\underline{{{f1}}}"
        lines.append(
            f"        {method_label} & "
            f"{m['avg_exec']:.1f} & {m['avg_non_exec']:.1f} & "
            f"{m['TP']:.1f} & {m['FN']:.1f} & "
            f"{m['TN']:.1f} & {m['FP']:.1f} & "
            f"{prec} & {rec} & {f1} \\\\"
        )


def build_table_block(metrics: dict[str, dict]) -> list[str]:
    block_lines: list[str] = []
    all_methods = [method for _, methods in METHOD_BLOCKS for method in methods if metrics.get(method, {}).get("n_runs", 0) > 0]
    f1_means = [metrics.get(m, {}).get("f1_mean", float("nan")) for m in all_methods]
    best_global, second_global = rank_for_highlight(f1_means)

    cursor = 0
    emitted_blocks = 0
    for _, methods in METHOD_BLOCKS:
        present_methods = [m for m in methods if metrics.get(m, {}).get("n_runs", 0) > 0]
        if not present_methods:
            continue
        if emitted_blocks > 0:
            block_lines.append(r"        \midrule")
        block_rows = [(m, metrics[m]) for m in present_methods]
        block_global_indices = list(range(cursor, cursor + len(present_methods)))
        local_best = {
            block_global_indices.index(g) for g in best_global if g in block_global_indices
        }
        local_second = {
            block_global_indices.index(g) for g in second_global if g in block_global_indices
        }
        emit_block(block_rows, block_lines, local_best, local_second)
        cursor += len(present_methods)
        emitted_blocks += 1
    return block_lines


def main() -> None:
    df = collect_rows()
    out_dir = Path(__file__).resolve().parent
    tex_path = out_dir / "eid_bench_real_f1_table.tex"
    pdf_path = out_dir / "eid_bench_real_f1_table.pdf"

    # Per-dataset metrics: aggregate per-run F1 within each (dataset, llm) cell.
    per_dataset_metrics: dict[str, dict[str, dict]] = {}
    for dataset in DATASET_ORDER:
        sub = df[df["dataset_name"] == dataset]
        per_dataset_metrics[dataset] = {}
        for llm, llm_sub in sub.groupby("llm_name", sort=False):
            per_run = [compute_run_metrics(rsub) for _, rsub in llm_sub.groupby("run_idx", sort=True)]
            per_dataset_metrics[dataset][llm] = aggregate_runs(per_run)

    # Overall metrics: pool counts across datasets per matched run_idx, then aggregate.
    overall_per_run = per_run_metrics_for_overall(df)
    overall_metrics = {llm: aggregate_runs(runs) for llm, runs in overall_per_run.items()}

    body: list[str] = []
    body.extend(build_table_block(overall_metrics))
    table_body = "\n".join(body)

    doc = r"""\documentclass[10pt]{article}
\usepackage[a4paper,landscape,margin=1cm]{geometry}
\usepackage{booktabs}
\usepackage{array}
\pagestyle{empty}

\begin{document}
\begin{table}[t]
\centering
\renewcommand{\arraystretch}{1.1}
\setlength{\tabcolsep}{4pt}
\small
\begin{tabular}{l c c | c c | c c | c c c}
\toprule
& & & \multicolumn{2}{c|}{\textbf{Data to Pass}} & \multicolumn{2}{c|}{\textbf{Data to Reject}} & \multicolumn{3}{c}{\textbf{Metrics}} \\
\textbf{Method} & \textbf{Exec.} & \textbf{Non-exec.}
& \textbf{Passed$\uparrow$} & \textbf{False alarm$\downarrow$}
& \textbf{Rejected$\uparrow$} & \textbf{Missed$\downarrow$}
& \textbf{Precision$\uparrow$} & \textbf{Recall$\uparrow$} & \textbf{F1$\uparrow$} \\
\midrule
""" + table_body + r"""
\bottomrule
\end{tabular}
\caption{Detection performance with respect to the impact of data errors on downstream tasks in \textsc{EIDBench-real} (overall, all 3 datasets pooled, mean $\pm$ std across runs). The best F1 (mean) is in bold, the second-best underlined. \texttt{Exec.} reports the average number of executable constraints, \texttt{Non-exec.} the average non-executable. Convention: \emph{safe = positive class}; Passed = safe bundles correctly cleared, False alarm = safe bundles incorrectly flagged, Rejected = harmful corruptions correctly detected, Missed = harmful corruptions missed.}
\label{tab:realetlbench-f1}
\end{table}

\end{document}
"""
    tex_path.write_text(doc, encoding="utf-8")

    if not shutil.which("pdflatex"):
        raise SystemExit("pdflatex not found on PATH")

    for _ in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", str(out_dir), str(tex_path)],
            cwd=out_dir, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            raise SystemExit(f"pdflatex failed (rc={proc.returncode})")

    for suffix in (".aux", ".log", ".out"):
        aux = pdf_path.with_suffix(suffix)
        if aux.exists():
            aux.unlink()

    print(f"Wrote: {tex_path}")
    print(f"Wrote: {pdf_path}")


if __name__ == "__main__":
    main()
