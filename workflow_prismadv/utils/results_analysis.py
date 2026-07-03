from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from tqdm import tqdm

from prismadv.data_models import ValidationResults
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root


def deduplicate_by_dataset(results_df: pd.DataFrame, unique_subset_cols: list) -> pd.DataFrame:
    """
    Deduplicate a DataFrame by dataset, keeping the first occurrence based on timestamp order.
    If a column in unique_subset_cols is missing from results_df, print a warning and ignore it.
    """
    df_before_drop = results_df.copy()
    df_before_drop['timestamp'] = pd.to_numeric(df_before_drop['timestamp'], errors='coerce')

    # filter valid columns
    existing_cols = [c for c in unique_subset_cols if c in df_before_drop.columns]
    missing_cols = [c for c in unique_subset_cols if c not in df_before_drop.columns]
    if missing_cols:
        print(f"[deduplicate_by_dataset] Ignored missing columns: {missing_cols}")

    dedup_list = []
    for ds in df_before_drop['dataset_name'].unique():
        df_ds = df_before_drop[df_before_drop['dataset_name'] == ds].copy()
        df_ds = df_ds.sort_values('timestamp')
        df_unique_ds = df_ds.drop_duplicates(subset=existing_cols, keep='first')
        dedup_list.append(df_unique_ds)

    return pd.concat(dedup_list, ignore_index=True)


import pandas as pd
from typing import Dict, List, Iterable


def format_confusion_table_rows(
        confusion_matrices_df: pd.DataFrame,
        dataset_name_options: Iterable[str],
        model_order: Iterable[str],
        model_prefix: str = "prismaDV",
        decimals: int = 1,
        print_output: bool = True,
) -> Dict[str, List[str]]:
    # aggregate like the original snippet
    results = (
        confusion_matrices_df
        .groupby(['model', 'dataset_name'], as_index=False)
        .agg({
            "average_constraints": "mean",
            "average_num_non_compilable": "mean",
            "safe,predicted_as_safe": "sum",
            "safe,predicted_as_unsafe": "sum",
            "unsafe,predicted_as_unsafe": "sum",
            "unsafe,predicted_as_safe": "sum",
            "precision": "mean",
            "recall": "mean",
            "f1": "mean"
        })
    )

    def fmt_percent(x):
        if isinstance(x, str) and x == '-':
            return '-'
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return '-'
        return f"{x * 100:.{decimals}f}\\%"

    out: Dict[str, List[str]] = {}

    for dataset in dataset_name_options:
        rows: List[str] = []
        single = results[results['dataset_name'] == dataset].copy()
        if single.empty:
            if print_output:
                print(f"=== Dataset: {dataset} ===")
            out[dataset] = rows
            continue

        # enforce model order, dropping missing
        single = single.set_index('model').loc[
            [m for m in model_order if m in single['model'].index]
        ].reset_index()

        if print_output:
            print(f"=== Dataset: {dataset} ===")

        for _, row in single.iterrows():
            model = row['model']
            avg_cons = row['average_constraints']
            average_num_non_compilable = row['average_num_non_compilable']
            safe_safe = row['safe,predicted_as_safe']
            safe_unsafe = row['safe,predicted_as_unsafe']
            unsafe_unsafe = row['unsafe,predicted_as_unsafe']
            unsafe_safe = row['unsafe,predicted_as_safe']
            precision = row['precision']
            recall = row['recall']
            f1 = row['f1']

            line = (
                f"& \\texttt{{{model_prefix} [{model}]}} "
                f"& {avg_cons:.1f} "
                f"& {average_num_non_compilable:.1f} "
                f"& {int(safe_safe)} "
                f"& {int(safe_unsafe)} "
                f"& {int(unsafe_unsafe)} "
                f"& {int(unsafe_safe)} "
                f"& {fmt_percent(precision)} "
                f"& {fmt_percent(recall)} "
                f"& {fmt_percent(f1)} \\\\"
            )
            rows.append(line)
            if print_output:
                print(line)

        out[dataset] = rows

    return out


def format_results_latex(
        confusion_matrex_df: pd.DataFrame,
        dataset_name_options: list,
        model_order: list,
        decimals: int = 1,
        model_label_prefix: str = "prismaDV",
        include_overall: bool = True,
        beta: float = 2,  # NEW: F-beta weight
) -> str:
    """
    Aggregate metrics and emit LaTeX table rows per dataset and per model.
    Optionally includes an overall summary combining all datasets.
    Adds an F_beta column at the end.

    Args:
        confusion_matrex_df: Input DataFrame.
        dataset_name_options: Iterable of dataset names to render.
        model_order: Preferred ordering of models.
        decimals: Decimal places for percentage metrics.
        model_label_prefix: Label prefix in the LaTeX output.
        include_overall: Whether to include overall aggregated results.
        beta: Beta for F_beta. <1 emphasizes precision, >1 emphasizes recall.

    Returns:
        str: Multi-line string with LaTeX rows.
    """

    def fmt_percent(x, decimals=1):
        if isinstance(x, str) and x == '-':
            return '-'
        if pd.isna(x):
            return '-'
        return f"{float(x) * 100:.{decimals}f}\\%"

    def fbeta_score(p, r, beta):
        if pd.isna(p) or pd.isna(r):
            return float('nan')
        b2 = beta * beta
        denom = b2 * p + r
        if denom == 0:
            return 0.0
        return (1 + b2) * p * r / denom

    def pr_f1_from_counts(tp, fp, fn):
        # Pool counts then recompute (matches EIDBench-real generate_latex_table),
        # rather than averaging per-dataset ratios.
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float('nan')
        return precision, recall, f1

    confusion_matrex_df['f1'] = pd.to_numeric(confusion_matrex_df['f1'], errors='coerce')
    confusion_matrex_df['precision'] = pd.to_numeric(confusion_matrex_df['precision'], errors='coerce')
    confusion_matrex_df['recall'] = pd.to_numeric(confusion_matrex_df['recall'], errors='coerce')
    # Aggregate per-dataset like original
    results = (
        confusion_matrex_df
        .groupby(['model', 'dataset_name'], as_index=False)
        .agg({
            "average_constraints": "mean",
            "average_num_non_compilable": "mean",
            "safe,predicted_as_safe": "sum",
            "safe,predicted_as_unsafe": "sum",
            "unsafe,predicted_as_unsafe": "sum",
            "unsafe,predicted_as_safe": "sum",
            "precision": "mean",
            "recall": "mean",
            "f1": "mean"
        })
    )

    lines = []

    # Per-dataset outputs
    for dataset in dataset_name_options:
        lines.append(f"%% === Dataset: {dataset} ===")
        single_dataset_result = results[results['dataset_name'] == dataset].copy()
        if single_dataset_result.empty:
            continue

        present_models = [m for m in model_order if m in single_dataset_result['model'].values]
        if not present_models:
            continue

        single_dataset_result = (
            single_dataset_result
            .set_index('model')
            .loc[present_models]
            .reset_index()
        )

        for _, row in single_dataset_result.iterrows():
            model = row['model']
            avg_cons = row['average_constraints']
            average_num_non_compilable = row['average_num_non_compilable']
            safe_safe = int(row['safe,predicted_as_safe'])
            safe_unsafe = int(row['safe,predicted_as_unsafe'])
            unsafe_unsafe = int(row['unsafe,predicted_as_unsafe'])
            unsafe_safe = int(row['unsafe,predicted_as_safe'])
            precision, recall, f1 = pr_f1_from_counts(safe_safe, unsafe_safe, safe_unsafe)
            fbeta = fbeta_score(precision, recall, beta)

            lines.append(
                f"& \\texttt{{{model_label_prefix} [{model}]}} "
                f"& {avg_cons:.1f} "
                f"& {average_num_non_compilable:.1f} "
                f"& {safe_safe} "
                f"& {safe_unsafe} "
                f"& {unsafe_unsafe} "
                f"& {unsafe_safe} "
                f"& {fmt_percent(precision, decimals)} "
                f"& {fmt_percent(recall, decimals)} "
                f"& {fmt_percent(f1, decimals)} \\\\"
                # f"& {fmt_percent(fbeta, decimals)} \\\\"
            )

    # Overall results across all datasets
    if include_overall:
        lines.append("\n%% === Overall (All Datasets Combined) ===")
        overall_results = (
            confusion_matrex_df
            .groupby(['model'], as_index=False)
            .agg({
                "average_constraints": "mean",
                "average_num_non_compilable": "mean",
                "safe,predicted_as_safe": "sum",
                "safe,predicted_as_unsafe": "sum",
                "unsafe,predicted_as_unsafe": "sum",
                "unsafe,predicted_as_safe": "sum",
                "precision": "mean",
                "recall": "mean",
                "f1": "mean"
            })
        )

        present_models = [m for m in model_order if m in overall_results['model'].values]
        overall_results = (
            overall_results
            .set_index('model')
            .loc[present_models]
            .reset_index()
        )

        for _, row in overall_results.iterrows():
            model = row['model']
            avg_cons = row['average_constraints']
            average_num_non_compilable = row['average_num_non_compilable']
            safe_safe = int(row['safe,predicted_as_safe'])
            safe_unsafe = int(row['safe,predicted_as_unsafe'])
            unsafe_unsafe = int(row['unsafe,predicted_as_unsafe'])
            unsafe_safe = int(row['unsafe,predicted_as_safe'])
            precision, recall, f1 = pr_f1_from_counts(safe_safe, unsafe_safe, safe_unsafe)
            fbeta = fbeta_score(precision, recall, beta)

            lines.append(
                f"& \\texttt{{{model_label_prefix} [{model}]}} "
                f"& {avg_cons:.1f} "
                f"& {average_num_non_compilable:.1f} "
                f"& {safe_safe} "
                f"& {safe_unsafe} "
                f"& {unsafe_unsafe} "
                f"& {unsafe_safe} "
                f"& {fmt_percent(precision, decimals)} "
                f"& {fmt_percent(recall, decimals)} "
                f"& {fmt_percent(f1, decimals)} \\\\"
                # f"& {fmt_percent(fbeta, decimals)} \\\\"
            )

    return "\n".join(lines)


def build_correct_grids(
        df_all_unique: pd.DataFrame,
        dataset_name_options: list,
        threshold: float = 0.5,
) -> dict:
    """
    Build per-model, per-dataset grids with 4 classes per cell:
      tp: predicted_as_safe == True and is_safe == True
      tn: predicted_as_safe == False and is_safe == False
      fp: predicted_as_safe == True and is_safe == False
      fn: predicted_as_safe == False and is_safe == True

    Cell value = dominant class if its proportion >= threshold, else NaN.
    Returns:
        grids: { model: { dataset: DataFrame[str|NaN] } }
               index=processed_data_label, columns=script_name
    """
    if df_all_unique.empty:
        return {}

    df = df_all_unique.copy()

    # normalize booleans once upfront
    def to_bool(x):
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        return True if s == "true" else False if s == "false" else np.nan

    df["is_safe_bool"] = df["is_safe"].map(to_bool)
    df["pred_bool"] = df["predicted_as_safe"].map(to_bool)

    # drop rows with unknown truth or prediction
    df = df.dropna(subset=["is_safe_bool", "pred_bool"])

    # Convert to string once for grouping
    df["_script_str"] = df["script_name"].astype(str)
    df["_label_str"] = df["processed_data_label"].astype(str)

    # Pre-compute confusion category for each row
    df["_category"] = np.where(
        df["pred_bool"] & df["is_safe_bool"], "tp",
        np.where(
            ~df["pred_bool"] & ~df["is_safe_bool"], "tn",
            np.where(
                df["pred_bool"] & ~df["is_safe_bool"], "fp", "fn"
            )
        )
    )

    # Aggregate counts using groupby once
    counts = df.groupby(
        ["llm_name", "dataset_name", "_label_str", "_script_str", "_category"],
        sort=False
    ).size().unstack(fill_value=0)

    # Ensure all categories exist
    for cat in ["tp", "tn", "fp", "fn"]:
        if cat not in counts.columns:
            counts[cat] = 0

    counts["total"] = counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"]
    counts = counts.reset_index()

    grids: dict[str, dict[str, pd.DataFrame]] = {}

    for model, model_counts in counts.groupby("llm_name", sort=False):
        grids[model] = {}

        unique_datasets = model_counts["dataset_name"].unique()
        ordered_datasets = sorted(
            unique_datasets,
            key=lambda x: dataset_name_options.index(x) if x in dataset_name_options else len(dataset_name_options),
        )

        for dataset_name in ordered_datasets:
            ddf = model_counts[model_counts["dataset_name"] == dataset_name]

            # Get all unique values for this dataset
            y_vals = list(ddf["_label_str"].unique())
            x_vals = sorted(ddf["_script_str"].unique().tolist())

            # Create pivot tables for each category
            pivot_data = ddf.pivot_table(
                index="_label_str",
                columns="_script_str",
                values=["tp", "tn", "fp", "fn", "total"],
                aggfunc="sum",
                fill_value=0
            )

            # Build the grid
            grid_data = []
            for y in y_vals:
                row_vals = []
                for x in x_vals:
                    try:
                        total = pivot_data.loc[y, ("total", x)]
                        if total == 0:
                            row_vals.append(np.nan)
                            continue

                        props = {
                            "tp": pivot_data.loc[y, ("tp", x)] / total,
                            "tn": pivot_data.loc[y, ("tn", x)] / total,
                            "fp": pivot_data.loc[y, ("fp", x)] / total,
                            "fn": pivot_data.loc[y, ("fn", x)] / total,
                        }
                        cls, p = max(props.items(), key=lambda kv: kv[1])
                        row_vals.append(cls if p >= threshold else np.nan)
                    except KeyError:
                        row_vals.append(np.nan)
                grid_data.append(row_vals)

            grids[model][dataset_name] = pd.DataFrame(grid_data, index=y_vals, columns=x_vals, dtype="object")

    return grids


def plot_correct_grids(
        grids: dict,
        model_order: list | None = None,
        dataset_name_options: list | None = None,
        figsize_base: float = 4.0,
):
    """
    Plot 4-class grids (tp, fn, tn, fp) with NaN as no data.

    Color code:
      tp -> green
      tn -> blue-gray
      fp -> orange
      fn -> red
      NaN -> light gray
    """
    # fixed class order → int codes
    class_to_int = {"fn": 0, "fp": 1, "tn": 2, "tp": 3}
    colors = ["#D73027", "#F6A03A", "#6BA3C8", "#1A9850"]  # fn, fp, tn, tp
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)
    cmap.set_bad(color="#D3D3D3")  # NaN

    models = list(grids.keys())
    if model_order:
        models = [m for m in model_order if m in grids] + [m for m in models if m not in (model_order or [])]

    for model in models:
        ds_dict = grids[model]
        datasets = list(ds_dict.keys())
        if dataset_name_options:
            datasets = sorted(
                datasets,
                key=lambda x: dataset_name_options.index(x) if x in dataset_name_options else len(dataset_name_options),
            )
        if not datasets:
            continue

        fig, axes = plt.subplots(
            1, len(datasets),
            figsize=(figsize_base * len(datasets), 3.2),
            squeeze=False
        )
        axes = axes[0]

        for ax, dataset_name in zip(axes, datasets):
            grid_df = ds_dict[dataset_name]

            # map classes to ints, keep NaN
            mapped = grid_df.replace(class_to_int)
            arr = mapped.to_numpy(dtype=float)

            im = ax.imshow(arr, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

            ax.set_xticks(np.arange(len(grid_df.columns)))
            ax.set_yticks(np.arange(len(grid_df.index)))
            ax.set_xticklabels(grid_df.columns.astype(str), rotation=90, fontsize=7)
            ax.set_yticklabels(grid_df.index.astype(str), fontsize=7)
            ax.set_xlabel("script_name", fontsize=9)
            ax.set_ylabel("processed_data_label", fontsize=9)
            ax.set_title(f"{model}\n{dataset_name}", fontsize=9, pad=6)

            ax.set_xticks(np.arange(len(grid_df.columns)) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(grid_df.index)) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linewidth=0.8)
            ax.tick_params(which="minor", bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        handles = [
            Patch(facecolor="#1A9850", edgecolor='none', label='TP'),
            Patch(facecolor="#6BA3C8", edgecolor='none', label='TN'),
            Patch(facecolor="#F6A03A", edgecolor='none', label='FP'),
            Patch(facecolor="#D73027", edgecolor='none', label='FN'),
            Patch(facecolor="#D3D3D3", edgecolor='none', label='No data'),
        ]
        fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=8)
        fig.suptitle(f"Model: {model}", fontsize=11, y=1.05)
        fig.tight_layout()
        plt.show()


# ---- generic collector -------------------------------------------------------

def _process_single_constraint_file(
        item: dict,
        row_meta_extractor,
        constraints_parser,
        validation_check_kwargs_fn,
        model_order,
) -> list[dict]:
    """
    Process a single constraint file and return rows for clean/corrupted data.
    Returns a list of row dicts (0, 1, or 2 rows depending on available validation results).
    """
    pm = item["pm"]
    dataset_name = item["dataset_name"]
    subtask_name = item["subtask_name"]
    processed_data_label = item["processed_data_label"]
    script_name = item["script_name"]
    constraint_file = item["constraint_file"]
    validation_dir = item["validation_dir"]

    rows = []

    try:
        with open(constraint_file, "r") as f:
            raw = yaml.load(f, Loader=yaml.FullLoader)
    except Exception:
        return rows

    # variant-specific metadata
    try:
        meta = row_meta_extractor(constraint_file, raw)
    except Exception:
        return rows

    llm_name = meta.get("llm_name")
    if model_order is not None and llm_name not in model_order:
        return rows

    # optional parsed constraints (kept for parity with original code)
    try:
        _ = constraints_parser(raw)
    except Exception:
        return rows

    # iterate clean/corrupted
    for is_clean in (True, False):
        stem = constraint_file.stem
        val_name = (
            f"validation_results_on_clean_test_data__{stem}.yaml"
            if is_clean else
            f"validation_results_on_corrupted_test_data__{stem}.yaml"
        )
        val_path = validation_dir / val_name
        try:
            vr = ValidationResults.from_yaml(val_path)
        except FileNotFoundError:
            continue

        kwargs = validation_check_kwargs_fn() if validation_check_kwargs_fn else {}
        num_passed_warning, num_failed_warning, num_failed_error, num_passed_error, num_non_compilable = vr.check_result(
            **kwargs)
        total_constraints = num_passed_warning + num_failed_warning + num_failed_error + num_passed_error
        predicted_as_safe = (num_failed_error == 0)

        exec_path = pm.get_execution_output_validation_path(subtask_name, processed_data_label,
                                                            script_name) / "basic_metrics_evaluation.json"
        try:
            with open(exec_path, "r") as f:
                exec_results = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            continue

        is_safe = exec_results['clean_data_is_safe'] if is_clean else exec_results['corrupted_data_is_safe']

        row = {
            "llm_name": llm_name,
            "timestamp": meta.get("timestamp"),
            "llm_temperature": meta.get("llm_temperature"),
            "dataset_name": dataset_name,
            "subtask_name": subtask_name,
            "processed_data_label": processed_data_label,
            "script_name": script_name,
            "is_clean": is_clean,
            "num_passed_warning": num_passed_warning,
            "num_failed_warning": num_failed_warning,
            "num_failed_error": num_failed_error,
            "num_passed_error": num_passed_error,
            "num_non_compilable": num_non_compilable,
            "total_constraints": total_constraints,
            "predicted_as_safe": predicted_as_safe,
            "is_safe": is_safe,
        }
        # include use_async only if present
        if "use_async" in meta:
            row["use_async"] = meta["use_async"]

        rows.append(row)

    return rows


def collect_results(
        dataset_name_options,
        unique_subset_cols,
        constraint_glob: str,
        row_meta_extractor,
        constraints_parser,
        validation_check_kwargs_fn=None,
        model_order=None,
        show_progress: bool = True,
        max_workers: int | None = None,
) -> pd.DataFrame:
    """
    Walk project structure, read constraints + validation + execution results,
    emit a unified results_df, then deduplicate by dataset.

    Args:
        dataset_name_options: iterable of dataset names to scan.
        unique_subset_cols: columns for deduplication.
        constraint_glob: glob pattern for constraint files (e.g., "post_processed_prismadv--*.yaml").
        row_meta_extractor: fn(constraint_file_path, raw_constraint_dict) -> dict with keys:
            - llm_name (str)
            - llm_temperature (float or None)
            - use_async (optional, bool)  # include only if available
            - timestamp (str)
        constraints_parser: fn(raw_constraint_dict) -> Any  # parsed constraints object if needed
        validation_check_kwargs_fn: optional fn() -> dict passed to ValidationResults.check_result(...)
        model_order: optional list to filter llm_name.
        show_progress: whether to show a progress bar (default True).
        max_workers: max number of parallel threads for file I/O (default: None = auto, set to 1 for sequential).

    Returns:
        pd.DataFrame (deduplicated by dataset).
    """
    # First pass: collect all work items to know total count for progress bar
    work_items = []
    for dataset_name in dataset_name_options:
        pm = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
        for subtask_name in pm.get_available_subtasks():
            processed_data_labels = pm.get_available_processed_data_labels_for_subtask(subtask_name)
            script_names = [p.stem for p in pm.get_available_script_path_list_for_subtask(subtask_name)]

            for processed_data_label in processed_data_labels:
                if int(processed_data_label) == 0:
                    continue

                for script_name in script_names:
                    constraints_path: Path = pm.get_constraints_path(subtask_name, processed_data_label, script_name)
                    validation_dir: Path = pm.get_constraints_validation_path(subtask_name, processed_data_label,
                                                                              script_name)

                    for constraint_file in constraints_path.glob(constraint_glob):
                        work_items.append({
                            "pm": pm,
                            "dataset_name": dataset_name,
                            "subtask_name": subtask_name,
                            "processed_data_label": processed_data_label,
                            "script_name": script_name,
                            "constraint_file": constraint_file,
                            "validation_dir": validation_dir,
                        })

    rows_list = []  # Collect rows in a list instead of concatenating DataFrames

    # Process in parallel using ThreadPoolExecutor (I/O-bound)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                _process_single_constraint_file,
                item,
                row_meta_extractor,
                constraints_parser,
                validation_check_kwargs_fn,
                model_order,
            ): item
            for item in work_items
        }

        # Collect results with progress bar
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Collecting results",
            disable=not show_progress,
        ):
            try:
                rows = future.result()
                rows_list.extend(rows)
            except Exception:
                # Skip failed items
                pass

    # Create DataFrame once from the list of rows
    results_df = pd.DataFrame(rows_list)

    # deduplicate
    if results_df.empty:
        return results_df
    df_all_unique = deduplicate_by_dataset(results_df, unique_subset_cols=unique_subset_cols)
    return df_all_unique


def build_confusion_matrices(df_all_unique: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-(dataset, model, subtask) confusion counts and metrics.

    Expects columns:
      ['dataset_name','llm_name','subtask_name','total_constraints',
       'is_safe','predicted_as_safe']

    Returns:
      pd.DataFrame with columns:
        ['dataset_name','model','sub_task','average_constraints',
         'safe,predicted_as_safe','safe,predicted_as_unsafe',
         'unsafe,predicted_as_safe','unsafe,predicted_as_unsafe',
         'precision','recall','f1']
      NaNs are filled with '-'.
    """
    if df_all_unique.empty:
        return pd.DataFrame()

    # Pre-compute boolean columns once (avoid repeated string conversions)
    df = df_all_unique.copy()
    df["_y_true"] = df["is_safe"].astype(str).str.lower().map({"true": True, "false": False})
    df["_y_pred"] = df["predicted_as_safe"].astype(str).str.lower().map({"true": True, "false": False})

    out = []

    # Use groupby instead of triple nested loop
    for (dataset_name, model, subtask), subtask_df in df.groupby(
        ['dataset_name', 'llm_name', 'subtask_name'], sort=False
    ):
        avg_cons = subtask_df['total_constraints'].mean()
        avg_num_non_compilable = subtask_df['num_non_compilable'].mean()

        y_true = subtask_df["_y_true"]
        y_pred = subtask_df["_y_pred"]

        tp = (y_true & y_pred).sum()
        fn = (y_true & ~y_pred).sum()
        fp = (~y_true & y_pred).sum()
        tn = (~y_true & ~y_pred).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
            else np.nan
        )

        out.append({
            "dataset_name": dataset_name,
            "model": model,
            "sub_task": subtask,
            "average_constraints": avg_cons,
            "average_num_non_compilable": avg_num_non_compilable,
            "safe,predicted_as_safe": tp,
            "safe,predicted_as_unsafe": fn,
            "unsafe,predicted_as_unsafe": tn,
            "unsafe,predicted_as_safe": fp,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    return pd.DataFrame(out).fillna('-')
