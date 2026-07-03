#!/usr/bin/env python3
"""Generate EIDBench-real table-local constraints with EIDBench-real PrismaDV."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import oyaml as yaml

from prismadv.data_models.config import PrismaDVConfig
from prismadv.data_models.constraints_v2 import ser_column_group_key
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.loader import MultiTableLoader
from prismadv.llm.langchain.models.prismadv_multi_table import MultiTablePrismaLangChainDV
from prismadv.project_manager import MultiTableProjectManager
from workflow_prismadv.eid_bench_real_experiments import utils


DEFAULT_MODEL_NAME = "gpt-5-mini"
DEFAULT_SCRIPT_ID = "project"
DEFAULT_FILENAME_PREFIX = "prismadv_real_etl"


def log_progress(message: str) -> None:
    print(f"[real-etl-prismadv] {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-id", default="omop_cdm_synthea")
    parser.add_argument("--script-id", default=DEFAULT_SCRIPT_ID)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--sync", action="store_true", help="Use sync LLM calls instead of async calls.")
    parser.add_argument("--no-dataflow", action="store_true")
    parser.add_argument(
        "--no-correlation",
        action="store_true",
        help="Disable same-table column-group correlation discovery.",
    )
    parser.add_argument(
        "--no-assumption",
        action="store_true",
        help="Skip the assumption stage and run direct code generation.",
    )
    parser.add_argument(
        "--filename-prefix",
        default=DEFAULT_FILENAME_PREFIX,
        help=(
            "Prefix for the generated constraint artifact filename. Use a custom "
            "prefix (e.g., prismadv_real_etl_wo_correlation) to keep ablation runs "
            "separate from main-table artifacts."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-run generation even if a matching artifact already exists. "
            "Prior artifacts are preserved; the new run is written to a fresh timestamped file."
        ),
    )
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def default_prismadv_input_path(
    example_id: str,
    script_id: str,
    project_root: Path,
) -> Path:
    return utils.constraints_dir(example_id, script_id, project_root) / "prismadv_inputs.yaml"


def make_output_filename(model_name: str, filename_prefix: str = DEFAULT_FILENAME_PREFIX) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{filename_prefix}--{model_name}--{timestamp}.yaml"


def build_downstream_task_description(prismadv_inputs: dict[str, Any]) -> str:
    context = prismadv_inputs["context"]
    script = context["script"]
    if context["script_id"] == "project":
        script_descriptions = [
            (
                f"{script_info['script_id']} reads {', '.join(script_info['reads'])} "
                f"and writes {', '.join(script_info['writes'])}"
            )
            for script_info in script.get("scripts", [])
        ]
        return (
            f"Whole EIDBench-real project for {context['display_name']}. "
            f"It reads tables {', '.join(script['reads'])}. "
            f"Project scripts: {'; '.join(script_descriptions)}."
        )
    return (
        f"EIDBench-real script {context['script_id']} for {context['display_name']}. "
        f"It reads tables {', '.join(script['reads'])} and writes {', '.join(script['writes'])}."
    )


def build_prismadv_config(
    prismadv_inputs: dict[str, Any],
    *,
    model_name: str,
    temperature: float,
    use_async: bool,
    use_dataflow: bool,
    correlation_detection: bool = True,
    with_assumptions: bool = True,
) -> PrismaDVConfig:
    return PrismaDVConfig.from_dict(
        {
            "model": {
                "use_async": use_async,
                "use_dataflow": use_dataflow,
                "correlation_detection": correlation_detection,
                "with_assumptions": with_assumptions,
                "downstream_task_description": build_downstream_task_description(prismadv_inputs),
            },
            "llm": {
                "model_name": model_name,
                "temperature": temperature,
                "max_tokens": None,
                "seed": None,
            },
            "io": {
                "overwrite": False,
            },
        }
    )


def build_table_validation_inputs(
    project_manager: MultiTableProjectManager,
    prismadv_inputs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    context = prismadv_inputs["context"]
    table_names = context["script"]["reads"]
    bundle = MultiTableLoader(project_manager).load_pandas(
        variant=context["variant"],
        corruption_label=context.get("corruption_label"),
        table_names=table_names,
    )
    dq_manager = DeequDataQualityManager()
    spark_session = None
    spark_sessions = {}
    data_samples = {}
    for table_name, table in bundle.tables.items():
        log_progress(f"starting Spark dataframe for table {table_name} ({len(table)} rows)")
        spark_df, spark_session = dq_manager.spark_df_from_pandas_df(
            table,
            spark_session=spark_session,
        )
        spark_sessions[table_name] = spark_session
        data_samples[table_name] = spark_df
        log_progress(f"ready Spark dataframe for table {table_name}")
    return spark_sessions, data_samples, spark_session


def close_spark_session(spark_session: Any) -> None:
    if spark_session is None:
        return
    try:
        spark_session.sparkContext._gateway.close()
    finally:
        spark_session.stop()


def existing_output_for_config(
    output_dir: Path,
    prismadv_config: PrismaDVConfig,
    *,
    filename_prefix: str = DEFAULT_FILENAME_PREFIX,
) -> Path | None:
    for path in sorted(output_dir.glob(f"{filename_prefix}--*.yaml")):
        raw_data = load_yaml(path)
        existing_config = PrismaDVConfig.from_dict(raw_data["prismadv_config"])
        if existing_config == prismadv_config:
            return path
    return None


def generate_constraints(
    *,
    project_root: Path,
    example_id: str,
    script_id: str,
    input_path: Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.6,
    use_async: bool = True,
    use_dataflow: bool = True,
    correlation_detection: bool = True,
    with_assumptions: bool = True,
    filename_prefix: str = DEFAULT_FILENAME_PREFIX,
    overwrite: bool = False,
    generator_cls=MultiTablePrismaLangChainDV,
) -> Path:
    project_root = project_root.resolve()
    input_path = input_path or default_prismadv_input_path(example_id, script_id, project_root)
    log_progress(f"loading input artifact: {input_path}")
    prismadv_inputs = load_yaml(input_path)
    project_manager = MultiTableProjectManager(project_root=project_root, example_id=example_id)
    output_dir = utils.constraints_dir(example_id, script_id, project_root, create=True)
    prismadv_config = build_prismadv_config(
        prismadv_inputs,
        model_name=model_name,
        temperature=temperature,
        use_async=use_async,
        use_dataflow=use_dataflow,
        correlation_detection=correlation_detection,
        with_assumptions=with_assumptions,
    )
    log_progress(
        f"configured generation: example={example_id}, script_id={script_id}, "
        f"model={model_name}, dataflow={use_dataflow}, correlation={correlation_detection}, "
        f"with_assumptions={with_assumptions}, async={use_async}"
    )

    existing_output = existing_output_for_config(
        output_dir, prismadv_config, filename_prefix=filename_prefix
    )
    if existing_output is not None and not overwrite:
        print(f"Constraints file for this EIDBench-real PrismaDV config already exists. Skipping: {existing_output}")
        return existing_output

    spark_session = None
    input_variables = {
        **prismadv_inputs,
        "cfg_use_dataflow": prismadv_config.model.use_dataflow,
    }
    try:
        log_progress("building Spark validation inputs")
        spark_sessions, data_samples, spark_session = build_table_validation_inputs(
            project_manager,
            prismadv_inputs,
        )
        input_variables["spark_sessions"] = spark_sessions
        input_variables["data_samples"] = data_samples
        log_progress("Spark validation inputs ready")

        log_progress("initializing EIDBench-real PrismaDV generator")
        generator = generator_cls.from_config(prismadv_config)
        log_progress("starting constraint inference")
        if prismadv_config.model.use_async:
            constraints_with_sources, data_flow_locations, cost_summary = asyncio.run(
                generator.ainvoke(input_variables)
            )
        else:
            constraints_with_sources, data_flow_locations, cost_summary = generator.invoke(input_variables)
        log_progress("constraint inference complete")
    finally:
        log_progress("closing Spark session")
        close_spark_session(spark_session)

    output_path = output_dir / make_output_filename(model_name, filename_prefix)
    result = {
        "prismadv_config": prismadv_config.to_dict(),
        "input_artifact": str(input_path),
        "constraints": constraints_with_sources.to_dict()["constraints"],
        "column_data_flow_locations": {
            ser_column_group_key(column_group): locations.to_dict()
            for column_group, locations in data_flow_locations.items()
        },
        "cost_summary": cost_summary,
    }
    log_progress(f"writing constraints: {output_path}")
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result, handle, sort_keys=False)
    log_progress("done")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = utils.resolve_project_root(args.project_root)
    output_path = generate_constraints(
        project_root=project_root,
        example_id=args.example_id,
        script_id=args.script_id,
        input_path=args.input_path,
        model_name=args.model_name,
        temperature=args.temperature,
        use_async=not args.sync,
        use_dataflow=not args.no_dataflow,
        correlation_detection=not args.no_correlation,
        with_assumptions=not args.no_assumption,
        filename_prefix=args.filename_prefix,
        overwrite=args.overwrite,
    )
    print(
        yaml.safe_dump(
            {
                "example_id": args.example_id,
                "script_id": args.script_id,
                "output_path": str(output_path),
            },
            sort_keys=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
