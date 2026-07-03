from __future__ import annotations

import logging
from typing import Any, FrozenSet

import oyaml as yaml

from prismadv.data_models.config import PrismaDVConfig
from prismadv.data_models.constraints_v2 import (
    AssumptionEntry,
    CodeEntry,
    ColumnConstraintsWithSources,
    ConstraintsWithSources,
    SourceLocation,
    SourceLocations,
)
from prismadv.dq_manager import DeequDataQualityManager
from prismadv.ir_translator.deequ_constraints.function_manager import DeequFunctionManager
from prismadv.llm.langchain.models.prismadv.costs import arun_with_cost, run_with_cost
from prismadv.llm.langchain.models.prismadv.runtime import Runtime
from prismadv.llm.langchain.models.prismadv.utils import batched_gather
from prismadv.llm.langchain.models.prismadv_multi_table.chains import build_chains
from prismadv.llm.langchain.models.prismadv_multi_table.tasks import MultiTablePrismaDVTasks
from prismadv.llm.langchain.models.prismadv_multi_table.types import (
    ColumnGroupRef,
    ColumnRef,
    MultiFileCodeContext,
)
from prismadv.llm_backend.entry import get_langchain_model


class MultiTablePrismaLangChainDV:
    """Table-aware PrismaDV orchestrator for EIDBench-real.

    This orchestrator intentionally handles table-local constraints only. It
    does not generate or validate cross-table constraints such as foreign keys.
    """

    def __init__(
        self,
        model_name: str,
        downstream_task_description: str,
        use_column_correlation_detection: bool = False,
        with_assumptions: bool = True,
        temperature: float = 0.6,
        logger: logging.Logger | None = None,
    ):
        if not model_name:
            raise ValueError("Model name is required.")
        self.model_name = model_name
        self.model = get_langchain_model(model_name, temperature=temperature)
        self.downstream_task_description = downstream_task_description
        self.use_column_correlation_detection = use_column_correlation_detection
        self.with_assumptions = with_assumptions
        self.logger = logger
        self._chains = build_chains(self.model)
        self.runtime = Runtime(self._chains, logger=self.logger)
        self.max_concurrent_calls = 15 if "gemini" in model_name.lower() else None

    @staticmethod
    def _log(message: str) -> None:
        print(f"[real-etl-prismadv] {message}", flush=True)

    @classmethod
    def from_config(cls, config: PrismaDVConfig, logger: logging.Logger | None = None):
        return cls(
            model_name=config.llm.model_name,
            use_column_correlation_detection=config.model.correlation_detection,
            with_assumptions=config.model.with_assumptions,
            downstream_task_description=config.model.downstream_task_description,
            temperature=config.llm.temperature,
            logger=logger,
        )

    def invoke(self, input_variables: dict[str, Any]):
        cost_summary: dict[str, Any] = {}
        prepared = self._prepare_input_variables(input_variables)

        self._log("LLM stage: table column access detection")
        column_refs = run_with_cost(
            "table_column_access_detection",
            lambda: self._detect_table_columns(prepared),
            cost_summary,
        )
        self._log(
            "LLM stage complete: table column access detection "
            f"({len(column_refs)} columns)"
        )
        if prepared["cfg_use_dataflow"]:
            self._log("LLM stage: table column data flow inspection")
            data_flow_locations = run_with_cost(
                "table_column_data_flow_inspection",
                lambda: self._inspect_data_flow(column_refs, prepared),
                cost_summary,
            )
            self._log("LLM stage complete: table column data flow inspection")
        else:
            data_flow_locations = {column_ref.key: SourceLocations() for column_ref in column_refs}

        if self.with_assumptions:
            self._log("LLM stage: table column assumption generation")
            assumptions = run_with_cost(
                "table_column_assumption_generation",
                lambda: self._generate_assumptions(column_refs, data_flow_locations, prepared),
                cost_summary,
            )
            self._log("LLM stage complete: table column assumption generation")
            self._log("LLM stage: table column code generation and validation")
            codes = run_with_cost(
                "table_column_code_generation",
                lambda: self._generate_code(column_refs, assumptions, prepared),
                cost_summary,
            )
            self._log("LLM stage complete: table column code generation and validation")
        else:
            assumptions = {column_ref.key: [] for column_ref in column_refs}
            self._log("LLM stage: table column direct code generation and validation")
            codes = run_with_cost(
                "table_column_direct_code_generation",
                lambda: self._generate_direct_code(column_refs, data_flow_locations, prepared),
                cost_summary,
            )
            self._log("LLM stage complete: table column direct code generation and validation")

        group_assumptions: dict[FrozenSet[str], list[AssumptionEntry]] = {}
        group_codes: dict[FrozenSet[str], list[CodeEntry]] = {}
        column_groups: list[ColumnGroupRef] = []
        group_data_flow_locations: dict[FrozenSet[str], SourceLocations] = {}
        if self.use_column_correlation_detection:
            self._log("LLM stage: same-table column group discovery")
            column_groups = run_with_cost(
                "table_column_group_discovery",
                lambda: self._discover_column_groups(column_refs, prepared),
                cost_summary,
            )
            self._log(
                "LLM stage complete: same-table column group discovery "
                f"({len(column_groups)} groups)"
            )
            if column_groups:
                if prepared["cfg_use_dataflow"]:
                    self._log("LLM stage: column group data flow inspection")
                    group_data_flow_locations = run_with_cost(
                        "table_column_group_data_flow_inspection",
                        lambda: self._inspect_group_data_flow(
                            column_groups, data_flow_locations, prepared
                        ),
                        cost_summary,
                    )
                    self._log("LLM stage complete: column group data flow inspection")
                else:
                    group_data_flow_locations = {
                        group.key: SourceLocations() for group in column_groups
                    }
                if self.with_assumptions:
                    self._log("LLM stage: column group assumption generation")
                    group_assumptions = run_with_cost(
                        "table_column_group_assumption_generation",
                        lambda: self._generate_group_assumptions(
                            column_groups, group_data_flow_locations, prepared
                        ),
                        cost_summary,
                    )
                    self._log("LLM stage complete: column group assumption generation")
                    self._log("LLM stage: column group code generation and validation")
                    group_codes = run_with_cost(
                        "table_column_group_code_generation",
                        lambda: self._generate_group_code(
                            column_groups, group_assumptions, prepared
                        ),
                        cost_summary,
                    )
                    self._log("LLM stage complete: column group code generation and validation")
                else:
                    group_assumptions = {group.key: [] for group in column_groups}
                    self._log("LLM stage: column group direct code generation and validation")
                    group_codes = run_with_cost(
                        "table_column_group_direct_code_generation",
                        lambda: self._generate_group_direct_code(
                            column_groups, group_data_flow_locations, prepared
                        ),
                        cost_summary,
                    )
                    self._log("LLM stage complete: column group direct code generation and validation")

        constraints = self.constraints_from_table_outputs(
            column_refs,
            assumptions,
            codes,
            column_groups=column_groups,
            group_assumptions=group_assumptions,
            group_codes=group_codes,
        )
        merged_data_flow_locations = {**data_flow_locations, **group_data_flow_locations}
        return constraints, merged_data_flow_locations, cost_summary

    async def ainvoke(self, input_variables: dict[str, Any]):
        cost_summary: dict[str, Any] = {}
        prepared = self._prepare_input_variables(input_variables)

        self._log("LLM stage: table column access detection")
        column_refs = await arun_with_cost(
            "table_column_access_detection",
            lambda: self._adetect_table_columns(prepared),
            cost_summary,
        )
        self._log(
            "LLM stage complete: table column access detection "
            f"({len(column_refs)} columns)"
        )
        if prepared["cfg_use_dataflow"]:
            self._log("LLM stage: table column data flow inspection")
            data_flow_locations = await arun_with_cost(
                "table_column_data_flow_inspection",
                lambda: self._ainspect_data_flow(column_refs, prepared),
                cost_summary,
            )
            self._log("LLM stage complete: table column data flow inspection")
        else:
            data_flow_locations = {column_ref.key: SourceLocations() for column_ref in column_refs}

        if self.with_assumptions:
            self._log("LLM stage: table column assumption generation")
            assumptions = await arun_with_cost(
                "table_column_assumption_generation",
                lambda: self._agenerate_assumptions(column_refs, data_flow_locations, prepared),
                cost_summary,
            )
            self._log("LLM stage complete: table column assumption generation")
            self._log("LLM stage: table column code generation and validation")
            codes = await arun_with_cost(
                "table_column_code_generation",
                lambda: self._agenerate_code(column_refs, assumptions, prepared),
                cost_summary,
            )
            self._log("LLM stage complete: table column code generation and validation")
        else:
            assumptions = {column_ref.key: [] for column_ref in column_refs}
            self._log("LLM stage: table column direct code generation and validation")
            codes = await arun_with_cost(
                "table_column_direct_code_generation",
                lambda: self._agenerate_direct_code(column_refs, data_flow_locations, prepared),
                cost_summary,
            )
            self._log("LLM stage complete: table column direct code generation and validation")

        group_assumptions: dict[FrozenSet[str], list[AssumptionEntry]] = {}
        group_codes: dict[FrozenSet[str], list[CodeEntry]] = {}
        column_groups: list[ColumnGroupRef] = []
        group_data_flow_locations: dict[FrozenSet[str], SourceLocations] = {}
        if self.use_column_correlation_detection:
            self._log("LLM stage: same-table column group discovery")
            column_groups = await arun_with_cost(
                "table_column_group_discovery",
                lambda: self._adiscover_column_groups(column_refs, prepared),
                cost_summary,
            )
            self._log(
                "LLM stage complete: same-table column group discovery "
                f"({len(column_groups)} groups)"
            )
            if column_groups:
                if prepared["cfg_use_dataflow"]:
                    self._log("LLM stage: column group data flow inspection")
                    group_data_flow_locations = await arun_with_cost(
                        "table_column_group_data_flow_inspection",
                        lambda: self._ainspect_group_data_flow(
                            column_groups, data_flow_locations, prepared
                        ),
                        cost_summary,
                    )
                    self._log("LLM stage complete: column group data flow inspection")
                else:
                    group_data_flow_locations = {
                        group.key: SourceLocations() for group in column_groups
                    }
                if self.with_assumptions:
                    self._log("LLM stage: column group assumption generation")
                    group_assumptions = await arun_with_cost(
                        "table_column_group_assumption_generation",
                        lambda: self._agenerate_group_assumptions(
                            column_groups, group_data_flow_locations, prepared
                        ),
                        cost_summary,
                    )
                    self._log("LLM stage complete: column group assumption generation")
                    self._log("LLM stage: column group code generation and validation")
                    group_codes = await arun_with_cost(
                        "table_column_group_code_generation",
                        lambda: self._agenerate_group_code(
                            column_groups, group_assumptions, prepared
                        ),
                        cost_summary,
                    )
                    self._log("LLM stage complete: column group code generation and validation")
                else:
                    group_assumptions = {group.key: [] for group in column_groups}
                    self._log("LLM stage: column group direct code generation and validation")
                    group_codes = await arun_with_cost(
                        "table_column_group_direct_code_generation",
                        lambda: self._agenerate_group_direct_code(
                            column_groups, group_data_flow_locations, prepared
                        ),
                        cost_summary,
                    )
                    self._log("LLM stage complete: column group direct code generation and validation")

        constraints = self.constraints_from_table_outputs(
            column_refs,
            assumptions,
            codes,
            column_groups=column_groups,
            group_assumptions=group_assumptions,
            group_codes=group_codes,
        )
        merged_data_flow_locations = {**data_flow_locations, **group_data_flow_locations}
        return constraints, merged_data_flow_locations, cost_summary

    def _prepare_input_variables(self, input_variables: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(input_variables)
        prepared["code_context_obj"] = MultiFileCodeContext.from_prismadv_inputs(prepared)
        prepared["code_context_with_lines"] = prepared["code_context_obj"].with_line_numbers()
        prepared["tables_desc"] = yaml.safe_dump(prepared["table_profiles"], sort_keys=False)
        prepared["downstream_task_description"] = self.downstream_task_description
        prepared.setdefault("cfg_use_dataflow", True)
        prepared.setdefault("spark_sessions", {})
        prepared.setdefault("data_samples", {})
        return prepared

    def _detect_table_columns(self, vars: dict[str, Any]) -> list[ColumnRef]:
        provided_refs = vars.get("column_refs_to_consider")
        if provided_refs:
            return self._normalize_column_refs(provided_refs, vars["table_profiles"])

        payload = {
            "tables_desc": vars["tables_desc"],
            "code_context": vars["code_context_with_lines"],
            "downstream_task_description": vars["downstream_task_description"],
        }
        response = self.runtime.run_task(MultiTablePrismaDVTasks.TABLE_COLUMN_ACCESS_DETECTION, payload)
        return self._column_refs_from_response(response, vars["table_profiles"])

    async def _adetect_table_columns(self, vars: dict[str, Any]) -> list[ColumnRef]:
        provided_refs = vars.get("column_refs_to_consider")
        if provided_refs:
            return self._normalize_column_refs(provided_refs, vars["table_profiles"])

        payload = {
            "tables_desc": vars["tables_desc"],
            "code_context": vars["code_context_with_lines"],
            "downstream_task_description": vars["downstream_task_description"],
        }
        response = await self.runtime.arun_task(MultiTablePrismaDVTasks.TABLE_COLUMN_ACCESS_DETECTION, payload)
        return self._column_refs_from_response(response, vars["table_profiles"])

    def _inspect_data_flow(
        self,
        column_refs: list[ColumnRef],
        vars: dict[str, Any],
    ) -> dict[str, SourceLocations]:
        locations = {}
        total = len(column_refs)
        for index, column_ref in enumerate(column_refs, start=1):
            self._log(f"data flow {index}/{total}: {column_ref.key}")
            response = self.runtime.run_task(
                MultiTablePrismaDVTasks.TABLE_COLUMN_DATA_FLOW_INSPECTION,
                self._data_flow_payload(column_ref, vars),
            )
            locations[column_ref.key] = self._source_locations_from_response(response)
        return locations

    async def _ainspect_data_flow(
        self,
        column_refs: list[ColumnRef],
        vars: dict[str, Any],
    ) -> dict[str, SourceLocations]:
        total = len(column_refs)

        async def inspect_one(index: int, column_ref: ColumnRef) -> tuple[str, SourceLocations]:
            self._log(f"data flow {index}/{total}: {column_ref.key}")
            response = await self.runtime.arun_task(
                MultiTablePrismaDVTasks.TABLE_COLUMN_DATA_FLOW_INSPECTION,
                self._data_flow_payload(column_ref, vars),
            )
            return column_ref.key, self._source_locations_from_response(response)

        results = await batched_gather(
            [inspect_one(index, column_ref) for index, column_ref in enumerate(column_refs, start=1)],
            max_concurrent=self.max_concurrent_calls,
        )
        return dict(results)

    @staticmethod
    def _data_flow_payload(column_ref: ColumnRef, vars: dict[str, Any]) -> dict[str, Any]:
        return {
            "target_table": column_ref.table,
            "target_column": column_ref.column,
            "script_reads": ", ".join(vars["context"]["script"]["reads"]),
            "code_context": vars["code_context_with_lines"],
        }

    def _generate_assumptions(
        self,
        column_refs: list[ColumnRef],
        data_flow_locations: dict[str, SourceLocations],
        vars: dict[str, Any],
    ) -> dict[str, list[AssumptionEntry]]:
        assumptions = {}
        total = len(column_refs)
        for index, column_ref in enumerate(column_refs, start=1):
            self._log(f"assumptions {index}/{total}: {column_ref.key}")
            response = self.runtime.run_task(
                MultiTablePrismaDVTasks.TABLE_COLUMN_ASSUMPTION_GENERATION,
                self._assumption_payload(column_ref, data_flow_locations[column_ref.key], vars),
            )
            assumptions[column_ref.key] = [
                AssumptionEntry.from_dict(assumption)
                for assumption in response.get("assumptions", [])
            ]
        return assumptions

    async def _agenerate_assumptions(
        self,
        column_refs: list[ColumnRef],
        data_flow_locations: dict[str, SourceLocations],
        vars: dict[str, Any],
    ) -> dict[str, list[AssumptionEntry]]:
        total = len(column_refs)

        async def generate_one(index: int, column_ref: ColumnRef) -> tuple[str, list[AssumptionEntry]]:
            self._log(f"assumptions {index}/{total}: {column_ref.key}")
            response = await self.runtime.arun_task(
                MultiTablePrismaDVTasks.TABLE_COLUMN_ASSUMPTION_GENERATION,
                self._assumption_payload(column_ref, data_flow_locations[column_ref.key], vars),
            )
            return column_ref.key, [
                AssumptionEntry.from_dict(assumption)
                for assumption in response.get("assumptions", [])
            ]

        results = await batched_gather(
            [generate_one(index, column_ref) for index, column_ref in enumerate(column_refs, start=1)],
            max_concurrent=self.max_concurrent_calls,
        )
        return dict(results)

    @staticmethod
    def _assumption_payload(
        column_ref: ColumnRef,
        source_locations: SourceLocations,
        vars: dict[str, Any],
    ) -> dict[str, Any]:
        table_profile = vars["table_profiles"][column_ref.table]
        column_profile = _profile_for_column(table_profile, column_ref.column)
        focused_code = (
            vars["code_context_obj"].add_highlighted_line_numbers(source_locations.sources)
            if vars.get("cfg_use_dataflow")
            else vars["code_context_with_lines"]
        )
        return {
            "target_table": column_ref.table,
            "target_column": column_ref.column,
            "target_table_desc": yaml.safe_dump(table_profile, sort_keys=False),
            "target_column_desc": yaml.safe_dump(column_profile, sort_keys=False),
            "focused_code": focused_code,
            "downstream_task_description": vars["downstream_task_description"],
        }

    def _generate_code(
        self,
        column_refs: list[ColumnRef],
        assumptions: dict[str, list[AssumptionEntry]],
        vars: dict[str, Any],
    ) -> dict[str, list[CodeEntry]]:
        codes = {}
        total = len(column_refs)
        for index, column_ref in enumerate(column_refs, start=1):
            self._log(f"code generation {index}/{total}: {column_ref.key}")
            response = self.runtime.run_task(
                MultiTablePrismaDVTasks.TABLE_COLUMN_CODE_GENERATION,
                self._code_payload(column_ref, assumptions[column_ref.key], vars),
            )
            codes[column_ref.key] = self._code_entries_from_response(
                response,
                assumptions[column_ref.key],
                column_ref,
                vars,
            )
        return codes

    async def _agenerate_code(
        self,
        column_refs: list[ColumnRef],
        assumptions: dict[str, list[AssumptionEntry]],
        vars: dict[str, Any],
    ) -> dict[str, list[CodeEntry]]:
        total = len(column_refs)

        async def generate_one(index: int, column_ref: ColumnRef) -> tuple[str, list[CodeEntry]]:
            self._log(f"code generation {index}/{total}: {column_ref.key}")
            response = await self.runtime.arun_task(
                MultiTablePrismaDVTasks.TABLE_COLUMN_CODE_GENERATION,
                self._code_payload(column_ref, assumptions[column_ref.key], vars),
            )
            return column_ref.key, self._code_entries_from_response(
                response,
                assumptions[column_ref.key],
                column_ref,
                vars,
            )

        results = await batched_gather(
            [generate_one(index, column_ref) for index, column_ref in enumerate(column_refs, start=1)],
            max_concurrent=self.max_concurrent_calls,
        )
        return dict(results)

    @staticmethod
    def _code_payload(
        column_ref: ColumnRef,
        assumptions: list[AssumptionEntry],
        vars: dict[str, Any],
    ) -> dict[str, Any]:
        function_manager = DeequFunctionManager()
        table_profile = vars["table_profiles"][column_ref.table]
        column_profile = _profile_for_column(table_profile, column_ref.column)
        return {
            "row_level_functions": "\n".join(function_manager.get_constraints(is_row_level=True)),
            "aggregate_level_functions": "\n".join(function_manager.get_constraints(is_row_level=False)),
            "target_table": column_ref.table,
            "target_column": column_ref.column,
            "target_table_desc": yaml.safe_dump(table_profile, sort_keys=False),
            "target_column_desc": yaml.safe_dump(column_profile, sort_keys=False),
            "code_context": vars["code_context_with_lines"],
            "downstream_task_description": vars["downstream_task_description"],
            "assumptions": _assumptions_text(assumptions),
        }

    @staticmethod
    def _code_entries_from_response(
        response: dict[str, Any],
        assumptions: list[AssumptionEntry],
        column_ref: ColumnRef,
        vars: dict[str, Any],
    ) -> list[CodeEntry]:
        raw_codes = response.get("constraint_code", [])
        for raw_code in raw_codes:
            try:
                raw_code["source_assumptions"] = [
                    assumptions[index].uid for index in raw_code.get("linked assumptions", [])
                ]
            except Exception:
                raw_code["source_assumptions"] = []

        codes = [CodeEntry.from_dict(raw_code) for raw_code in raw_codes if "suggestion" in raw_code]
        spark = vars["spark_sessions"].get(column_ref.table)
        data_sample = vars["data_samples"].get(column_ref.table)
        if spark is None or data_sample is None:
            return codes

        MultiTablePrismaLangChainDV._log(f"validating {len(codes)} constraints: {column_ref.key}")
        validation_results = DeequDataQualityManager().validate_constraints_with_reasons(
            spark,
            data_sample,
            [code.suggestion for code in codes],
        )
        for code, (validity, reason_if_invalid) in zip(codes, validation_results):
            code.validity = validity
            code.reason_if_invalid = reason_if_invalid
        return codes

    def _generate_direct_code(
        self,
        column_refs: list[ColumnRef],
        data_flow_locations: dict[str, SourceLocations],
        vars: dict[str, Any],
    ) -> dict[str, list[CodeEntry]]:
        codes: dict[str, list[CodeEntry]] = {}
        total = len(column_refs)
        for index, column_ref in enumerate(column_refs, start=1):
            self._log(f"direct code generation {index}/{total}: {column_ref.key}")
            response = self.runtime.run_task(
                MultiTablePrismaDVTasks.TABLE_COLUMN_DIRECT_CODE_GENERATION,
                self._direct_code_payload(
                    column_ref,
                    data_flow_locations.get(column_ref.key, SourceLocations()),
                    vars,
                ),
            )
            codes[column_ref.key] = self._code_entries_from_response(response, [], column_ref, vars)
        return codes

    async def _agenerate_direct_code(
        self,
        column_refs: list[ColumnRef],
        data_flow_locations: dict[str, SourceLocations],
        vars: dict[str, Any],
    ) -> dict[str, list[CodeEntry]]:
        total = len(column_refs)

        async def generate_one(index: int, column_ref: ColumnRef) -> tuple[str, list[CodeEntry]]:
            self._log(f"direct code generation {index}/{total}: {column_ref.key}")
            response = await self.runtime.arun_task(
                MultiTablePrismaDVTasks.TABLE_COLUMN_DIRECT_CODE_GENERATION,
                self._direct_code_payload(
                    column_ref,
                    data_flow_locations.get(column_ref.key, SourceLocations()),
                    vars,
                ),
            )
            return column_ref.key, self._code_entries_from_response(response, [], column_ref, vars)

        results = await batched_gather(
            [generate_one(index, column_ref) for index, column_ref in enumerate(column_refs, start=1)],
            max_concurrent=self.max_concurrent_calls,
        )
        return dict(results)

    @staticmethod
    def _direct_code_payload(
        column_ref: ColumnRef,
        source_locations: SourceLocations,
        vars: dict[str, Any],
    ) -> dict[str, Any]:
        function_manager = DeequFunctionManager()
        table_profile = vars["table_profiles"][column_ref.table]
        column_profile = _profile_for_column(table_profile, column_ref.column)
        focused_code = (
            vars["code_context_obj"].add_highlighted_line_numbers(source_locations.sources)
            if vars.get("cfg_use_dataflow")
            else vars["code_context_with_lines"]
        )
        return {
            "row_level_functions": "\n".join(function_manager.get_constraints(is_row_level=True)),
            "aggregate_level_functions": "\n".join(function_manager.get_constraints(is_row_level=False)),
            "target_table": column_ref.table,
            "target_column": column_ref.column,
            "target_table_desc": yaml.safe_dump(table_profile, sort_keys=False),
            "target_column_desc": yaml.safe_dump(column_profile, sort_keys=False),
            "focused_code": focused_code,
            "downstream_task_description": vars["downstream_task_description"],
        }

    def _generate_group_direct_code(
        self,
        column_groups: list[ColumnGroupRef],
        group_data_flow_locations: dict[FrozenSet[str], SourceLocations],
        vars: dict[str, Any],
    ) -> dict[FrozenSet[str], list[CodeEntry]]:
        codes: dict[FrozenSet[str], list[CodeEntry]] = {}
        total = len(column_groups)
        for index, column_group in enumerate(column_groups, start=1):
            self._log(f"group direct code generation {index}/{total}: {sorted(column_group.key)}")
            response = self.runtime.run_task(
                MultiTablePrismaDVTasks.TABLE_COLUMN_GROUP_DIRECT_CODE_GENERATION,
                self._group_direct_code_payload(
                    column_group,
                    group_data_flow_locations.get(column_group.key, SourceLocations()),
                    vars,
                ),
            )
            codes[column_group.key] = self._group_code_entries_from_response(
                response, [], column_group, vars
            )
        return codes

    async def _agenerate_group_direct_code(
        self,
        column_groups: list[ColumnGroupRef],
        group_data_flow_locations: dict[FrozenSet[str], SourceLocations],
        vars: dict[str, Any],
    ) -> dict[FrozenSet[str], list[CodeEntry]]:
        total = len(column_groups)

        async def generate_one(
            index: int, column_group: ColumnGroupRef
        ) -> tuple[FrozenSet[str], list[CodeEntry]]:
            self._log(f"group direct code generation {index}/{total}: {sorted(column_group.key)}")
            response = await self.runtime.arun_task(
                MultiTablePrismaDVTasks.TABLE_COLUMN_GROUP_DIRECT_CODE_GENERATION,
                self._group_direct_code_payload(
                    column_group,
                    group_data_flow_locations.get(column_group.key, SourceLocations()),
                    vars,
                ),
            )
            return column_group.key, self._group_code_entries_from_response(
                response, [], column_group, vars
            )

        results = await batched_gather(
            [generate_one(index, group) for index, group in enumerate(column_groups, start=1)],
            max_concurrent=self.max_concurrent_calls,
        )
        return dict(results)

    @staticmethod
    def _group_direct_code_payload(
        column_group: ColumnGroupRef,
        source_locations: SourceLocations,
        vars: dict[str, Any],
    ) -> dict[str, Any]:
        function_manager = DeequFunctionManager()
        table_profile = vars["table_profiles"][column_group.table]
        group_profile = {
            "table_name": column_group.table,
            "columns": [_profile_for_column(table_profile, column) for column in sorted(column_group.columns)],
        }
        focused_code = (
            vars["code_context_obj"].add_highlighted_line_numbers(source_locations.sources)
            if vars.get("cfg_use_dataflow")
            else vars["code_context_with_lines"]
        )
        return {
            "multi_column_functions": "\n".join(
                function_manager.get_constraints(can_be_used_for_multiple_columns=True)
            ),
            "target_table": column_group.table,
            "target_columns": ", ".join(sorted(column_group.columns)),
            "target_columns_desc": yaml.safe_dump(group_profile, sort_keys=False),
            "focused_code": focused_code,
            "downstream_task_description": vars["downstream_task_description"],
        }

    def _discover_column_groups(
        self,
        column_refs: list[ColumnRef],
        vars: dict[str, Any],
    ) -> list[ColumnGroupRef]:
        if not column_refs:
            return []
        response = self.runtime.run_task(
            MultiTablePrismaDVTasks.TABLE_COLUMN_GROUP_DISCOVERY,
            self._group_discovery_payload(column_refs, vars),
        )
        return self._column_groups_from_response(response, column_refs, vars["table_profiles"])

    async def _adiscover_column_groups(
        self,
        column_refs: list[ColumnRef],
        vars: dict[str, Any],
    ) -> list[ColumnGroupRef]:
        if not column_refs:
            return []
        response = await self.runtime.arun_task(
            MultiTablePrismaDVTasks.TABLE_COLUMN_GROUP_DISCOVERY,
            self._group_discovery_payload(column_refs, vars),
        )
        return self._column_groups_from_response(response, column_refs, vars["table_profiles"])

    @staticmethod
    def _group_discovery_payload(
        column_refs: list[ColumnRef],
        vars: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "accessed_columns": ", ".join(column_ref.key for column_ref in column_refs),
            "tables_desc": vars["tables_desc"],
            "code_context": vars["code_context_with_lines"],
            "downstream_task_description": vars["downstream_task_description"],
        }

    def _inspect_group_data_flow(
        self,
        column_groups: list[ColumnGroupRef],
        single_locations: dict[str, SourceLocations],
        vars: dict[str, Any],
    ) -> dict[FrozenSet[str], SourceLocations]:
        locations: dict[FrozenSet[str], SourceLocations] = {}
        total = len(column_groups)
        for index, column_group in enumerate(column_groups, start=1):
            self._log(f"group data flow {index}/{total}: {sorted(column_group.key)}")
            response = self.runtime.run_task(
                MultiTablePrismaDVTasks.TABLE_COLUMN_GROUP_DATA_FLOW_INSPECTION,
                self._group_data_flow_payload(column_group, single_locations, vars),
            )
            locations[column_group.key] = self._source_locations_from_response(response)
        return locations

    async def _ainspect_group_data_flow(
        self,
        column_groups: list[ColumnGroupRef],
        single_locations: dict[str, SourceLocations],
        vars: dict[str, Any],
    ) -> dict[FrozenSet[str], SourceLocations]:
        total = len(column_groups)

        async def inspect_one(
            index: int, column_group: ColumnGroupRef
        ) -> tuple[FrozenSet[str], SourceLocations]:
            self._log(f"group data flow {index}/{total}: {sorted(column_group.key)}")
            response = await self.runtime.arun_task(
                MultiTablePrismaDVTasks.TABLE_COLUMN_GROUP_DATA_FLOW_INSPECTION,
                self._group_data_flow_payload(column_group, single_locations, vars),
            )
            return column_group.key, self._source_locations_from_response(response)

        results = await batched_gather(
            [inspect_one(index, group) for index, group in enumerate(column_groups, start=1)],
            max_concurrent=self.max_concurrent_calls,
        )
        return dict(results)

    @staticmethod
    def _group_data_flow_payload(
        column_group: ColumnGroupRef,
        single_locations: dict[str, SourceLocations],
        vars: dict[str, Any],
    ) -> dict[str, Any]:
        related = [
            single_locations[column_ref.key]
            for column_ref in column_group.to_column_refs()
            if column_ref.key in single_locations
        ]
        seed = SourceLocations.merge_sources(*related) if related else SourceLocations()
        focused_code = (
            vars["code_context_obj"].add_highlighted_line_numbers(seed.sources)
            if vars.get("cfg_use_dataflow")
            else vars["code_context_with_lines"]
        )
        return {
            "target_table": column_group.table,
            "target_columns": ", ".join(sorted(column_group.columns)),
            "script_reads": ", ".join(vars["context"]["script"]["reads"]),
            "code_context": focused_code,
        }

    def _generate_group_assumptions(
        self,
        column_groups: list[ColumnGroupRef],
        group_data_flow_locations: dict[FrozenSet[str], SourceLocations],
        vars: dict[str, Any],
    ) -> dict[FrozenSet[str], list[AssumptionEntry]]:
        assumptions: dict[FrozenSet[str], list[AssumptionEntry]] = {}
        total = len(column_groups)
        for index, column_group in enumerate(column_groups, start=1):
            self._log(f"group assumptions {index}/{total}: {sorted(column_group.key)}")
            response = self.runtime.run_task(
                MultiTablePrismaDVTasks.TABLE_COLUMN_GROUP_ASSUMPTION_GENERATION,
                self._group_assumption_payload(
                    column_group, group_data_flow_locations.get(column_group.key, SourceLocations()), vars
                ),
            )
            assumptions[column_group.key] = [
                AssumptionEntry.from_dict(assumption)
                for assumption in response.get("assumptions", [])
            ]
        return assumptions

    async def _agenerate_group_assumptions(
        self,
        column_groups: list[ColumnGroupRef],
        group_data_flow_locations: dict[FrozenSet[str], SourceLocations],
        vars: dict[str, Any],
    ) -> dict[FrozenSet[str], list[AssumptionEntry]]:
        total = len(column_groups)

        async def generate_one(
            index: int, column_group: ColumnGroupRef
        ) -> tuple[FrozenSet[str], list[AssumptionEntry]]:
            self._log(f"group assumptions {index}/{total}: {sorted(column_group.key)}")
            response = await self.runtime.arun_task(
                MultiTablePrismaDVTasks.TABLE_COLUMN_GROUP_ASSUMPTION_GENERATION,
                self._group_assumption_payload(
                    column_group,
                    group_data_flow_locations.get(column_group.key, SourceLocations()),
                    vars,
                ),
            )
            return column_group.key, [
                AssumptionEntry.from_dict(assumption)
                for assumption in response.get("assumptions", [])
            ]

        results = await batched_gather(
            [generate_one(index, group) for index, group in enumerate(column_groups, start=1)],
            max_concurrent=self.max_concurrent_calls,
        )
        return dict(results)

    @staticmethod
    def _group_assumption_payload(
        column_group: ColumnGroupRef,
        source_locations: SourceLocations,
        vars: dict[str, Any],
    ) -> dict[str, Any]:
        table_profile = vars["table_profiles"][column_group.table]
        group_profile = {
            "table_name": column_group.table,
            "columns": [_profile_for_column(table_profile, column) for column in sorted(column_group.columns)],
        }
        focused_code = (
            vars["code_context_obj"].add_highlighted_line_numbers(source_locations.sources)
            if vars.get("cfg_use_dataflow")
            else vars["code_context_with_lines"]
        )
        return {
            "target_table": column_group.table,
            "target_columns": ", ".join(sorted(column_group.columns)),
            "target_columns_desc": yaml.safe_dump(group_profile, sort_keys=False),
            "focused_code": focused_code,
            "downstream_task_description": vars["downstream_task_description"],
        }

    def _generate_group_code(
        self,
        column_groups: list[ColumnGroupRef],
        group_assumptions: dict[FrozenSet[str], list[AssumptionEntry]],
        vars: dict[str, Any],
    ) -> dict[FrozenSet[str], list[CodeEntry]]:
        codes: dict[FrozenSet[str], list[CodeEntry]] = {}
        total = len(column_groups)
        for index, column_group in enumerate(column_groups, start=1):
            self._log(f"group code generation {index}/{total}: {sorted(column_group.key)}")
            response = self.runtime.run_task(
                MultiTablePrismaDVTasks.TABLE_COLUMN_GROUP_CODE_GENERATION,
                self._group_code_payload(
                    column_group, group_assumptions.get(column_group.key, []), vars
                ),
            )
            codes[column_group.key] = self._group_code_entries_from_response(
                response,
                group_assumptions.get(column_group.key, []),
                column_group,
                vars,
            )
        return codes

    async def _agenerate_group_code(
        self,
        column_groups: list[ColumnGroupRef],
        group_assumptions: dict[FrozenSet[str], list[AssumptionEntry]],
        vars: dict[str, Any],
    ) -> dict[FrozenSet[str], list[CodeEntry]]:
        total = len(column_groups)

        async def generate_one(
            index: int, column_group: ColumnGroupRef
        ) -> tuple[FrozenSet[str], list[CodeEntry]]:
            self._log(f"group code generation {index}/{total}: {sorted(column_group.key)}")
            response = await self.runtime.arun_task(
                MultiTablePrismaDVTasks.TABLE_COLUMN_GROUP_CODE_GENERATION,
                self._group_code_payload(
                    column_group, group_assumptions.get(column_group.key, []), vars
                ),
            )
            return column_group.key, self._group_code_entries_from_response(
                response,
                group_assumptions.get(column_group.key, []),
                column_group,
                vars,
            )

        results = await batched_gather(
            [generate_one(index, group) for index, group in enumerate(column_groups, start=1)],
            max_concurrent=self.max_concurrent_calls,
        )
        return dict(results)

    @staticmethod
    def _group_code_payload(
        column_group: ColumnGroupRef,
        assumptions: list[AssumptionEntry],
        vars: dict[str, Any],
    ) -> dict[str, Any]:
        function_manager = DeequFunctionManager()
        table_profile = vars["table_profiles"][column_group.table]
        group_profile = {
            "table_name": column_group.table,
            "columns": [_profile_for_column(table_profile, column) for column in sorted(column_group.columns)],
        }
        return {
            "multi_column_functions": "\n".join(
                function_manager.get_constraints(can_be_used_for_multiple_columns=True)
            ),
            "target_table": column_group.table,
            "target_columns": ", ".join(sorted(column_group.columns)),
            "target_columns_desc": yaml.safe_dump(group_profile, sort_keys=False),
            "code_context": vars["code_context_with_lines"],
            "downstream_task_description": vars["downstream_task_description"],
            "assumptions": _assumptions_text(assumptions),
        }

    @staticmethod
    def _group_code_entries_from_response(
        response: dict[str, Any],
        assumptions: list[AssumptionEntry],
        column_group: ColumnGroupRef,
        vars: dict[str, Any],
    ) -> list[CodeEntry]:
        raw_codes = response.get("constraint_code", [])
        for raw_code in raw_codes:
            try:
                raw_code["source_assumptions"] = [
                    assumptions[index].uid for index in raw_code.get("linked assumptions", [])
                ]
            except Exception:
                raw_code["source_assumptions"] = []

        codes = [CodeEntry.from_dict(raw_code) for raw_code in raw_codes if "suggestion" in raw_code]
        spark = vars["spark_sessions"].get(column_group.table)
        data_sample = vars["data_samples"].get(column_group.table)
        if spark is None or data_sample is None:
            return codes

        MultiTablePrismaLangChainDV._log(
            f"validating {len(codes)} group constraints: {column_group.table}.{sorted(column_group.columns)}"
        )
        validation_results = DeequDataQualityManager().validate_constraints_with_reasons(
            spark,
            data_sample,
            [code.suggestion for code in codes],
        )
        for code, (validity, reason_if_invalid) in zip(codes, validation_results):
            code.validity = validity
            code.reason_if_invalid = reason_if_invalid
        return codes

    @staticmethod
    def constraints_from_table_outputs(
        column_refs: list[ColumnRef],
        assumptions: dict[str, list[AssumptionEntry]],
        codes: dict[str, list[CodeEntry]],
        *,
        column_groups: list[ColumnGroupRef] | None = None,
        group_assumptions: dict[FrozenSet[str], list[AssumptionEntry]] | None = None,
        group_codes: dict[FrozenSet[str], list[CodeEntry]] | None = None,
    ) -> ConstraintsWithSources:
        constraints = ConstraintsWithSources()
        for column_ref in column_refs:
            constraints.data_map[column_ref.key] = ColumnConstraintsWithSources(
                assumptions=assumptions.get(column_ref.key, []),
                code=codes.get(column_ref.key, []),
                table_name=column_ref.table,
                column_group=column_ref.column,
            )
        for column_group in column_groups or []:
            group_key = column_group.key
            constraints.data_map[group_key] = ColumnConstraintsWithSources(
                assumptions=(group_assumptions or {}).get(group_key, []),
                code=(group_codes or {}).get(group_key, []),
                table_name=column_group.table,
                column_group=column_group.column_group,
            )
        return constraints

    @staticmethod
    def _normalize_column_refs(
        provided_refs: list[ColumnRef | dict[str, Any] | str],
        table_profiles: dict[str, Any],
    ) -> list[ColumnRef]:
        refs = []
        for ref in provided_refs:
            if isinstance(ref, ColumnRef):
                column_ref = ref
            elif isinstance(ref, str):
                column_ref = ColumnRef.from_key(ref)
            else:
                column_ref = ColumnRef.from_dict(ref)
            if _column_exists(table_profiles, column_ref):
                refs.append(column_ref)
        return sorted(dict.fromkeys(refs))

    @staticmethod
    def _column_refs_from_response(
        response: Any,
        table_profiles: dict[str, Any],
    ) -> list[ColumnRef]:
        raw_columns = response.get("columns", response) if isinstance(response, dict) else response
        refs = []
        for raw_column in raw_columns or []:
            if isinstance(raw_column, str):
                column_ref = ColumnRef.from_key(raw_column)
            else:
                column_ref = ColumnRef.from_dict(raw_column)
            if _column_exists(table_profiles, column_ref):
                refs.append(column_ref)
        return sorted(dict.fromkeys(refs))

    @staticmethod
    def _column_groups_from_response(
        response: Any,
        column_refs: list[ColumnRef],
        table_profiles: dict[str, Any],
    ) -> list[ColumnGroupRef]:
        raw_groups = response.get("groups", response) if isinstance(response, dict) else response
        accessed_by_table: dict[str, set[str]] = {}
        for column_ref in column_refs:
            accessed_by_table.setdefault(column_ref.table, set()).add(column_ref.column)

        groups: list[ColumnGroupRef] = []
        seen_keys: set[FrozenSet[str]] = set()
        for raw_group in raw_groups or []:
            if not isinstance(raw_group, dict):
                continue
            try:
                candidate = ColumnGroupRef.from_dict(raw_group)
            except Exception:
                continue
            if candidate.table not in accessed_by_table:
                continue
            allowed = accessed_by_table[candidate.table]
            filtered = frozenset(column for column in candidate.columns if column in allowed)
            if len(filtered) < 2:
                continue
            existing_columns = {column["name"] for column in table_profiles.get(candidate.table, {}).get("columns", [])}
            if not filtered.issubset(existing_columns):
                continue
            normalized = ColumnGroupRef(
                table=candidate.table,
                columns=filtered,
                correlation_type=candidate.correlation_type,
            )
            if normalized.key in seen_keys:
                continue
            seen_keys.add(normalized.key)
            groups.append(normalized)
        return groups

    @staticmethod
    def _source_locations_from_response(response: dict[str, Any]) -> SourceLocations:
        sources = []
        for source in response.get("sources", []):
            if not source.get("file"):
                continue
            sources.append(
                SourceLocation(
                    file=source["file"],
                    start_line=int(source["start_line"]),
                    end_line=int(source["end_line"]),
                )
            )
        return SourceLocations(sources)


def _profile_for_column(table_profile: dict[str, Any], column_name: str) -> dict[str, Any]:
    for column_profile in table_profile.get("columns", []):
        if column_profile["name"] == column_name:
            return column_profile
    raise KeyError(f"column <{column_name}> not found in table profile <{table_profile.get('table_name')}>")


def _column_exists(table_profiles: dict[str, Any], column_ref: ColumnRef) -> bool:
    table_profile = table_profiles.get(column_ref.table)
    if table_profile is None:
        return False
    return any(column["name"] == column_ref.column for column in table_profile.get("columns", []))


def _assumptions_text(assumptions: list[AssumptionEntry]) -> str:
    return "\n".join(f"Assumption {index}: {assumption.to_string()}" for index, assumption in enumerate(assumptions))
