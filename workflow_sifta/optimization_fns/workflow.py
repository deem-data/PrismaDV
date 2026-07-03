"""DSPy optimization workflow class."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import dspy
from pyspark.sql import SparkSession

from prismadv.data_models import ValidationResults
from prismadv.data_models.constraints_v2 import ConstraintsWithSources
from prismadv.data_models.trajectory import DVTrajectoryColumnGroupSuite
from prismadv.llm.dspy.models.column_wise_module import (
    ConstraintGenerationModule,
)
from prismadv.llm.dspy.models.sampler.trajectory_retrieval import aggregate_trajectories
from prismadv.llm.dspy.models.sampler.types import TrajectoryKey
from prismadv.project_manager.manager.base import ProjectManager
from workflow_sifta.optimization_fns.column_discovery import discover_columns_and_groups
from workflow_sifta.optimization_fns.constraint_generation import (
    generate_constraints_for_column as _generate_constraints_for_column,
    generate_constraints_for_column_from_example as _generate_constraints_for_column_from_example,
    combine_constraints as _combine_constraints,
)
from workflow_sifta.optimization_fns.dataset_preparation import (
    prepare_single_column_training_dataset as _prepare_single_column_training_dataset,
)
from workflow_sifta.optimization_fns.trajectory_creation import (
    create_trajectories_from_constraints as _create_trajectories_from_constraints,
)
from workflow_sifta.optimization_fns.validation import (
    validate_constraints_on_training_data as _validate_constraints_on_training_data,
    validate_constraints_on_test_data as _validate_constraints_on_test_data,
)


class OptimizationWorkflow:
    """Workflow class for DSPy prompt optimization."""

    # All available datasets in the project
    ALL_DATASETS = ["students", "hr_analytics", "sleep_health", "IPL_win_prediction", "imdb"]

    def __init__(
        self,
        train_dataset_subtasks: Dict[str, List[str]],
        train_processed_data_label_list: List[str],
        val_dataset_subtasks: Optional[Dict[str, List[str]]] = None,
        val_processed_data_label_list: Optional[List[str]] = None,
        test_dataset_subtasks: Optional[Dict[str, List[str]]] = None,
        test_processed_data_label_list: Optional[List[str]] = None,
        downstream_task_type: Union[str, Dict[str, str]] = "general",
        project_root: Union[Path, str] = None,
    ):
        """
        Initialize the optimization workflow.

        Args:
            train_dataset_subtasks: Dict mapping dataset_name to list of subtask names for training
                e.g., {"students": ["general_task"], "adult": ["general_task"]}
            train_processed_data_label_list: List of processed data labels for training
                e.g., ["1", "2", "3", "4", "5"]
            val_dataset_subtasks: Optional dict mapping dataset_name to list of subtask names for validation
                If None, uses train_dataset_subtasks
            val_processed_data_label_list: Optional list of processed data labels for validation
                If None, uses train_processed_data_label_list
            test_dataset_subtasks: Optional dict mapping dataset_name to list of subtask names for test
                If None, uses train_dataset_subtasks
            test_processed_data_label_list: Optional list of processed data labels for test
                If None, uses train_processed_data_label_list
            downstream_task_type: Task type (string for all datasets or dict per-dataset)
                e.g., "general" or {"students": "general", "adult": "general"}
            project_root: Optional project root path. If None, uses default from utils.
        """
        self.train_dataset_subtasks = train_dataset_subtasks
        self.train_processed_data_label_list = train_processed_data_label_list

        # Validation configuration (defaults to training if not specified)
        self.val_dataset_subtasks = val_dataset_subtasks if val_dataset_subtasks is not None else train_dataset_subtasks
        self.val_processed_data_label_list = val_processed_data_label_list if val_processed_data_label_list is not None else train_processed_data_label_list

        # Test configuration (defaults to training if not specified)
        self.test_dataset_subtasks = test_dataset_subtasks if test_dataset_subtasks is not None else train_dataset_subtasks
        self.test_processed_data_label_list = test_processed_data_label_list if test_processed_data_label_list is not None else train_processed_data_label_list

        self.downstream_task_type = downstream_task_type
        self.project_root = project_root

        # Create ProjectManager instances for each unique dataset across train/val/test
        all_datasets = set(train_dataset_subtasks.keys())
        all_datasets.update(self.val_dataset_subtasks.keys())
        all_datasets.update(self.test_dataset_subtasks.keys())

        self.project_managers = {}
        for dataset_name in all_datasets:
            # Get task type for this dataset
            if isinstance(downstream_task_type, dict):
                task_type = downstream_task_type.get(dataset_name, "general")
            else:
                task_type = downstream_task_type

            if project_root:
                self.project_managers[dataset_name] = ProjectManager(
                    project_root=project_root,
                    dataset_name=dataset_name,
                    downstream_task_type=task_type,
                )
            else:
                self.project_managers[dataset_name] = ProjectManager(
                    dataset_name=dataset_name,
                    downstream_task_type=task_type,
                )

        # For backward compatibility, set primary project_manager to first dataset
        if train_dataset_subtasks:
            first_dataset = list(train_dataset_subtasks.keys())[0]
            self.project_manager = self.project_managers[first_dataset]

        # Single DQ manager instance with its own Spark session management
        from prismadv.dq_manager import DeequDataQualityManager
        self._dq_manager = DeequDataQualityManager()

    def __del__(self):
        """Cleanup Spark sessions when workflow is destroyed."""
        try:
            self.cleanup_spark_sessions()
        except Exception:
            pass  # Ignore errors during cleanup

    def get_available_script_names_for_subtask(
        self,
        dataset_name: str,
        subtask_name: str,
    ) -> List[str]:
        """
        Get all available script names for a given dataset and subtask.

        Args:
            dataset_name: Name of the dataset (e.g., "students")
            subtask_name: Name of the subtask (e.g., "general_task")

        Returns:
            List of script names (without .py extension), sorted alphabetically.
        """
        pm = self.project_managers[dataset_name]
        script_paths = pm.get_available_script_path_list_for_subtask(subtask_name)
        return [p.stem for p in script_paths]

    def discover_columns_and_groups(
        self,
        dataset_name: str,
        subtask_name: str,
        script_name: str,
        processed_data_label: str = "0",
    ) -> Dict[str, List]:
        """
        Discover accessed columns and correlated groups using ColumnDiscoveryModule.

        Args:
            dataset_name: Name of the dataset
            subtask_name: Name of the subtask
            script_name: Name of the script (without .py extension)
            processed_data_label: Label for processed data (default: "0" for training data)

        Returns:
            Dict with keys:
                - "columns_to_consider": List of column names
                - "correlated_groups": List of correlated group dicts
                - "column_desc_dict": Dict mapping column names to descriptions
                - "source_code": Source code string
                - "downstream_task_description": Task description string
        """
        pm = self.project_managers[dataset_name]
        return discover_columns_and_groups(
            project_manager=pm,
            subtask_name=subtask_name,
            script_name=script_name,
            processed_data_label=processed_data_label,
        )

    def prepare_single_column_training_dataset(
        self,
        script_name_list: Union[List[str], Dict[str, List[str]]],
        processed_data_label: Union[str, Dict[str, str]] = "0",
        new_processed_data_label_list: Union[
            List[str], Dict[str, List[str]], None
        ] = None,
    ) -> List[dspy.Example]:
        """
        Prepare training dataset for single-column constraint generation.

        Args:
            script_name_list: List of script names (applied to all datasets) or
                dict mapping dataset_name to list of script names
            processed_data_label: Label for processed data (default: "0" for training data).
                Can be a string (applied to all) or dict mapping dataset_name to label.
            new_processed_data_label_list: List of new processed data labels to check safety for.
                Can be a list (applied to all datasets) or dict mapping dataset_name to list of labels.
                If None, no safety information is recorded.

        Returns:
            List of dicts, each containing training example information
        """
        return _prepare_single_column_training_dataset(
            project_manager=self.project_manager,  # Only used for backward compat, not actually used
            dataset_subtasks=self.train_dataset_subtasks,
            script_name_list=script_name_list,
            processed_data_label=processed_data_label,
            new_processed_data_label_list=new_processed_data_label_list,
        )

    def prepare_single_column_validation_dataset(
        self,
        script_name_list: Union[List[str], Dict[str, List[str]]],
        processed_data_label: Union[str, Dict[str, str]] = "0",
        new_processed_data_label_list: Union[
            List[str], Dict[str, List[str]], None
        ] = None,
    ) -> List[dspy.Example]:
        """
        Prepare validation dataset for single-column constraint generation.
        Uses val_dataset_subtasks and val_processed_data_label_list configured at init.

        Args:
            script_name_list: List of script names (applied to all datasets) or
                dict mapping dataset_name to list of script names
            processed_data_label: Label for processed data (default: "0" for training data).
                Can be a string (applied to all) or dict mapping dataset_name to label.
            new_processed_data_label_list: List of new processed data labels to check safety for.
                Can be a list (applied to all datasets) or dict mapping dataset_name to list of labels.
                If None, uses val_processed_data_label_list from init.

        Returns:
            List of dicts, each containing validation example information
        """
        # Use val_processed_data_label_list if new_processed_data_label_list not provided
        if new_processed_data_label_list is None:
            new_processed_data_label_list = self.val_processed_data_label_list

        return _prepare_single_column_training_dataset(
            project_manager=self.project_manager,  # Only used for backward compat, not actually used
            dataset_subtasks=self.val_dataset_subtasks,
            script_name_list=script_name_list,
            processed_data_label=processed_data_label,
            new_processed_data_label_list=new_processed_data_label_list,
        )

    def generate_constraints_for_column(
        self,
        constraint_module: ConstraintGenerationModule,
        column_name: str,
        column_desc_dict: Dict[str, Dict],
        source_code: str,
        downstream_task_description: str,
        sink_variable: str = "",
    ) -> Dict:
        """
        Generate constraints for a single column using ConstraintGenerationModule.

        Args:
            constraint_module: ConstraintGenerationModule instance (optimizable)
            column_name: Name of the column to generate constraints for
            column_desc_dict: Dict mapping column names to descriptions
            source_code: Source code string
            downstream_task_description: Task description string
            sink_variable: Sink variable name (default: "")

        Returns:
            Dict with keys:
                - "assumptions": List of AssumptionEntry objects
                - "code": List of CodeEntry objects
        """
        return _generate_constraints_for_column(
            constraint_module=constraint_module,
            column_name=column_name,
            column_desc_dict=column_desc_dict,
            source_code=source_code,
            downstream_task_description=downstream_task_description,
            sink_variable=sink_variable,
        )

    def generate_constraints_for_column_from_example(
        self,
        constraint_module: ConstraintGenerationModule,
        training_example: Dict,
    ) -> Dict:
        """
        Generate constraints for a single column using a training example dict.

        Args:
            constraint_module: ConstraintGenerationModule instance (optimizable)
            training_example: Dict from prepare_single_column_training_dataset()

        Returns:
            Dict with keys:
                - "assumptions": List of AssumptionEntry objects
                - "code": List of CodeEntry objects
        """
        return _generate_constraints_for_column_from_example(
            constraint_module=constraint_module,
            training_example=training_example,
        )

    def combine_constraints(
        self,
        single_column_results: Optional[Dict[str, Dict]] = None,
        multi_column_results: Optional[Dict[frozenset, Dict]] = None,
    ) -> ConstraintsWithSources:
        """
        Combine single-column and multi-column constraint results into ConstraintsWithSources.

        Args:
            single_column_results: Optional dict mapping column names to result dicts.
                If None, single-column constraints are not included.
            multi_column_results: Optional dict mapping column groups (frozenset) to result dicts.
                If None, multi-column constraints are not included.

        Returns:
            ConstraintsWithSources object containing all constraints

        Raises:
            ValueError: If both single_column_results and multi_column_results are None
        """
        return _combine_constraints(
            single_column_results=single_column_results,
            multi_column_results=multi_column_results,
        )

    def cleanup_spark_sessions(self):
        """
        Stop the Spark session. Call this when done with the workflow
        to free up resources.
        """
        self._dq_manager.cleanup_spark_session()

    def _reset_spark_session(self):
        """
        Reset the Spark session. Called when a session becomes corrupted.
        """
        self._dq_manager.reset_spark_session()

    def _get_or_create_spark_session(self) -> SparkSession:
        """
        Get or create the Spark session from the DQ manager. The session is reused across
        all validations for efficiency.

        Returns:
            SparkSession instance
        """
        return self._dq_manager.get_or_create_spark_session()

    def validate_constraints_on_training_data(
        self,
        dataset_name: str,
        subtask_name: str,
        constraints_with_sources: ConstraintsWithSources,
        processed_data_label: str = "0",
    ) -> ConstraintsWithSources:
        """
        Validate constraints on training data to set validity flags for a specific dataset/subtask.
        Uses a global Spark session that is reused across all validations for efficiency.

        Args:
            dataset_name: Name of the dataset to validate for
            subtask_name: Name of the subtask to validate for
            constraints_with_sources: ConstraintsWithSources to validate
            processed_data_label: Label for processed training data (default: "0")

        Returns:
            ConstraintsWithSources with validity flags set
        """
        pm = self.project_managers[dataset_name]

        # Get or create global Spark session (reused across all validations)
        spark_session = self._get_or_create_spark_session()

        # Call validation function with global session and reset callback
        constraints_with_sources = _validate_constraints_on_training_data(
            project_manager=pm,
            subtask_name=subtask_name,
            processed_data_label=processed_data_label,
            constraints_with_sources=constraints_with_sources,
            spark_session=spark_session,
            session_reset_callback=self._reset_spark_session,
        )

        return constraints_with_sources

    def validate_constraints_on_test_data(
        self,
        dataset_name: str,
        subtask_name: str,
        constraints_with_sources: ConstraintsWithSources,
        processed_data_label: str,
        clean: bool = False,
    ) -> ValidationResults:
        """
        Validate constraints on test data for a specific dataset/subtask.
        Uses a global Spark session that is reused across all validations for efficiency.

        Args:
            dataset_name: Name of the dataset to validate for
            subtask_name: Name of the subtask to validate for
            constraints_with_sources: ConstraintsWithSources to validate
            processed_data_label: Label for processed test data
            clean: Whether to use clean test data (default: False for corrupted data)

        Returns:
            ValidationResults object containing validation results
        """
        pm = self.project_managers[dataset_name]

        # Get or create global Spark session (reused across all validations)
        spark_session = self._get_or_create_spark_session()

        # Call validation function with global session and reset callback
        validation_results = _validate_constraints_on_test_data(
            project_manager=pm,
            subtask_name=subtask_name,
            processed_data_label=processed_data_label,
            constraints_with_sources=constraints_with_sources,
            spark_session=spark_session,
            session_reset_callback=self._reset_spark_session,
            clean=clean,
        )

        return validation_results

    def create_trajectories_from_constraints(
        self,
        script_name_list: Union[List[str], Dict[str, List[str]]],
        processed_data_label: Union[str, Dict[str, str]],
        llm_name: str,
        constraints_with_sources: ConstraintsWithSources,
        validation_results_dict: Dict[str, Dict[str, ValidationResults]],
        clean: bool = False,
    ) -> Dict[str, Dict[str, List]]:
        """
        Create trajectories from constraints and validation results across all datasets.

        Args:
            script_name_list: List of script names (applied to all datasets) or
                dict mapping dataset_name to list of script names
            processed_data_label: Label for processed data.
                Can be a string (applied to all) or dict mapping dataset_name to label.
            llm_name: Name of the LLM used
            constraints_with_sources: ConstraintsWithSources object
            validation_results_dict: Dict with structure {dataset_name: {script_name: ValidationResults}}
            clean: Whether using clean test data (default: False)

        Returns:
            Nested dict with structure: {dataset_name: {script_name: [trajectories]}}
        """
        all_trajectories = {}

        # Create trajectories for each dataset/subtask/script combination
        for dataset_name, subtask_list in self.train_dataset_subtasks.items():
            pm = self.project_managers[dataset_name]

            # Get script names for this dataset
            if isinstance(script_name_list, dict):
                scripts_to_use = script_name_list.get(dataset_name, [])
            else:
                scripts_to_use = script_name_list

            # Get processed_data_label for this dataset
            if isinstance(processed_data_label, dict):
                label = processed_data_label.get(dataset_name, "1")
            else:
                label = processed_data_label

            if dataset_name not in all_trajectories:
                all_trajectories[dataset_name] = {}

            for subtask_name in subtask_list:
                for script_name in scripts_to_use:
                    if script_name not in validation_results_dict.get(dataset_name, {}):
                        continue

                    validation_results = validation_results_dict[dataset_name][
                        script_name
                    ]

                    trajectories = _create_trajectories_from_constraints(
                        project_manager=pm,
                        dataset_name=dataset_name,
                        subtask_name=subtask_name,
                        script_name=script_name,
                        processed_data_label=label,
                        llm_name=llm_name,
                        constraints_with_sources=constraints_with_sources,
                        validation_results=validation_results,
                        clean=clean,
                    )

                    if script_name not in all_trajectories[dataset_name]:
                        all_trajectories[dataset_name][script_name] = []
                    all_trajectories[dataset_name][script_name].extend(trajectories)

        return all_trajectories

    def aggregate_trajectories_for_sampling(
        self,
        trajectories_dict: Dict[str, Dict[str, List]],
        llm_name: str,
    ) -> Dict[TrajectoryKey, List[DVTrajectoryColumnGroupSuite]]:
        """
        Aggregate trajectories from multiple datasets into the format needed for sampling.

        Args:
            trajectories_dict: Nested dict with structure {dataset_name: {script_name: [trajectories]}}
            llm_name: Name of the LLM

        Returns:
            Dict mapping TrajectoryKey to list of DVTrajectoryColumnGroupSuite objects
        """
        # Convert to format expected by aggregate_trajectories: {subtask_name: {script_name: [trajectories]}}
        all_trajectories = {}

        for dataset_name, script_dict in trajectories_dict.items():
            subtask_list = self.train_dataset_subtasks[dataset_name]

            for subtask_name in subtask_list:
                if subtask_name not in all_trajectories:
                    all_trajectories[subtask_name] = {}

                for script_name, trajectories in script_dict.items():
                    if script_name not in all_trajectories[subtask_name]:
                        all_trajectories[subtask_name][script_name] = []
                    all_trajectories[subtask_name][script_name].extend(trajectories)

        # Aggregate trajectories into suites
        aggregated_trajectories = aggregate_trajectories(all_trajectories)

        return aggregated_trajectories

    def evaluate_f1_on_test_set(
        self,
        module: ConstraintGenerationModule,
        test_script_name_list: List[str],
        num_threads: int,
    ) -> Dict:
        """
        Evaluate a module on the test set using F1 score.

        This method:
        1. For each test script, discovers columns and generates constraints
        2. Validates constraints on training data (to set validity flags)
        3. Validates constraints on corrupted test data for each processed_data_label
        4. Reads ground truth (is_safe) from execution results
        5. Calculates precision, recall, and F1 score

        Args:
            module: The ConstraintGenerationModule to evaluate
            test_script_name_list: List of test script names
            num_threads: Number of threads for per-column constraint generation

        Returns:
            Dict with keys:
                - "f1": Overall F1 score
                - "precision": Overall precision
                - "recall": Overall recall
                - "tp", "fp", "tn", "fn": Confusion matrix counts
                - "per_script_results": Per-script detailed results
                - "total_constraints": Total constraints generated
        """
        import oyaml as yaml

        per_script_results = []
        all_tp, all_fp, all_tn, all_fn = 0, 0, 0, 0
        total_constraints = 0
        total_non_compilable = 0

        # Count total scripts for progress
        total_scripts = sum(
            len(subtask_list) * len(test_script_name_list)
            for subtask_list in self.test_dataset_subtasks.values()
        )
        script_counter = 0

        print(f"[F1 Eval] Starting evaluation on {total_scripts} script(s) x {len(self.test_processed_data_label_list)} label(s)")

        for dataset_name, subtask_list in self.test_dataset_subtasks.items():
            pm = self.project_managers.get(dataset_name)
            if pm is None:
                # Create project manager for test dataset if not exists
                pm = ProjectManager(
                    project_root=self.project_root,
                    dataset_name=dataset_name,
                ) if self.project_root else ProjectManager(dataset_name=dataset_name)

            for subtask_name in subtask_list:
                for script_name in test_script_name_list:
                    script_counter += 1
                    print(f"\n[F1 Eval] Processing script {script_counter}/{total_scripts}: {dataset_name}/{subtask_name}/{script_name}")
                    try:
                        # Step 1: Discover columns and groups
                        print(f"  [Step 1/5] Discovering columns...")
                        discovery_result = discover_columns_and_groups(
                            project_manager=pm,
                            subtask_name=subtask_name,
                            script_name=script_name,
                            processed_data_label="0",
                        )

                        columns_to_consider = discovery_result["columns_to_consider"]
                        column_desc_dict = discovery_result["column_desc_dict"]
                        source_code = discovery_result["source_code"]
                        downstream_task_description = discovery_result["downstream_task_description"]
                        print(f"  [Step 1/5] Found {len(columns_to_consider)} columns: {columns_to_consider}")

                        # Step 2: Generate constraints for each column
                        print(f"  [Step 2/5] Generating constraints for {len(columns_to_consider)} columns...")
                        single_column_results = {}
                        if num_threads <= 1:
                            for col_idx, column_name in enumerate(columns_to_consider, 1):
                                try:
                                    print(
                                        f"    Generating for column {col_idx}/{len(columns_to_consider)}: {column_name}...",
                                        end=" ",
                                        flush=True,
                                    )
                                    result = _generate_constraints_for_column(
                                        constraint_module=module,
                                        column_name=column_name,
                                        column_desc_dict=column_desc_dict,
                                        source_code=source_code,
                                        downstream_task_description=downstream_task_description,
                                    )
                                    single_column_results[column_name] = result
                                    print("done")
                                except Exception as e:
                                    print(f"failed ({e})")
                                    # Skip columns that fail
                                    continue
                        else:
                            from concurrent.futures import ThreadPoolExecutor, as_completed

                            def _run_column(col_name: str):
                                result = _generate_constraints_for_column(
                                    constraint_module=module,
                                    column_name=col_name,
                                    column_desc_dict=column_desc_dict,
                                    source_code=source_code,
                                    downstream_task_description=downstream_task_description,
                                )
                                return col_name, result

                            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                                futures = {
                                    executor.submit(_run_column, col_name): col_name
                                    for col_name in columns_to_consider
                                }
                                for future in as_completed(futures):
                                    column_name = futures[future]
                                    try:
                                        col_name, result = future.result()
                                        single_column_results[col_name] = result
                                    except Exception as e:
                                        print(f"    Column generation failed: {column_name} ({e})")

                        if not single_column_results:
                            print(f"  [Warning] No constraints generated for this script, skipping...")
                            continue

                        # Step 3: Combine constraints
                        print(f"  [Step 3/5] Combining constraints from {len(single_column_results)} columns...")
                        constraints_with_sources = _combine_constraints(
                            single_column_results=single_column_results
                        )

                        # Step 4: Validate on training data to set validity flags
                        print(f"  [Step 4/5] Validating on training data...")
                        spark_session = self._get_or_create_spark_session()
                        constraints_with_sources = _validate_constraints_on_training_data(
                            project_manager=pm,
                            subtask_name=subtask_name,
                            processed_data_label="0",
                            constraints_with_sources=constraints_with_sources,
                            spark_session=spark_session,
                            session_reset_callback=self._reset_spark_session,
                        )

                        # Step 5: For each test label, validate and compare with ground truth
                        print(f"  [Step 5/5] Validating on {len(self.test_processed_data_label_list)} test labels...")
                        label_progress = 0
                        for processed_data_label in self.test_processed_data_label_list:
                            label_progress += 1
                            try:
                                # Validate on corrupted test data
                                validation_results = _validate_constraints_on_test_data(
                                    project_manager=pm,
                                    subtask_name=subtask_name,
                                    processed_data_label=processed_data_label,
                                    constraints_with_sources=constraints_with_sources,
                                    spark_session=spark_session,
                                    session_reset_callback=self._reset_spark_session,
                                    clean=False,
                                )

                                # Get constraint counts
                                num_passed_warning, num_failed_warning, num_failed_error, num_passed_error, num_non_compilable = \
                                    validation_results.check_result()
                                num_constraints = num_passed_warning + num_failed_warning + num_failed_error + num_passed_error
                                total_constraints += num_constraints
                                total_non_compilable += num_non_compilable

                                # Prediction: safe if no error-level constraints failed
                                predicted_as_safe = (num_failed_error == 0)

                                # Read ground truth from execution results
                                exec_path = pm.get_execution_output_validation_path(
                                    subtask_name, processed_data_label, script_name
                                ) / "basic_metrics_evaluation.json"

                                with open(exec_path, "r") as f:
                                    exec_results = yaml.load(f, Loader=yaml.FullLoader)
                                is_safe = exec_results.get('corrupted_data_is_safe', True)

                                # Update confusion matrix
                                # Positive class = "unsafe/error" (standard for anomaly detection)
                                # TP = correctly detected error, TN = correctly accepted safe data
                                # FP = false alarm, FN = missed error (BAD!)
                                if not is_safe and not predicted_as_safe:
                                    all_tp += 1  # TP: error detected correctly
                                elif not is_safe and predicted_as_safe:
                                    all_fn += 1  # FN: error missed (BAD!)
                                elif is_safe and not predicted_as_safe:
                                    all_fp += 1  # FP: false alarm
                                else:  # is_safe and predicted_as_safe
                                    all_tn += 1  # TN: safe data accepted correctly

                                per_script_results.append({
                                    "dataset_name": dataset_name,
                                    "subtask_name": subtask_name,
                                    "script_name": script_name,
                                    "processed_data_label": processed_data_label,
                                    "num_constraints": num_constraints,
                                    "num_non_compilable": num_non_compilable,
                                    "num_failed_error": num_failed_error,
                                    "predicted_as_safe": predicted_as_safe,
                                    "is_safe": is_safe,
                                })

                            except FileNotFoundError:
                                # Skip if execution results not found
                                continue
                            except Exception as e:
                                # Skip on other errors
                                continue

                        # Print progress after each script
                        print(f"  [Script Summary] Labels processed: {label_progress}, Running totals: TP={all_tp}, FP={all_fp}, TN={all_tn}, FN={all_fn}")

                    except Exception as e:
                        print(f"  [Error] Script failed: {e}")
                        # Skip scripts that fail
                        continue

        # Print final summary before calculating metrics
        print(f"\n[F1 Eval] Evaluation complete!")
        print(f"[F1 Eval] Final confusion matrix: TP={all_tp}, FP={all_fp}, TN={all_tn}, FN={all_fn}")
        print(f"[F1 Eval] Total constraints validated: {total_constraints}, Non-compilable: {total_non_compilable}")

        # Calculate metrics
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else float('nan')
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else float('nan')

        import math
        if math.isfinite(precision) and math.isfinite(recall) and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float('nan')

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": all_tp,
            "fp": all_fp,
            "tn": all_tn,
            "fn": all_fn,
            "total_constraints": total_constraints,
            "total_non_compilable": total_non_compilable,
            "per_script_results": per_script_results,
        }
