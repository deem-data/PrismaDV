"""Trajectory manager for handling DSPy trajectory operations."""

from pathlib import Path
from typing import List

import oyaml as yaml

from prismadv.data_models import ValidationResults, ConstraintsWithSources
from prismadv.data_models.trajectory import DVTrajectory
from prismadv.loader import FileLoader


class TrajectoryManager:
    """Manages DSPy trajectory loading and formatting.
    
    This class handles loading constraint files, validation results,
    and formatting them into DSPy trajectory objects.
    """

    def __init__(self, path_manager, metadata_manager, script_manager, data_manager,
                 constraint_manager, execution_manager):
        """Initialize the trajectory manager.
        
        Args:
            path_manager: PathManager instance for accessing directory paths.
            metadata_manager: MetadataManager instance for accessing metadata.
            script_manager: ScriptManager instance for accessing scripts.
            data_manager: DataManager instance for accessing data paths.
            constraint_manager: ConstraintManager instance for accessing constraint paths.
            execution_manager: ExecutionManager instance for accessing execution paths.
        """
        self._path_manager = path_manager
        self._metadata_manager = metadata_manager
        self._script_manager = script_manager
        self._data_manager = data_manager
        self._constraint_manager = constraint_manager
        self._execution_manager = execution_manager

    def get_dspy_trajectory(self, subtask_name: str, processed_data_label: str, script_name: str,
                            llm_name: str = None, dspy_prefix: str = None) -> List[DVTrajectory]:
        """Get DSPy trajectories for a specific script and processed data label.
        
        Args:
            subtask_name: Name of the subtask.
            processed_data_label: Label identifying the processed data version.
            script_name: Name of the script.
            llm_name: Optional LLM name to filter trajectories.
            dspy_prefix: Optional prefix for DSPy files (default: "prisma_dspy_dv").
            
        Returns:
            List of DVTrajectory objects.
            
        Raises:
            ValueError: If processed_data_label is 0 or no suggestion files found.
        """
        if int(processed_data_label) == 0:
            raise ValueError("Processed data label must be greater than 0 for DSPy trajectory retrieval.")

        clean = False
        new_test_data_path = self._data_manager.get_new_test_data_path(
            subtask_name, processed_data_label, clean=clean
        )
        constraints_path = self._constraint_manager.get_constraints_path(
            subtask_name, processed_data_label, script_name
        )
        validation_results_dir = self._constraint_manager.get_constraints_validation_path(
            subtask_name, processed_data_label, script_name
        )

        dspy_prefix = dspy_prefix or "prisma_dspy_dv"
        suggestion_file_list = list(constraints_path.glob(f"{dspy_prefix}--*.yaml"))
        if llm_name is not None:
            suggestion_file_list = [file for file in suggestion_file_list if f"{llm_name}" in file.name]
        if not suggestion_file_list:
            raise ValueError(
                f"No suggestion file found in <{constraints_path}> matching llm_name <{llm_name}>."
            )

        trajectories = []
        for suggestion_file_path in suggestion_file_list:
            if llm_name is None:
                llm_name = suggestion_file_path.stem.split("--")[1]

            validation_results_path = (validation_results_dir /
                                       f"validation_results_on_corrupted_test_data__{suggestion_file_path.stem}.yaml")
            validation_results = ValidationResults.from_yaml(validation_results_path)

            exec_path = (self._execution_manager.get_execution_output_validation_path(
                subtask_name, processed_data_label, script_name
            ) / "basic_metrics_evaluation.json")
            try:
                with open(exec_path, "r") as f:
                    exec_results = yaml.load(f, Loader=yaml.FullLoader)
            except FileNotFoundError:
                raise ValueError(f"Execution results file <{exec_path}> does not exist.")

            is_safe = exec_results['clean_data_is_safe'] if clean else exec_results['corrupted_data_is_safe']

            with open(f"{suggestion_file_path}", "r") as f:
                raw_constraint_dict = yaml.load(f, Loader=yaml.FullLoader)
            constraints = ConstraintsWithSources.from_dict(raw_constraint_dict)

            task_name = self._metadata_manager.get_task_name_from_subtask(subtask_name)
            script_path = self._script_manager.get_script_path(task_name, script_name)
            script = FileLoader.load_py_file(script_path)

            trajectory_part = self.format_trajectory_for_dspy(
                self._path_manager.dataset_name, llm_name, script_path, script,
                new_test_data_path, constraints, validation_results, processed_data_label, is_safe
            )
            trajectories += trajectory_part
        return trajectories

    def get_dspy_trajectories(self, subtask_name: str, processed_data_label_list: List[str],
                              script_name_list: List[str], llm_name: str = None,
                              dspy_prefix: str = None) -> List[DVTrajectory]:
        """Get DSPy trajectories for multiple scripts and processed data labels.
        
        Args:
            subtask_name: Name of the subtask.
            processed_data_label_list: List of processed data labels.
            script_name_list: List of script names.
            llm_name: Optional LLM name to filter trajectories.
            dspy_prefix: Optional prefix for DSPy files.
            
        Returns:
            List of DVTrajectory objects.
        """
        trajectories = []
        for processed_data_label in processed_data_label_list:
            for script_name in script_name_list:
                dspy_trajectories = self.get_dspy_trajectory(
                    subtask_name, processed_data_label, script_name,
                    llm_name=llm_name, dspy_prefix=dspy_prefix
                )
                trajectories += dspy_trajectories
        return trajectories

    def format_trajectory_for_dspy(self, dataset_name: str, llm_name: str, script_path: Path,
                                   script, data_path: Path, constraints: ConstraintsWithSources,
                                   validation_results: ValidationResults, processed_data_label: str,
                                   is_safe: bool) -> List[DVTrajectory]:
        """Format constraint and validation data into DSPy trajectory objects.
        
        Args:
            dataset_name: Name of the dataset.
            llm_name: Name of the LLM used.
            script_path: Path to the script file.
            script: Loaded script object.
            data_path: Path to the data file.
            constraints: Constraints with sources.
            validation_results: Validation results.
            processed_data_label: Label identifying the processed data version.
            is_safe: Whether the execution was safe.
            
        Returns:
            List of DVTrajectory objects.
        """
        trajectories = []
        for column_group in constraints.data_map.keys():
            all_assumptions_on_the_column_group = constraints.data_map[column_group].assumptions
            for constraint_code in constraints.data_map[column_group].code:
                if constraint_code.validity == False:
                    continue
                constraint_code_validation_results = validation_results.retrieve_by_column_and_suggestion(
                    column_group, constraint_code.suggestion
                )
                assumptions_sources = [
                    assumption for assumption in all_assumptions_on_the_column_group
                    if assumption.uid in constraint_code.source_assumptions
                ]

                trajectories.append(
                    DVTrajectory(
                        dataset_name=dataset_name,
                        llm_name=llm_name,
                        script_path=script_path,
                        script=script,
                        data_path=data_path,
                        processed_data_label=processed_data_label,
                        column_group=column_group,
                        assumptions=assumptions_sources,
                        constraint=constraint_code,
                        validation_results=constraint_code_validation_results,
                        is_safe=is_safe
                    )
                )
        return trajectories
