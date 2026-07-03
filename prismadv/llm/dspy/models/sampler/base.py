from collections import defaultdict
from typing import Dict, List, Union, Any

import numpy as np

from prismadv.data_models.trajectory import DVTrajectoryColumnGroupSuite
from prismadv.llm.dspy.models.sampler.display import (
    print_result as print_result_func,
    format_result as format_result_func,
    print_constraint_explanation,
    create_score_distribution_plot,
    extract_trajectory_score,
)
from prismadv.llm.dspy.models.sampler.fail_precision import (
    fail_precision as compute_fail_precision_dict,
    compute_overall_fail_precision_for_key,
)
from prismadv.llm.dspy.models.sampler.sampling import (
    select_top_fail_precision,
    build_candidate_list,
    sample_from_candidates,
    filter_suites_to_selected_constraints,
    build_constraint_scores_mapping_from_selected,
)
from prismadv.llm.dspy.models.sampler.trajectory_retrieval import retrieve_trajectories, aggregate_trajectories
# Import types and module functions
from prismadv.llm.dspy.models.sampler.types import (
    TrajectoryKey,
    ColumnGroup,
    ConstraintUID,
    FailPrecisionScore,
    SampledTrajectoryResult,
)
from prismadv.llm.dspy.models.sampler.utils import get_constraint_text, find_suite_by_constraint


class TrajectorySampler:
    """
    A sampler for selecting high-quality constraint trajectories based on fail precision metrics.
    
    The sampler aggregates trajectories by (dataset, subtask, script, llm) and column groups,
    then enables sampling based on fail precision scores with optional temperature-based randomization.
    
    Usage:
        sampler = TrajectorySampler(...)
        sampler.calculate()  # Compute and cache fail precision scores
        results = sampler.sample(...)  # Sample using cached scores
    
    Attributes:
        aggregated_trajectories: Dict mapping trajectory keys to lists of column group suites.
                                Structure: {TrajectoryKey: [DVTrajectoryColumnGroupSuite]}
        _fail_precision_dicts: Cached fail precision scores per trajectory key (set by calculate())
        _overall_fail_precisions: Cached overall fail precision scores per trajectory key (set by calculate())
    """

    def __init__(
            self,
            dataset_subtasks: Dict[str, List[str]],
            processed_data_label_list: Union[List[str], Dict[str, List[str]]],
            llm_name: str,
            dspy_prefix: str,
            downstream_task_type: Union[str, Dict[str, str]] = "general",
            script_name_list: Union[None, List[str], Dict[str, List[str]]] = None,
    ) -> None:
        """
        Initialize the sampler by retrieving and aggregating trajectories.
        
        Args:
            dataset_subtasks: Dict mapping dataset_name to list of subtask names
            processed_data_label_list: List of data labels to process, or dict 
                mapping dataset_name to list of labels
            llm_name: Name of the LLM used for trajectories
            dspy_prefix: Prefix for dspy trajectories
            downstream_task_type: Task type for ProjectManager. Can be a string (applied to 
                all datasets) or dict mapping dataset_name to task type. Default: "general"
            script_name_list: Optional list of script names to filter, or dict mapping 
                dataset_name to list of script names. If None, all available scripts are used.
                Default: None
        """
        # Retrieve trajectories using configuration parameters
        all_trajectories = retrieve_trajectories(
            dataset_subtasks=dataset_subtasks,
            processed_data_label_list=processed_data_label_list,
            llm_name=llm_name,
            dspy_prefix=dspy_prefix,
            downstream_task_type=downstream_task_type,
            script_name_list=script_name_list,
        )
        self.aggregated_trajectories = aggregate_trajectories(all_trajectories)

        # Cache for calculated fail precision scores (populated by calculate() method)
        self._fail_precision_dicts: Dict[
            TrajectoryKey, Dict[ColumnGroup, Dict[Union[str, ConstraintUID], FailPrecisionScore]]] = {}
        self._overall_fail_precisions: Dict[TrajectoryKey, float] = {}
        self._calculated: bool = False

    @classmethod
    def from_aggregated_trajectories(
            cls,
            aggregated_trajectories: Dict[TrajectoryKey, List[DVTrajectoryColumnGroupSuite]],
    ) -> 'TrajectorySampler':
        """
        Create a TrajectorySampler instance from pre-aggregated trajectories.
        
        This is useful when you have trajectories created in-memory (e.g., from constraints
        generated during optimization) rather than loading them from disk.
        
        Args:
            aggregated_trajectories: Dict mapping trajectory keys to lists of column group suites.
                Structure: {TrajectoryKey: [DVTrajectoryColumnGroupSuite]}
                
        Returns:
            TrajectorySampler instance ready to use with calculate() and sample()
            
        Example:
            >>> trajectories = create_trajectories_from_constraints(...)
            >>> aggregated = aggregate_trajectories_for_sampling(...)
            >>> sampler = TrajectorySampler.from_aggregated_trajectories(aggregated)
            >>> sampler.calculate()
            >>> results = sampler.sample(temperature=0.0, num_top_column_groups=3, num_top_constraints=5)
        """
        # Create a minimal instance without calling __init__'s trajectory loading
        instance = cls.__new__(cls)
        instance.aggregated_trajectories = aggregated_trajectories
        instance._fail_precision_dicts = {}
        instance._overall_fail_precisions = {}
        instance._calculated = False
        return instance

    def calculate(self) -> None:
        """
        Calculate and cache fail precision scores for all trajectory keys.

        This method computes fail precision scores for all trajectories and caches them
        for use by the sample() method. Call this before calling sample() to avoid
        recalculating scores on each sampling operation.

        The cached scores are stored in:
        - self._fail_precision_dicts: Per-key fail precision dictionaries
        - self._overall_fail_precisions: Per-key overall fail precision scores
        """
        self._fail_precision_dicts = {}
        self._overall_fail_precisions = {}

        for key, column_group_suites in self.aggregated_trajectories.items():
            # Compute fail precision for all trajectories under this key
            fail_precision_dict = self.fail_precision(column_group_suites)
            self._fail_precision_dicts[key] = fail_precision_dict

            # Compute overall fail precision for this key (aggregate across all column groups)
            overall_fail_precision = compute_overall_fail_precision_for_key(
                column_group_suites,
                fail_precision_dict
            )
            self._overall_fail_precisions[key] = overall_fail_precision

        self._calculated = True

    def sample(
            self,
            temperature: float = 0.0,
            num_top_column_groups: int = 5,
            num_top_constraints: int = 10,
    ) -> Dict[TrajectoryKey, SampledTrajectoryResult]:
        """
        Sample trajectory column group suites based on fail precision scores.
        
        Note: This method uses cached scores from calculate(). Call calculate() first
        to compute and cache the scores. If calculate() has not been called, this method
        will raise a ValueError.

        Args:
            temperature: Sampling temperature in [0, 1].
                - 0.0: Deterministically select top-n by highest fail_precision
                - >0.0: Sample probabilistically using softmax with temperature scaling
            num_top_column_groups: Maximum number of column groups to consider per key
            num_top_constraints: Maximum number of constraints to consider per column group

        Returns:
            Dict mapping trajectory keys to SampledTrajectoryResult objects containing:
            - suites: List of sampled DVTrajectoryColumnGroupSuite objects
            - overall_fail_precision: Overall fail precision score for this key
            - constraint_scores: Dict mapping constraint UIDs to fail precision scores
            
        Raises:
            AssertionError: If temperature is not in [0, 1]
            ValueError: If calculate() has not been called yet
        """
        assert 0.0 <= temperature <= 1.0, "temperature must be in [0, 1]"

        # Check if scores have been calculated
        if not self._calculated:
            raise ValueError(
                "Scores have not been calculated. Call calculate() before calling sample()."
            )

        sampled_results = {}
        max_samples_per_key = max(1, num_top_column_groups * num_top_constraints)

        for key, column_group_suites in self.aggregated_trajectories.items():
            # Use cached fail precision scores
            fail_precision_dict = self._fail_precision_dicts[key]
            overall_fail_precision = self._overall_fail_precisions[key]

            # Select top column groups and constraints based on fail precision
            top_fail_precision = select_top_fail_precision(
                fail_precision_dict,
                num_top_column_groups=num_top_column_groups,
                num_top_constraints=num_top_constraints,
                overall_key="overall",
            )

            # Build candidate list: (suite, score) tuples for selected constraints only
            candidates = build_candidate_list(
                column_group_suites,
                top_fail_precision
            )

            if not candidates:
                sampled_results[key] = SampledTrajectoryResult(
                    suites=[],
                    overall_fail_precision=overall_fail_precision,
                    constraint_scores={}
                )
                continue

            # Sample from candidates based on temperature
            chosen_suites = sample_from_candidates(
                candidates,
                max_samples_per_key,
                temperature
            )

            # Filter suites to only include trajectories with selected constraint_uids
            # and deduplicate suites by column_group
            filtered_suites = filter_suites_to_selected_constraints(
                chosen_suites,
                top_fail_precision
            )

            # Build constraint scores mapping from selected constraints only (no NaN scores)
            constraint_scores = build_constraint_scores_mapping_from_selected(
                filtered_suites,
                top_fail_precision
            )

            sampled_results[key] = SampledTrajectoryResult(
                suites=filtered_suites,
                overall_fail_precision=overall_fail_precision,
                constraint_scores=constraint_scores
            )

        return sampled_results

    def showall(
            self,
            sort_by_score: bool = True
    ) -> Dict[TrajectoryKey, Dict[str, Any]]:
        """
        Get all constraints with their fail precision scores, including NaN scores.
        
        This method returns comprehensive information about all constraints for each
        trajectory key, including those that didn't fail validation (NaN scores).
        
        Args:
            sort_by_score: If True, sort constraints by score (NaN scores last)
            
        Returns:
            Dict mapping trajectory keys to structured result data containing:
            - overall_fail_precision: Overall score for the key
            - column_groups: Dict mapping column groups to their constraints and scores
            
        Raises:
            ValueError: If calculate() has not been called yet
        """
        # Check if scores have been calculated
        if not self._calculated:
            raise ValueError(
                "Scores have not been calculated. Call calculate() before calling showall()."
            )

        results = {}

        for key, column_group_suites in self.aggregated_trajectories.items():
            fail_precision_dict = self._fail_precision_dicts[key]
            overall_fail_precision = self._overall_fail_precisions[key]

            column_groups_data = {}

            for suite in column_group_suites:
                column_group = suite.column_group
                column_group_scores = fail_precision_dict.get(column_group, {})
                column_group_overall = column_group_scores.get("overall", np.nan)

                # Collect all constraints from trajectories (to ensure we get all, including those not in scores)
                all_constraint_uids = set()
                for trajectory in suite.trajectories:
                    try:
                        constraint_uid = trajectory.constraint.uid
                        all_constraint_uids.add(constraint_uid)
                    except (AttributeError, Exception):
                        continue

                # Build constraints data with scores (use NaN if not in fail_precision_dict)
                constraints_data = []
                for constraint_uid in all_constraint_uids:
                    score = column_group_scores.get(constraint_uid, np.nan)
                    constraints_data.append({
                        "constraint_uid": constraint_uid,
                        "score": score,
                        "constraint_text": get_constraint_text(suite, constraint_uid)
                    })

                # Sort by score (NaN last)
                if sort_by_score:
                    constraints_data.sort(
                        key=lambda x: (np.isnan(x["score"]) if isinstance(x["score"], (int, float)) else False,
                                       -x["score"] if not np.isnan(x["score"]) else 0)
                    )

                column_groups_data[str(column_group)] = {
                    "overall_score": column_group_overall,
                    "constraints": constraints_data
                }

            results[key] = {
                "overall_fail_precision": overall_fail_precision,
                "column_groups": column_groups_data
            }

        return results

    def print_result(
            self,
            results: Union[
                Dict[TrajectoryKey, SampledTrajectoryResult],
                Dict[TrajectoryKey, Dict[str, Any]]
            ],
            result_type: str = "auto"
    ) -> None:
        """
        Print results from either sample() or showall() in a formatted way.
        
        Constraints are deduplicated by constraint_uid before printing, so each unique
        constraint is shown only once even if it appears in multiple trajectories with
        different data labels.
        
        Args:
            results: Results from either sample() or showall()
            result_type: Type of results - "sample", "showall", or "auto" (auto-detect)
        """
        print_result_func(results, result_type)

    def format_result(
            self,
            results: Union[
                Dict[TrajectoryKey, SampledTrajectoryResult],
                Dict[TrajectoryKey, Dict[str, Any]]
            ],
            result_type: str = "auto"
    ) -> str:
        """
        Format results from either sample() or showall() as a string.

        Constraints are deduplicated by constraint_uid before formatting, so each unique
        constraint is shown only once even if it appears in multiple trajectories with
        different data labels.

        Args:
            results: Results from either sample() or showall()
            result_type: Type of results - "sample", "showall", or "auto" (auto-detect)

        Returns:
            Formatted string representation of the results
        """
        return format_result_func(results, result_type)

    def explain_constraint(
            self,
            constraint_uid: ConstraintUID,
            print_output: bool = True
    ) -> Dict[str, Any]:
        """
        Explain why a constraint has a particular fail precision score by retrieving
        all trajectories that include this constraint and showing their is_safe values.
        
        This helps understand why constraints have low fail precision scores (e.g., 0.0):
        - If constraint fails on safe data (is_safe=True, prediction=False): False positive
        - If constraint passes on unsafe data (is_safe=False, prediction=True): False negative
        
        Args:
            constraint_uid: The constraint UID to explain
            print_output: If True, print formatted explanation to console
            
        Returns:
            Dict containing:
            - constraint_text: The constraint text
            - fail_precision_score: The fail precision score (if calculated)
            - trajectories: List of trajectory details with is_safe and validation results
            - summary: Summary statistics
        """
        trajectories_info = []
        constraint_text = None
        fail_precision_score = None

        # Search through all aggregated trajectories
        for key, column_group_suites in self.aggregated_trajectories.items():
            dataset_name, subtask_name, script_name, llm_name = key

            for suite in column_group_suites:
                for trajectory in suite.trajectories:
                    try:
                        traj_constraint_uid = trajectory.constraint.uid
                        if traj_constraint_uid == constraint_uid:
                            # Extract constraint text (only need to do this once)
                            if constraint_text is None:
                                constraint_text = getattr(trajectory.constraint, "suggestion", "")

                            # Get validation result
                            try:
                                validation_result = trajectory.validation_results.status
                            except Exception:
                                validation_result = False

                            # Get fail precision score if available
                            if fail_precision_score is None and self._fail_precision_dicts:
                                fail_precision_dict = self._fail_precision_dicts.get(key, {})
                                column_group = suite.column_group
                                column_group_scores = fail_precision_dict.get(column_group, {})
                                fail_precision_score = column_group_scores.get(constraint_uid, np.nan)

                            trajectories_info.append({
                                "trajectory_key": key,
                                "dataset_name": dataset_name,
                                "subtask_name": subtask_name,
                                "script_name": script_name,
                                "llm_name": llm_name,
                                "column_group": suite.column_group,
                                "processed_data_label": trajectory.processed_data_label,
                                "is_safe": trajectory.is_safe,
                                "validation_passed": validation_result,
                                "constraint_failed": not validation_result,
                                # Constraint failed if validation didn't pass
                            })
                    except (AttributeError, Exception):
                        continue

        if not trajectories_info:
            if print_output:
                print(f"No trajectories found for constraint UID: {constraint_uid}")
            return {
                "constraint_uid": constraint_uid,
                "constraint_text": None,
                "fail_precision_score": None,
                "trajectories": [],
                "summary": {}
            }

        # Calculate summary statistics
        total_trajectories = len(trajectories_info)
        safe_count = sum(1 for t in trajectories_info if t["is_safe"])
        unsafe_count = total_trajectories - safe_count

        # Count failures (constraint failed validation)
        failures = [t for t in trajectories_info if t["constraint_failed"]]
        total_failures = len(failures)

        # Count correct failures (failed on unsafe data)
        correct_failures = sum(1 for t in failures if not t["is_safe"])

        # Count false positives (failed on safe data)
        false_positives = sum(1 for t in failures if t["is_safe"])

        # Count false negatives (passed on unsafe data)
        passed_on_unsafe = [t for t in trajectories_info if not t["is_safe"] and not t["constraint_failed"]]
        false_negatives = len(passed_on_unsafe)

        summary = {
            "total_trajectories": total_trajectories,
            "safe_data_count": safe_count,
            "unsafe_data_count": unsafe_count,
            "total_failures": total_failures,
            "correct_failures": correct_failures,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "fail_precision": (correct_failures / total_failures) if total_failures > 0 else np.nan
        }

        result = {
            "constraint_uid": constraint_uid,
            "constraint_text": constraint_text,
            "fail_precision_score": fail_precision_score,
            "trajectories": trajectories_info,
            "summary": summary
        }

        if print_output:
            print_constraint_explanation(result)

        return result

    def plot_score_level_distribution(
            self,
            sampled_suites: Union[
                Dict[TrajectoryKey, List[DVTrajectoryColumnGroupSuite]], Dict[TrajectoryKey, SampledTrajectoryResult]],
            num_bins: int = 10,
    ) -> None:
        """
        Plot the distribution of fail precision scores by constraint level (warning vs error).

        Args:
            sampled_suites: Output from sample(), mapping trajectory keys to suite lists
            num_bins: Number of bins for the histogram (score range is [0, 1])
        """
        # Collect scores organized by constraint level
        scores_by_level = self._collect_scores_by_level(sampled_suites)

        # Extract scores for each level
        error_scores = np.array(scores_by_level.get("error", []), dtype=float)
        warning_scores = np.array(scores_by_level.get("warning", []), dtype=float)

        if len(error_scores) == 0 and len(warning_scores) == 0:
            print("No valid scores found for 'error' or 'warning' constraints.")
            return

        # Create and display the plot
        create_score_distribution_plot(error_scores, warning_scores, num_bins)

    def _collect_scores_by_level(
            self,
            sampled_suites: Union[
                Dict[TrajectoryKey, List[DVTrajectoryColumnGroupSuite]], Dict[TrajectoryKey, SampledTrajectoryResult]]
    ) -> Dict[str, List[float]]:
        """
        Collect fail precision scores organized by constraint level.
        
        Constraints are deduplicated by constraint_uid, so each unique constraint
        contributes only one score to the distribution, even if it appears in multiple
        trajectories with different data labels.
        
        Args:
            sampled_suites: Sampled column group suites to analyze (either old format or new SampledTrajectoryResult format)
            
        Returns:
            Dict mapping constraint levels ('error', 'warning') to lists of scores
        """
        scores_by_level = defaultdict(list)
        # Track seen constraint_uids per level to avoid duplicates
        seen_constraints_by_level = defaultdict(set)

        for key, value in sampled_suites.items():
            # Handle both old format (List[DVTrajectoryColumnGroupSuite]) and new format (SampledTrajectoryResult)
            if isinstance(value, SampledTrajectoryResult):
                suite_list = value.suites
                constraint_scores = value.constraint_scores
            else:
                suite_list = value
                constraint_scores = None

            if not suite_list:
                continue

            # If we have constraint_scores from the result, use them; otherwise recompute
            if constraint_scores is not None:
                fail_precision_dict = None
            else:
                # Recompute fail precision for this trajectory key (backward compatibility)
                column_group_suites = self.aggregated_trajectories[key]
                fail_precision_dict = self.fail_precision(column_group_suites)

            for suite in suite_list:
                # Deduplicate constraints within each suite
                unique_constraints_in_suite = {}

                for trajectory in suite.trajectories:
                    try:
                        constraint_uid = trajectory.constraint.uid

                        # Skip if we've already processed this constraint_uid in this suite
                        if constraint_uid in unique_constraints_in_suite:
                            continue

                        # Get score
                        if constraint_scores is not None:
                            # Use precomputed scores from result
                            score = constraint_scores.get(constraint_uid)
                        else:
                            # Extract score using old method
                            score = extract_trajectory_score(
                                trajectory,
                                fail_precision_dict
                            )

                        if score is None:
                            continue

                        # Get constraint level and validate it
                        level = getattr(trajectory.constraint, "level", None)
                        if level is None:
                            continue

                        level_str = str(level).lower()
                        if level_str not in ("warning", "error"):
                            continue

                        # Track this constraint_uid in this suite to avoid duplicates
                        unique_constraints_in_suite[constraint_uid] = {
                            "score": float(score),
                            "level": level_str
                        }
                    except (AttributeError, Exception):
                        continue

                # Add unique constraints to the scores_by_level, avoiding duplicates across suites
                for constraint_uid, constraint_info in unique_constraints_in_suite.items():
                    level_str = constraint_info["level"]
                    score = constraint_info["score"]

                    # Only add if we haven't seen this constraint_uid for this level yet
                    # (in case the same constraint appears in multiple suites)
                    if constraint_uid not in seen_constraints_by_level[level_str]:
                        scores_by_level[level_str].append(score)
                        seen_constraints_by_level[level_str].add(constraint_uid)

        return scores_by_level

    def fail_precision(
            self,
            column_group_suites: List[DVTrajectoryColumnGroupSuite]
    ) -> Dict[ColumnGroup, Dict[Union[str, ConstraintUID], FailPrecisionScore]]:
        """
        Compute fail precision scores for trajectory column group suites.
        
        Fail precision measures how well a constraint identifies actual data quality issues:
        fail_precision = (correct failures) / (total failures)
        where a "correct failure" means the constraint failed validation AND the data is unsafe.
        
        A high fail precision score means when the constraint fails, it's usually correct
        (the data is actually unsafe). A low score means the constraint often fails on safe data
        (false positives).
        
        Args:
            column_group_suites: List of DVTrajectoryColumnGroupSuite objects
            
        Returns:
            Nested dict: {column_group: {constraint_uid: fail_precision_score, "overall": overall_score}}
        """
        return compute_fail_precision_dict(column_group_suites)

    def select_top_fail_precision(
            self,
            fail_precision_dict: Dict[ColumnGroup, Dict[Union[str, ConstraintUID], FailPrecisionScore]],
            num_top_column_groups: int,
            num_top_constraints: int,
            overall_key: str = "overall",
    ) -> Dict[ColumnGroup, Dict[ConstraintUID, FailPrecisionScore]]:
        """
        Select top column groups and constraints based on fail precision scores.
        
        First, ranks column groups by their overall fail precision score.
        Then, for each top column group, selects the top N constraints by score.
        
        Args:
            fail_precision_dict: Fail precision scores per column group and constraint
            num_top_column_groups: Number of top column groups to select
            num_top_constraints: Number of top constraints per column group
            overall_key: Key used for overall scores in the fail_precision_dict
            
        Returns:
            Dict mapping selected column groups to their top constraints with scores
        """
        return select_top_fail_precision(
            fail_precision_dict,
            num_top_column_groups,
            num_top_constraints,
            overall_key
        )

    def find_suite_by_constraint(
            self,
            column_group_suites: List[DVTrajectoryColumnGroupSuite],
            column_group: ColumnGroup,
            constraint_uid: ConstraintUID
    ) -> Union[DVTrajectoryColumnGroupSuite, None]:
        """
        Find a column group suite matching a specific column group and constraint UID.
        
        Args:
            column_group_suites: List of DVTrajectoryColumnGroupSuite objects
            column_group: The column group to match
            constraint_uid: The constraint UID to search for
            
        Returns:
            Matching DVTrajectoryColumnGroupSuite or None if not found
        """
        return find_suite_by_constraint(column_group_suites, column_group, constraint_uid)
