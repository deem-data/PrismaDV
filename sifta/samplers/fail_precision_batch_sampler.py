"""Custom batch sampler that uses fail precision scores for probability-based sampling."""

import hashlib
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np

from prismadv.llm.dspy.models.sampler.base import TrajectorySampler
from sifta.proposer.reflective_mutation.base import BatchSampler
from workflow_sifta.optimization_fns import OptimizationWorkflow


class FailPrecisionBatchSampler(BatchSampler):
    """
    Batch sampler that samples training examples based on fail precision scores.
    
    Higher fail precision scores get higher sampling probability. Uses softmax-based
    probability distribution similar to TrajectorySampler's sample_from_candidates.
    
    Fail precision scores are calculated using the same logic as optimization_metrics.py:
    - Run candidate program on example to generate constraints
    - Create trajectories from constraints
    - Use TrajectorySampler to calculate overall fail precision
    """

    def __init__(
            self,
            trainset: list,
            workflow: OptimizationWorkflow,
            llm_name: str,
            minibatch_size: int,
            rng: random.Random | None = None,
            seed_candidate: Optional[dict[str, str]] = None,
            temperature: float = 0.5,
            precompute_scores: bool = True,
            cache_path: Optional[str | Path] = None,
            use_cache: bool = True,
            sampling_strategy: str = "worst_first",
    ):
        """
        Initialize the fail precision batch sampler.

        Args:
            trainset: List of training examples (dspy.Example objects)
            workflow: OptimizationWorkflow instance for calculating fail precision
            llm_name: LLM name used for trajectory creation
            minibatch_size: Size of each minibatch
            rng: Random number generator for reproducibility
            seed_candidate: Initial candidate program dict (for precomputing scores)
            temperature: Temperature for softmax sampling (0.0 = deterministic, >0.0 = stochastic)
            precompute_scores: If True, precompute fail precision scores at initialization
            cache_path: Path to cache file for storing/loading scores. If None, uses default location.
            use_cache: If True, load from cache and save to cache. If False, skip caching.
            sampling_strategy: Strategy for sampling examples:
                - "best_first": Prioritize highest-scoring (best performing) examples
                - "worst_first": Prioritize lowest-scoring (worst performing) examples
                - "extreme_first": Prioritize most extreme scores (furthest from 0.5)
                - "random": Uniform random sampling
        """
        self.trainset = trainset
        self.workflow = workflow
        self.llm_name = llm_name
        self.minibatch_size = minibatch_size
        self.temperature = temperature
        self.use_cache = use_cache
        self.sampling_strategy = sampling_strategy

        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

        # Cache for fail precision scores (index -> score)
        self._fail_precision_scores: dict[int, float] = {}
        # Cache for detailed information (index -> detailed dict)
        self._fail_precision_details: dict[int, dict] = {}
        self._probabilities: Optional[np.ndarray] = None
        self._epoch = -1
        self._sampled_this_epoch: set[int] = set()

        # Set up cache path
        if cache_path is None:
            # Default cache location: workflow_sifta/samplers/cache/
            cache_dir = Path(__file__).parent / "cache"
            cache_dir.mkdir(exist_ok=True)
            # Generate cache filename based on trainset and llm_name
            cache_key = self._generate_cache_key(seed_candidate)
            self.cache_path = cache_dir / f"fail_precision_scores_{cache_key}.json"
        else:
            self.cache_path = Path(cache_path)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Store candidate hash for validation
        self._candidate_hash = self._compute_candidate_hash(seed_candidate) if seed_candidate else None
        print(f"FailPrecisionBatchSampler initialized with candidate hash: {self._candidate_hash}")
        print(f"Cache path: {self.cache_path}")

        # Load cache if available
        if use_cache and self.cache_path.exists():
            self._load_cache()
            print(f"Loaded {len(self._fail_precision_scores)} cached scores from {self.cache_path}")

        # Precompute scores if seed_candidate provided
        if precompute_scores and seed_candidate is not None:
            self._precompute_all_scores(seed_candidate)

    def _calculate_fail_precision_for_example(
            self,
            example: dict,
            candidate: dict[str, str]
    ) -> dict:
        """
        Calculate fail precision and detailed metrics for a single training example.

        Reuses logic from create_metric_with_feedback in optimization_metrics.py.

        Args:
            example: Training example (dspy.Example)
            candidate: Candidate program dict (component_name -> component_text)

        Returns:
            Dict containing:
                - fail_precision: float (0-1) or NaN if calculation fails
                - validity_info: dict with constraint validity details
                - per_label_info: dict mapping label -> trajectory/safety info
        """
        try:
            column_name = example["target_column"]
            script_name_list = [example["script_name"]]
            dataset_name = example["dataset_name"]
            subtask_name = example["subtask_name"]

            # Create a temporary module with candidate instructions
            from prismadv.llm.dspy.models.column_wise_module import ConstraintGenerationModule
            module = ConstraintGenerationModule()

            # Set instructions from candidate
            # Use with_instructions() to avoid modifying the class-level signature
            for pred_name, pred in module.named_predictors():
                if pred_name in candidate:
                    pred.signature = pred.signature.with_instructions(candidate[pred_name])

            # Generate constraints for this example
            single_column_results = {}
            pred = module(
                code_script=example["code_script"],
                target_column=column_name,
                target_column_desc=example["target_column_desc"],
                downstream_task_description=example["downstream_task_description"],
            )
            single_column_results[column_name] = pred

            # Combine constraints
            constraints_with_sources = self.workflow.combine_constraints(
                single_column_results=single_column_results,
            )

            # Validate constraints on training data for this specific dataset/subtask
            constraints_with_sources = self.workflow.validate_constraints_on_training_data(
                dataset_name=dataset_name,
                subtask_name=subtask_name,
                constraints_with_sources=constraints_with_sources,
            )

            # Get validity information for detailed caching
            validity_info = constraints_with_sources.get_constraint_validity_info()

            # Create trajectories and collect per-label info
            all_trajectories_dict = {}
            per_label_info = {}
            script_name = example["script_name"]
            for label in self.workflow.train_processed_data_label_list:
                # Validate constraints on test data (script_name doesn't matter for validation)
                validation_results = self.workflow.validate_constraints_on_test_data(
                    dataset_name=dataset_name,
                    subtask_name=subtask_name,
                    constraints_with_sources=constraints_with_sources,
                    processed_data_label=label,
                    clean=False,
                )

                # Construct validation_results_dict with script_name from example
                validation_results_dict = {dataset_name: {script_name: validation_results}}
                trajectories_dict = self.workflow.create_trajectories_from_constraints(
                    script_name_list=script_name_list,
                    processed_data_label=label,
                    llm_name=self.llm_name,
                    constraints_with_sources=constraints_with_sources,
                    validation_results_dict=validation_results_dict,
                    clean=False,
                )

                # Collect per-label information
                label_trajectories = []
                for ds_name, script_dict in trajectories_dict.items():
                    if ds_name not in all_trajectories_dict:
                        all_trajectories_dict[ds_name] = {}
                    for sc_name, trajectories in script_dict.items():
                        if sc_name not in all_trajectories_dict[ds_name]:
                            all_trajectories_dict[ds_name][sc_name] = []
                        all_trajectories_dict[ds_name][sc_name].extend(trajectories)
                        label_trajectories.extend(trajectories)

                # Extract is_safe from trajectory objects (not from validation_results)
                # All trajectories for a given label should have the same is_safe value
                trajectory_is_safe = None
                if label_trajectories:
                    # Get is_safe from first trajectory (all should be the same for this label)
                    trajectory_is_safe = getattr(label_trajectories[0], 'is_safe', None)

                # Check if all trajectories passed validation (predicted_as_safe)
                predicted_as_safe = 0
                if label_trajectories:
                    all_passed = all(
                        getattr(traj.validation_results, 'status', False) is True
                        for traj in label_trajectories
                    )
                    predicted_as_safe = 1 if all_passed else 0

                # Store per-label info
                per_label_info[label] = {
                    "num_trajectories": len(label_trajectories),
                    "has_failures": len(label_trajectories) > 0,  # Trajectories indicate failures
                    "is_safe": trajectory_is_safe,
                    "predicted_as_safe": predicted_as_safe,  # 1 if all constraints passed, 0 otherwise
                }

            # Aggregate trajectories
            aggregated_trajectories = self.workflow.aggregate_trajectories_for_sampling(
                trajectories_dict=all_trajectories_dict,
                llm_name=self.llm_name,
            )

            # Calculate fail precision using TrajectorySampler
            if not aggregated_trajectories:
                return {
                    "fail_precision": np.nan,
                    "validity_info": validity_info,
                    "per_label_info": per_label_info,
                }

            sampler = TrajectorySampler.from_aggregated_trajectories(aggregated_trajectories)
            sampler.calculate()

            sampled_results = sampler.sample(
                temperature=0.0, num_top_column_groups=1, num_top_constraints=10
            )

            if len(sampled_results) == 0:
                return {
                    "fail_precision": np.nan,
                    "validity_info": validity_info,
                    "per_label_info": per_label_info,
                }

            fail_precision = list(sampled_results.values())[0].overall_fail_precision

            # Return detailed result dict
            return {
                "fail_precision": float(fail_precision) if not np.isnan(fail_precision) else np.nan,
                "validity_info": validity_info,
                "per_label_info": per_label_info,
            }

        except Exception as e:
            # Return error result dict
            example_id = f"{example.get('dataset_name', '?')}/{example.get('script_name', '?')}/{example.get('target_column', '?')}"
            print(f"Warning: Failed to calculate fail precision for example {example_id}: {e}")
            return {
                "fail_precision": np.nan,
                "validity_info": {},
                "per_label_info": {},
            }

    def _compute_candidate_hash(self, candidate: Optional[dict[str, str]]) -> Optional[str]:
        """Compute a hash for the candidate instructions for validation purposes."""
        if not candidate:
            return None
        candidate_str = json.dumps(candidate, sort_keys=True)
        return hashlib.md5(candidate_str.encode()).hexdigest()[:16]

    def _generate_cache_key(self, candidate: Optional[dict[str, str]]) -> str:
        """Generate a cache key based on trainset, workflow parameters, and candidate."""
        # Create a hash from trainset identifiers, workflow config, and candidate
        key_parts = []

        # Hash all examples' identifiers to ensure unique cache keys
        # Each example is uniquely identified by: dataset_name, subtask_name, script_name, target_column
        example_identifiers = []
        for example in self.trainset:
            example_id = (
                f"{example.get('dataset_name', '')}_"
                f"{example.get('subtask_name', '')}_"
                f"{example.get('script_name', '')}_"
                f"{example.get('target_column', '')}"
            )
            example_identifiers.append(example_id)

        # Sort identifiers for consistent hashing regardless of order
        example_identifiers.sort()

        # Create a hash of all example identifiers
        examples_str = "|".join(example_identifiers)
        examples_hash = hashlib.md5(examples_str.encode()).hexdigest()[:16]
        key_parts.append(f"examples_{examples_hash}")

        key_parts.append(f"llm_{self.llm_name}")
        key_parts.append(f"size_{len(self.trainset)}")

        # Add workflow parameters that affect fail precision calculation
        # train_processed_data_label_list is critical as it determines which labels are used
        labels_str = "_".join(sorted(self.workflow.train_processed_data_label_list))
        labels_hash = hashlib.md5(labels_str.encode()).hexdigest()[:8]
        key_parts.append(f"labels_{labels_hash}")

        # Add candidate hash if provided
        if candidate:
            candidate_str = json.dumps(candidate, sort_keys=True)
            candidate_hash = hashlib.md5(candidate_str.encode()).hexdigest()[:8]
            key_parts.append(f"candidate_{candidate_hash}")

        combined = "_".join(key_parts)
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def _load_cache(self):
        """Load fail precision scores and detailed data from cache file with validation."""
        try:
            with open(self.cache_path, 'r') as f:
                cache_data = json.load(f)

                # Validate cache metadata if present
                if "metadata" in cache_data:
                    metadata = cache_data["metadata"]
                    current_labels = sorted(self.workflow.train_processed_data_label_list)
                    cached_labels = metadata.get("train_processed_data_label_list", [])

                    # Warn if workflow parameters have changed
                    if cached_labels and cached_labels != current_labels:
                        print(f"Warning: Cache workflow labels mismatch!")
                        print(f"  Cached labels: {cached_labels}")
                        print(f"  Current labels: {current_labels}")
                        print(f"  Cache may be stale. Consider deleting: {self.cache_path}")

                    # Validate candidate hash
                    cached_candidate_hash = metadata.get("candidate_hash")
                    if cached_candidate_hash and self._candidate_hash:
                        if cached_candidate_hash != self._candidate_hash:
                            print(f"Warning: Cache candidate hash mismatch!")
                            print(f"  Cached candidate hash: {cached_candidate_hash}")
                            print(f"  Current candidate hash: {self._candidate_hash}")
                            print(f"  This indicates the prompt instructions have changed.")
                            print(f"  Cache will be invalidated. Scores will be recomputed.")
                            # Clear loaded data and return - scores will be recomputed
                            self._fail_precision_scores = {}
                            self._fail_precision_details = {}
                            return
                        else:
                            print(f"Cache candidate hash verified: {cached_candidate_hash}")

                # Check if this is the new format with "details" key
                if "details" in cache_data:
                    details_data = cache_data["details"]
                    for example_key, v in details_data.items():
                        if example_key == "metadata":
                            continue

                        # Check if this is new format (with example_info) or old format (numeric key)
                        if "example_info" in v:
                            # New format: has example_info with metadata
                            # Match against current trainset to find idx
                            example_info = v["example_info"]

                            # Try to find matching example in current trainset
                            found_idx = None
                            for idx, example in enumerate(self.trainset):
                                if (example.get('dataset_name') == example_info.get('dataset_name') and
                                        example.get('subtask_name') == example_info.get('subtask_name') and
                                        example.get('script_name') == example_info.get('script_name') and
                                        example.get('target_column') == example_info.get('target_column')):
                                    found_idx = idx
                                    break

                            if found_idx is not None:
                                # Extract score
                                fail_precision = v.get("fail_precision")
                                self._fail_precision_scores[found_idx] = (
                                    float(fail_precision) if fail_precision != "nan" else np.nan
                                )
                                # Store full details
                                self._fail_precision_details[found_idx] = v
                        else:
                            # Old format: numeric key
                            try:
                                idx = int(example_key)
                                fail_precision = v.get("fail_precision")
                                self._fail_precision_scores[idx] = (
                                    float(fail_precision) if fail_precision != "nan" else np.nan
                                )
                                self._fail_precision_details[idx] = v
                            except ValueError:
                                # Skip invalid keys
                                continue
                else:
                    # Old format: backward compatible - only scores
                    scores_data = cache_data.get("scores", cache_data)
                    self._fail_precision_scores = {
                        int(k): (float(v) if v != "nan" else np.nan)
                        for k, v in scores_data.items()
                        if k != "metadata"
                    }
        except Exception as e:
            print(f"Warning: Failed to load cache from {self.cache_path}: {e}")
            self._fail_precision_scores = {}
            self._fail_precision_details = {}

    def _get_example_key(self, example: dict) -> str:
        """Generate a descriptive key for an example from its metadata."""
        dataset_name = example.get('dataset_name', 'unknown')
        subtask_name = example.get('subtask_name', 'unknown')
        script_name = example.get('script_name', 'unknown')
        target_column = example.get('target_column', 'unknown')
        return f"{dataset_name}_{subtask_name}_{script_name}_{target_column}"

    def _save_cache(self):
        """Save fail precision scores and detailed data to cache file with metadata."""
        if not self.use_cache:
            return

        try:
            # Helper function to convert NaN to string for JSON serialization
            def serialize_value(v):
                if isinstance(v, float) and np.isnan(v):
                    return "nan"
                return v

            # Build detailed cache structure with descriptive keys
            details_to_save = {}
            for idx, detail in self._fail_precision_details.items():
                # Get example metadata for descriptive key
                example = self.trainset[idx]
                example_key = self._get_example_key(example)

                serialized_detail = {
                    "fail_precision": serialize_value(detail.get("fail_precision")),
                    "validity_info": detail.get("validity_info", {}),
                    "per_label_info": detail.get("per_label_info", {}),
                    # Store example metadata for identification
                    "example_info": {
                        "dataset_name": example.get('dataset_name', 'unknown'),
                        "subtask_name": example.get('subtask_name', 'unknown'),
                        "script_name": example.get('script_name', 'unknown'),
                        "target_column": example.get('target_column', 'unknown'),
                        "idx": idx,  # Keep idx for backward compatibility
                    }
                }
                details_to_save[example_key] = serialized_detail

            # Build cache structure with metadata
            cache_structure = {
                "metadata": {
                    "llm_name": self.llm_name,
                    "trainset_size": len(self.trainset),
                    "train_processed_data_label_list": sorted(self.workflow.train_processed_data_label_list),
                    "candidate_hash": self._candidate_hash,  # Store for validation on load
                },
                "details": details_to_save,
            }

            with open(self.cache_path, 'w') as f:
                json.dump(cache_structure, f, indent=2)
            print(f"Saved {len(self._fail_precision_details)} detailed results to cache: {self.cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache to {self.cache_path}: {e}")

    def _precompute_all_scores(self, candidate: dict[str, str]):
        """Precompute fail precision scores for all training examples."""
        # Check how many scores are already cached
        cached_count = sum(1 for idx in range(len(self.trainset)) if idx in self._fail_precision_scores)
        total_count = len(self.trainset)

        if cached_count == total_count:
            print(f"All {total_count} scores already cached. Skipping precomputation.")
            self._build_probability_distribution()
            return

        print(f"Precomputing fail precision scores for {total_count} examples...")
        print(f"  {cached_count} scores already cached, computing {total_count - cached_count} new scores...")

        computed_count = 0
        for idx, example in enumerate(self.trainset):
            # Skip if already cached
            if idx in self._fail_precision_scores:
                continue

            # Calculate and get detailed result
            result = self._calculate_fail_precision_for_example(example, candidate)

            # Store score and details separately
            self._fail_precision_scores[idx] = result["fail_precision"]
            self._fail_precision_details[idx] = result
            computed_count += 1

            # Save cache periodically (every 5 scores)
            if computed_count % 5 == 0:
                self._save_cache()
                print(
                    f"  Computed {computed_count}/{total_count - cached_count} new scores (total cached: {cached_count + computed_count}/{total_count})")

        # Final save
        self._save_cache()
        print(f"Score precomputation complete. Total cached: {len(self._fail_precision_scores)}/{total_count}")
        self._build_probability_distribution()

    def _build_probability_distribution(self):
        """Build probability distribution from fail precision scores.

        Supports different sampling strategies:
        - "best_first": Higher scores get higher sampling probability
        - "worst_first": Lower scores get higher sampling probability
        - "extreme_first": Most extreme scores (furthest from 0.5) get higher probability
        - "random": Uniform distribution ignoring scores

        Temperature controls the softness of sampling (higher = more uniform).
        """
        trainset_size = len(self.trainset)

        # Handle "random" strategy: uniform distribution
        if self.sampling_strategy == "random":
            self._probabilities = np.ones(trainset_size) / trainset_size
            return

        scores = np.array([
            self._fail_precision_scores.get(i, np.nan)
            for i in range(trainset_size)
        ])

        # Handle NaN scores
        nan_mask = np.isnan(scores)
        non_nan_scores = scores[~nan_mask]

        if len(non_nan_scores) == 0:
            # All scores are NaN, use uniform distribution
            self._probabilities = np.ones(trainset_size) / trainset_size
            return

        # Transform scores based on sampling strategy
        if self.sampling_strategy == "best_first":
            # Higher scores get higher probability
            # NaN scores get low probability
            min_non_nan = np.min(non_nan_scores)
            nan_probability = max(0.01 * min_non_nan, 1e-6)
            scores[nan_mask] = nan_probability
            sampling_weights = scores

        elif self.sampling_strategy == "worst_first":
            # Lower scores get higher probability
            # Invert: weight = max - score + epsilon
            max_score = np.nanmax(non_nan_scores)
            sampling_weights = max_score - scores + 1e-6
            # NaN gets high priority (treat as worst case)
            sampling_weights[nan_mask] = max_score + 1e-6

        elif self.sampling_strategy == "extreme_first":
            # Most extreme scores (furthest from 0.5) get higher probability
            # weight = abs(score - 0.5)
            sampling_weights = np.abs(scores - 0.5)
            # NaN gets medium priority (treat as 0.5, so weight = 0)
            # Add small epsilon to avoid zero weights
            sampling_weights = sampling_weights + 1e-6
            sampling_weights[nan_mask] = 1e-6

        else:
            raise ValueError(
                f"Unknown sampling_strategy: {self.sampling_strategy}. "
                f"Must be one of: 'best_first', 'worst_first', 'extreme_first', 'random'"
            )

        # Ensure all weights are non-negative
        sampling_weights = np.maximum(sampling_weights, 0.0)

        # Apply softmax with temperature
        if self.temperature == 0.0:
            # Deterministic: use weights directly (normalize)
            self._probabilities = sampling_weights / sampling_weights.sum()
        else:
            # Stochastic: use softmax with temperature
            temp = max(self.temperature, 1e-8)
            logits = sampling_weights / temp

            # Numerical stability: subtract max before exponentiating
            logits = logits - np.max(logits)
            exp_logits = np.exp(logits)
            self._probabilities = exp_logits / exp_logits.sum()

    def _reset_epoch(self):
        """Reset epoch tracking."""
        self._epoch += 1
        self._sampled_this_epoch = set()
        # Rebuild probability distribution (in case scores changed)
        if self._probabilities is None:
            self._build_probability_distribution()

    def next_minibatch_indices(self, trainset_size: int, iteration: int) -> list[int]:
        """
        Sample next minibatch indices based on fail precision scores.
        
        Args:
            trainset_size: Size of training set (should match len(self.trainset))
            iteration: Current iteration number
            
        Returns:
            List of indices for the minibatch
        """
        assert trainset_size == len(self.trainset), (
            f"trainset_size ({trainset_size}) doesn't match len(self.trainset) ({len(self.trainset)})"
        )

        # If all examples have been sampled this epoch, reset for next epoch
        examples_per_epoch = len(self.trainset)
        if len(self._sampled_this_epoch) >= examples_per_epoch:
            self._reset_epoch()

        # Build probability distribution if not already built
        if self._probabilities is None:
            self._build_probability_distribution()

        # Get available indices (not yet sampled this epoch)
        available_indices = [
            i for i in range(trainset_size)
            if i not in self._sampled_this_epoch
        ]

        if len(available_indices) == 0:
            # All sampled, reset epoch
            self._reset_epoch()
            available_indices = list(range(trainset_size))

        # Sample without replacement from available indices
        num_to_sample = min(self.minibatch_size, len(available_indices))

        if num_to_sample == 0:
            return []

        # Get probabilities for available indices
        available_probs = self._probabilities[available_indices]
        available_probs = available_probs / available_probs.sum()  # Renormalize

        # Sample without replacement using cumulative probabilities
        sampled_indices = []
        remaining_indices = available_indices.copy()
        remaining_probs = available_probs.copy()

        for _ in range(num_to_sample):
            if len(remaining_indices) == 0:
                break

            # Renormalize probabilities
            remaining_probs = remaining_probs / remaining_probs.sum()

            # Sample one index
            cumsum = np.cumsum(remaining_probs)
            r = self.rng.random()
            selected_idx = np.searchsorted(cumsum, r)
            selected_idx = min(selected_idx, len(remaining_indices) - 1)

            sampled_indices.append(remaining_indices[selected_idx])

            # Remove selected index
            remaining_indices.pop(selected_idx)
            remaining_probs = np.delete(remaining_probs, selected_idx)

        # Track the original sampled indices (before padding)
        # This ensures epoch reset logic works correctly
        original_sampled_count = len(sampled_indices)

        # If we got fewer than minibatch_size, pad to reach minibatch_size
        if len(sampled_indices) < self.minibatch_size:
            remaining_needed = self.minibatch_size - len(sampled_indices)

            # First, try to pad from remaining available indices (not yet sampled this epoch)
            remaining_available = [i for i in available_indices if i not in sampled_indices]
            if remaining_available:
                # Sample additional indices uniformly from remaining available
                num_to_add = min(remaining_needed, len(remaining_available))
                additional = self.rng.sample(remaining_available, num_to_add)
                sampled_indices.extend(additional)
                remaining_needed -= num_to_add

            # If still need more, pad by sampling from all indices (with replacement)
            # This mimics EpochShuffledBatchSampler's padding behavior
            if remaining_needed > 0:
                # Use probability distribution to sample from all indices (including already sampled)
                # This ensures padding favors higher-scoring examples
                all_indices = list(range(trainset_size))
                all_probs = self._probabilities.copy()
                all_probs = all_probs / all_probs.sum()  # Normalize

                # Sample with replacement for padding
                padding_indices = self.rng.choices(
                    all_indices,
                    weights=all_probs,
                    k=remaining_needed
                )
                sampled_indices.extend(padding_indices)

        # Track only the originally sampled indices (not padding duplicates)
        # This ensures epoch tracking counts each unique example only once
        self._sampled_this_epoch.update(sampled_indices[:original_sampled_count])
        print(f"Sampled minibatch indices: {sampled_indices}")
        return sampled_indices[:self.minibatch_size]
