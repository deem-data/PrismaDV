from typing import Any

from sifta.core.adapter import DataInst, SIFTAAdapter, RolloutOutput, Trajectory
from sifta.core.state import SIFTAState
from sifta.proposer.base import CandidateProposal, ProposeNewCandidate
from sifta.proposer.reflective_mutation.base import (
    BatchSampler,
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)


class ReflectiveMutationProposer(ProposeNewCandidate):
    """
    Implements current reflective mutation flow:
    - Select candidate via selector
    - Select minibatch via sampler
    - capture_traces_and_eval -> trajectories, subsample_scores
    - skip if all scores==perfect and skip_perfect_score
    - reflection + mutate -> new candidate
    - evaluate new candidate on same minibatch -> new_subsample_scores
    - Return proposal if improved; else None
    """

    def __init__(
            self,
            logger: Any,
            trainset: list[DataInst],
            adapter: SIFTAAdapter[DataInst, Trajectory, RolloutOutput],
            candidate_selector: CandidateSelector,
            module_selector: ReflectionComponentSelector,
            batch_sampler: BatchSampler,
            perfect_score: float,
            skip_perfect_score: bool,
            use_wandb: bool,
            reflection_lm: LanguageModel | None = None,
            use_global_proposer: bool = False,
    ):
        self.logger = logger
        self.trainset = trainset
        self.adapter = adapter
        self.candidate_selector = candidate_selector
        self.module_selector = module_selector
        self.batch_sampler = batch_sampler
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.use_wandb = use_wandb
        self.reflection_lm = reflection_lm
        self.use_global_proposer = use_global_proposer

    def propose_new_texts(
            self,
            candidate: dict[str, str],
            reflective_dataset: dict[str, list[dict[str, Any]]],
            components_to_update: list[str]
    ) -> dict[str, str]:
        # Use global proposer if enabled
        if self.use_global_proposer:
            return self.propose_new_texts_global(candidate, reflective_dataset)
        else:
            raise NotImplementedError("Local proposer not implemented in this example.")

    def propose_new_texts_global(
            self,
            candidate: dict[str, str],
            reflective_dataset: dict[str, list[dict[str, Any]]],
    ) -> dict[str, str]:
        """
        Global proposer that sees all modules and decides which to update.
        """
        from sifta.strategies.instruction_proposal import GlobalInstructionProposalSignature

        # Build all_modules dict with signatures
        all_modules: dict[str, dict] = {}
        module_signatures = {}

        # Try to get signatures from adapter if available
        if hasattr(self.adapter, 'get_module_signatures'):
            module_signatures = self.adapter.get_module_signatures()

        for name, instruction in candidate.items():
            sig_info = module_signatures.get(name, {})
            all_modules[name] = {
                "instruction": instruction,
                "inputs": sig_info.get("inputs", "N/A"),
                "outputs": sig_info.get("outputs", "N/A"),
            }

        # Build all_feedback - use reflective_dataset which has feedback for all modules
        all_feedback: dict[str, list] = {}
        for name in candidate.keys():
            all_feedback[name] = reflective_dataset.get(name, [])

        # Print concise summary
        feedback_summary = ", ".join(f"{name}:{len(fb)}" for name, fb in all_feedback.items())
        instr_summary = ", ".join(f"{name}:{len(candidate[name])}" for name in candidate)
        print(
            f"[Global] Modules: {list(all_modules.keys())} | Feedback: [{feedback_summary}] | Instr lens: [{instr_summary}]")

        # Call global proposer
        result = GlobalInstructionProposalSignature.run(
            lm=self.reflection_lm,
            input_dict={
                "all_modules": all_modules,
                "all_feedback": all_feedback,
            }
        )

        updated_instructions = result.get("updated_instructions", {})

        if not updated_instructions:
            print(f"[Global] WARNING: No updates returned, keeping original")
            return {}

        # Validate and summarize changes
        valid_updates = {}
        changes = []

        # Create case-insensitive lookup for robust matching
        candidate_lower_to_original = {k.lower(): k for k in candidate.keys()}

        for name, new_instruction in updated_instructions.items():
            name_lower = name.lower()
            if name_lower in candidate_lower_to_original:
                original_name = candidate_lower_to_original[name_lower]
                valid_updates[original_name] = new_instruction
                if new_instruction != candidate[original_name]:
                    changes.append(f"{original_name}:{len(candidate[original_name])}->{len(new_instruction)}")
            else:
                print(f"[Global] Warning: Unknown module '{name}', skipping")

        if changes:
            print(f"[Global] Updated: {', '.join(changes)}")
        else:
            print(f"[Global] No actual changes in instructions")

        return valid_updates

    def propose(self, state: SIFTAState) -> CandidateProposal | None:
        i = state.i + 1

        curr_prog_id = self.candidate_selector.select_candidate_idx(state)
        curr_prog = state.program_candidates[curr_prog_id]
        state.full_program_trace[-1]["selected_program_candidate"] = curr_prog_id
        self.logger.log(
            f"Iteration {i}: Selected program {curr_prog_id} score: {state.per_program_tracked_scores[curr_prog_id]}")

        if self.use_wandb:
            import wandb  # type: ignore
            wandb.log({"iteration": i, "selected_program_candidate": curr_prog_id}, step=i)

        subsample_ids = self.batch_sampler.next_minibatch_indices(len(self.trainset), i - 1)
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids
        minibatch = [self.trainset[j] for j in subsample_ids]

        # 1) Evaluate current program with traces
        eval_curr = self.adapter.evaluate(minibatch, curr_prog, capture_traces=True)
        print(
            f"[SIFTA] Iteration {i}: Captured {len(eval_curr.trajectories) if eval_curr.trajectories else 0} trajectories, scores: {eval_curr.scores}")
        self.logger.log(
            f"Iteration {i}: Captured {len(eval_curr.trajectories) if eval_curr.trajectories else 0} trajectories, scores: {eval_curr.scores}")
        if not eval_curr.trajectories or len(eval_curr.trajectories) == 0:
            print(f"[SIFTA] Iteration {i}: No trajectories captured. Skipping.")
            self.logger.log(f"Iteration {i}: No trajectories captured. Skipping.")
            return None

        state.total_num_evals += len(subsample_ids)
        state.full_program_trace[-1]["subsample_scores"] = eval_curr.scores

        if self.skip_perfect_score and all(s >= self.perfect_score for s in eval_curr.scores):
            self.logger.log(f"Iteration {i}: All subsample scores perfect. Skipping.")
            return None

        if self.use_wandb:
            import wandb  # type: ignore
            wandb.log({"subsample_score": sum(eval_curr.scores)}, step=i)

        # 2) Decide which predictors to update
        if self.use_global_proposer:
            # Global mode: get feedback for ALL predictors, let LM decide which to update
            predictor_names_to_update = list(curr_prog.keys())
            print(
                f"[SIFTA] Iteration {i}: Global mode - gathering feedback for all predictors: {predictor_names_to_update}")
        else:
            predictor_names_to_update = self.module_selector.select_modules(
                state, eval_curr.trajectories, eval_curr.scores, curr_prog_id, curr_prog
            )

        # 3) Build reflective dataset and propose texts
        try:
            reflective_dataset = self.adapter.make_reflective_dataset(
                curr_prog, eval_curr, predictor_names_to_update
            )
            print(f"[SIFTA] Iteration {i}: Proposing new texts for predictors: {predictor_names_to_update}")
            new_texts = self.propose_new_texts(
                curr_prog, reflective_dataset, predictor_names_to_update
            )
            for pname, text in new_texts.items():
                print(f"[SIFTA] Iteration {i}: Proposed new text for {pname}:")
                print(f"  {text}")
                self.logger.log(f"Iteration {i}: Proposed new text for {pname}: {text}")
            if self.use_wandb:
                import wandb  # type: ignore
                wandb.log({f"new_instruction_{pname}": text for pname, text in new_texts.items()}, step=i)
        except Exception as e:
            self.logger.log(f"Iteration {i}: Exception during reflection/proposal: {e}")
            import traceback
            self.logger.log(traceback.format_exc())
            return None

        # 4) Create candidate, evaluate on same minibatch (capture traces to compare outputs)
        new_candidate = curr_prog.copy()
        for pname, text in new_texts.items():
            assert pname in new_candidate, f"{pname} missing in candidate"
            new_candidate[pname] = text

        eval_new = self.adapter.evaluate(minibatch, new_candidate, capture_traces=True)
        state.total_num_evals += len(subsample_ids)
        state.full_program_trace[-1]["new_subsample_scores"] = eval_new.scores

        # Print detailed score comparison
        old_sum = sum(eval_curr.scores)
        new_sum = sum(eval_new.scores)
        print(f"\n[SIFTA] Iteration {i}: Score Comparison:")
        print(f"  Before: {eval_curr.scores} (sum={old_sum:.2f})")
        print(f"  After:  {eval_new.scores} (sum={new_sum:.2f})")
        print(f"  Change: {new_sum - old_sum:+.2f}")

        if new_sum < old_sum:
            print(f"  ⚠️  WORSE - Scores decreased by {old_sum - new_sum:.2f}")
        elif new_sum > old_sum:
            print(f"  ✓ BETTER - Scores improved by {new_sum - old_sum:.2f}")
        else:
            print(f"  = SAME - No change in scores")

        # Print constraint output comparison when scores change
        if new_sum != old_sum and eval_curr.trajectories and eval_new.trajectories:
            print(f"\n[SIFTA] Iteration {i}: Constraint Output Comparison:")
            # Show first example that changed scores
            for idx, (old_score, new_score) in enumerate(zip(eval_curr.scores, eval_new.scores)):
                if old_score != new_score:
                    print(f"\n  === Example {idx + 1} (score: {old_score:.2f} → {new_score:.2f}) ===")

                    # Get trajectories for this example
                    old_traj = eval_curr.trajectories[idx] if idx < len(eval_curr.trajectories) else None
                    new_traj = eval_new.trajectories[idx] if idx < len(eval_new.trajectories) else None

                    # Helper to get value from dict or object
                    def get_attr(obj, key, default=None):
                        if isinstance(obj, dict):
                            return obj.get(key, default)
                        return getattr(obj, key, default)

                    def print_trajectory_info(traj, label):
                        """Extract and print constraint info from DSPy trajectory structure.

                        DSPy trajectory structure:
                        - 'example': Input dict with script_name, target_column, dataset_name, etc.
                        - 'prediction': Module output dict with 'assumptions' and 'code' (generated constraints)
                        - 'trace': DSPy execution trace
                        - 'score': The fail precision score
                        """
                        if not traj:
                            print(f"  {label} Trajectory: None")
                            return

                        print(f"  {label} Trajectory:")

                        # Extract example (input to the metric)
                        example = get_attr(traj, 'example', {})
                        script_name = get_attr(example, 'script_name', 'N/A')
                        target_column = get_attr(example, 'target_column', 'N/A')
                        is_safe = get_attr(example, 'is_safe', 'N/A')

                        print(f"    Script: {script_name}, Column: {target_column}, is_safe: {is_safe}")

                        # Extract prediction (model's output)
                        prediction = get_attr(traj, 'prediction', {})

                        # Get generated constraints from prediction['code']
                        code_entries = get_attr(prediction, 'code', [])
                        if code_entries:
                            print(f"    Generated {len(code_entries)} constraint(s):")
                            for i, code_entry in enumerate(code_entries[:3]):  # Show first 3
                                suggestion = get_attr(code_entry, 'suggestion', str(code_entry))
                                print(f"      [{i + 1}] {suggestion}")
                                level = get_attr(code_entry, 'level')
                                if level:
                                    print(f"          Level: {level}")
                        else:
                            print(f"    No constraints generated")

                        # Get generated assumptions
                        assumptions = get_attr(prediction, 'assumptions', [])
                        if assumptions:
                            print(f"    Generated {len(assumptions)} assumption(s)")

                        # Show score
                        score = get_attr(traj, 'score')
                        if score is not None:
                            print(f"    Score: {score}")

                    # Print OLD and NEW trajectory info
                    print_trajectory_info(old_traj, "OLD")
                    print_trajectory_info(new_traj, "NEW")

                    # Only show first changed example to avoid too much output
                    break

        if self.use_wandb:
            import wandb  # type: ignore
            wandb.log({"new_subsample_score": new_sum}, step=i)

        return CandidateProposal(
            candidate=new_candidate,
            parent_program_ids=[curr_prog_id],
            subsample_indices=subsample_ids,
            subsample_scores_before=eval_curr.scores,
            subsample_scores_after=eval_new.scores,
            tag="reflective_mutation",
        )
