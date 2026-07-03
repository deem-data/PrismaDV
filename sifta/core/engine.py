import traceback
from typing import Any, Callable, Generic

from sifta.core.adapter import DataInst, RolloutOutput, Trajectory
from sifta.core.state import SIFTAState, initialize_sifta_state
from sifta.logging.utils import log_detailed_metrics_after_discovering_new_program
from sifta.logging.wandb_utils import initialize_wandb
from sifta.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer

# Import tqdm for progress bar functionality
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class GEPAEngine(Generic[DataInst, Trajectory, RolloutOutput]):
    """
    Orchestrates the optimization loop. It uses pluggable ProposeNewCandidate strategies.
    """

    def __init__(
            self,
            run_dir: str | None,
            evaluator: Callable[[list[DataInst], dict[str, str]], tuple[list[RolloutOutput], list[float]]],
            valset: list[DataInst] | None,
            seed_candidate: dict[str, str],
            # Controls
            max_metric_calls: int | None,
            perfect_score: float,
            seed: int,
            # Strategies and helpers
            reflective_proposer: ReflectiveMutationProposer,
            # Logging
            logger: Any,
            use_wandb: bool = False,
            wandb_api_key: str | None = None,
            wandb_init_kwargs: dict[str, Any] | None = None,
            track_best_outputs: bool = False,
            display_progress_bar: bool = False,
            raise_on_exception: bool = True,
    ):
        # Budget constraint: max_metric_calls must be set
        assert max_metric_calls is not None, "max_metric_calls must be set"

        self.logger = logger
        self.run_dir = run_dir
        self.evaluator = evaluator
        self.valset = valset
        self.seed_candidate = seed_candidate

        self.max_metric_calls = max_metric_calls

        self.perfect_score = perfect_score
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key
        self.seed = seed
        self.wandb_init_kwargs = wandb_init_kwargs or {}

        self.reflective_proposer = reflective_proposer

        self.track_best_outputs = track_best_outputs
        self.display_progress_bar = display_progress_bar

        self.raise_on_exception = raise_on_exception

    def _val_evaluator(self) -> Callable[[dict[str, str]], tuple[list[RolloutOutput], list[float]]]:
        assert self.valset is not None
        return lambda prog: self.evaluator(self.valset, prog)

    def _get_pareto_front_programs(self, state: SIFTAState) -> list:
        return state.program_at_pareto_front_valset

    def _run_full_eval_and_add(
            self,
            new_program: dict[str, str],
            state: SIFTAState,
            parent_program_idx: list[int],
    ) -> tuple[int, int]:
        num_metric_calls_by_discovery = state.total_num_evals

        valset_outputs, valset_subscores = self._val_evaluator()(new_program)
        valset_score = sum(valset_subscores) / len(valset_subscores)

        state.num_full_ds_evals += 1
        state.total_num_evals += len(valset_subscores)

        new_program_idx, linear_pareto_front_program_idx = state.update_state_with_new_program(
            parent_program_idx=parent_program_idx,
            new_program=new_program,
            valset_score=valset_score,
            valset_outputs=valset_outputs,
            valset_subscores=valset_subscores,
            run_dir=self.run_dir,
            num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery,
        )
        state.full_program_trace[-1]["new_program_idx"] = new_program_idx

        if new_program_idx == linear_pareto_front_program_idx:
            self.logger.log(f"Iteration {state.i + 1}: New program is on the linear pareto front")

        log_detailed_metrics_after_discovering_new_program(
            logger=self.logger,
            gepa_state=state,
            valset_score=valset_score,
            new_program_idx=new_program_idx,
            valset_subscores=valset_subscores,
            # new_instruction="Merged or Reflective program",
            use_wandb=self.use_wandb,
            linear_pareto_front_program_idx=linear_pareto_front_program_idx,
        )
        return new_program_idx, linear_pareto_front_program_idx

    def run(self) -> SIFTAState:
        if self.use_wandb:
            initialize_wandb(wandb_api_key=self.wandb_api_key, wandb_init_kwargs=self.wandb_init_kwargs)

        # Check tqdm availability if progress bar is enabled
        progress_bar = None
        if self.display_progress_bar:
            if tqdm is None:
                raise ImportError("tqdm must be installed when display_progress_bar is enabled")
            # Initialize progress bar
            progress_bar = tqdm(total=self.max_metric_calls, desc="SIFTA Optimization", unit="rollouts")
            progress_bar.update(0)
            last_pbar_val = 0

        # Prepare valset
        if self.valset is None:
            raise ValueError("valset must be provided to GEPAEngine.run()")

        # Initialize state (keeps your previous semantics)
        state = initialize_sifta_state(
            run_dir=self.run_dir,
            logger=self.logger,
            seed_candidate=self.seed_candidate,
            valset_evaluator=self._val_evaluator(),
            track_best_outputs=self.track_best_outputs,
        )

        assert len(state.pareto_front_valset) == len(self.valset)

        if self.use_wandb:
            import wandb  # type: ignore

            wandb.log(
                {
                    "base_program_full_valset_score": state.program_full_scores_val_set[0],
                    "iteration": state.i + 1,
                }
            )

        self.logger.log(
            f"Iteration {state.i + 1}: Base program full valset score: {state.program_full_scores_val_set[0]}"
        )

        # Main loop
        while state.total_num_evals < self.max_metric_calls:
            if self.display_progress_bar:
                delta = state.total_num_evals - last_pbar_val
                progress_bar.update(delta)
                last_pbar_val = state.total_num_evals

            assert state.is_consistent()
            try:
                state.save(self.run_dir)
                state.i += 1
                state.full_program_trace.append({"i": state.i})

                proposal = self.reflective_proposer.propose(state)
                if proposal is None:
                    print(f"[SIFTA] Iteration {state.i + 1}: Reflective mutation did not propose a new candidate")
                    self.logger.log(f"Iteration {state.i + 1}: Reflective mutation did not propose a new candidate")
                    continue

                # Acceptance: accept if score is strictly better, or if both are perfect
                old_scores = proposal.subsample_scores_before or []
                new_scores = proposal.subsample_scores_after or []
                old_sum = sum(old_scores)
                new_sum = sum(new_scores)

                # Check if both old and new have perfect scores (all 1.0)
                both_perfect = (
                        old_scores and new_scores and
                        all(s >= 1.0 for s in old_scores) and
                        all(s >= 1.0 for s in new_scores)
                )

                print(
                    f"[SIFTA] Iteration {state.i + 1}: Proposal scores - before: {old_scores} (sum={old_sum:.2f}), after: {new_scores} (sum={new_sum:.2f})")
                self.logger.log(
                    f"Iteration {state.i + 1}: Proposal scores - before: {old_scores} (sum={old_sum:.2f}), after: {new_scores} (sum={new_sum:.2f})")

                if new_sum > old_sum:
                    print(f"[SIFTA] Iteration {state.i + 1}: ACCEPTED ({new_sum:.2f} > {old_sum:.2f})")
                    self.logger.log(f"Iteration {state.i + 1}: Proposal ACCEPTED ({new_sum:.2f} > {old_sum:.2f})")
                elif both_perfect:
                    print(
                        f"[SIFTA] Iteration {state.i + 1}: ACCEPTED - both perfect scores ({new_sum:.2f} == {old_sum:.2f})")
                    self.logger.log(
                        f"Iteration {state.i + 1}: Proposal ACCEPTED - both perfect scores ({new_sum:.2f} == {old_sum:.2f})")
                elif new_sum == old_sum:
                    print(
                        f"[SIFTA] Iteration {state.i + 1}: ACCEPTED - equal scores, worth exploring ({new_sum:.2f} == {old_sum:.2f})")
                    self.logger.log(
                        f"Iteration {state.i + 1}: Proposal ACCEPTED - equal scores, worth exploring ({new_sum:.2f} == {old_sum:.2f})")
                else:
                    print(
                        f"[SIFTA] Iteration {state.i + 1}: REJECTED - new score worse ({new_sum:.2f} < {old_sum:.2f})")
                    self.logger.log(
                        f"Iteration {state.i + 1}: New subsample score is worse ({new_sum:.2f} < {old_sum:.2f}), rejecting proposal")
                    continue

                # Accept: full eval + add
                new_prog_idx, pareto_idx = self._run_full_eval_and_add(
                    new_program=proposal.candidate, state=state, parent_program_idx=proposal.parent_program_ids
                )
                print(
                    f"[SIFTA] Iteration {state.i + 1}: Added as program {new_prog_idx}, val scores: {state.program_full_scores_val_set}")
                print(
                    f"[SIFTA] Iteration {state.i + 1}: Current best program idx: {state.program_full_scores_val_set.index(max(state.program_full_scores_val_set))}")

            except Exception as e:
                self.logger.log(f"Iteration {state.i + 1}: Exception during optimization: {e}")
                self.logger.log(traceback.format_exc())
                if self.raise_on_exception:
                    raise e
                else:
                    continue

        # Close progress bar if it exists
        if self.display_progress_bar:
            progress_bar.close()

        state.save(self.run_dir)
        return state
