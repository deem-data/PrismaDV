import inspect
import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional, Protocol, Union

from dspy.clients.lm import LM
from dspy.primitives import Example, Module, Prediction
from dspy.teleprompt.teleprompt import Teleprompter

if TYPE_CHECKING:
    from sifta import SIFTAResult

    from sifta.dspy_sifta.teleprompt.sifta.sifta_utils import DspyAdapter, DSPyTrace, PredictorFeedbackFn, \
        ScoreWithFeedback

logger = logging.getLogger(__name__)

AUTO_RUN_SETTINGS = {
    "light": {"n": 6},
    "medium": {"n": 12},
    "heavy": {"n": 18},
}


class SIFTAFeedbackMetric(Protocol):
    def __call__(
            gold: Example,
            pred: Prediction,
            trace: Optional["DSPyTrace"],
            pred_name: str | None,
            pred_trace: Optional["DSPyTrace"],
    ) -> Union[float, "ScoreWithFeedback"]:
        """
        This function is called with the following arguments:
        - gold: The gold example.
        - pred: The predicted output.
        - trace: Optional. The trace of the program's execution.
        - pred_name: Optional. The name of the target predictor currently being optimized by SIFTA, for which
            the feedback is being requested.
        - pred_trace: Optional. The trace of the target predictor's execution SIFTA is seeking feedback for.

        Note the `pred_name` and `pred_trace` arguments. During optimization, SIFTA will call the metric to obtain
        feedback for individual predictors being optimized. SIFTA provides the name of the predictor in `pred_name`
        and the sub-trace (of the trace) corresponding to the predictor in `pred_trace`.
        If available at the predictor level, the metric should return dspy.Prediction(score: float, feedback: str) corresponding 
        to the predictor.
        If not available at the predictor level, the metric can also return a text feedback at the program level
        (using just the gold, pred and trace).
        If no feedback is returned, SIFTA will use a simple text feedback consisting of just the score:
        f"This trajectory got a score of {score}."
        """
        ...


@dataclass(frozen=True)
class DspySIFTAResult:
    """
    Additional data related to the SIFTA run.

    Fields:
    - candidates: list of proposed candidates (component_name -> component_text)
    - parents: lineage info; for each candidate i, parents[i] is a list of parent indices or None
    - val_aggregate_scores: per-candidate aggregate score on the validation set (higher is better)
    - val_subscores: per-candidate per-instance scores on the validation set (len == num_val_instances)
    - per_val_instance_best_candidates: for each val instance t, a set of candidate indices achieving the best score on t
    - discovery_eval_counts: Budget (number of metric calls / rollouts) consumed up to the discovery of each candidate

    - total_metric_calls: total number of metric calls made across the run
    - num_full_val_evals: number of full validation evaluations performed
    - log_dir: where artifacts were written (if any)
    - seed: RNG seed for reproducibility (if known)

    - best_idx: candidate index with the highest val_aggregate_scores
    - best_candidate: the program text mapping for best_idx
    """
    # Data about the proposed candidates
    candidates: list[Module]
    parents: list[list[int | None]]
    val_aggregate_scores: list[float]
    val_subscores: list[list[float]]
    per_val_instance_best_candidates: list[set[int]]
    discovery_eval_counts: list[int]

    # Optional data
    best_outputs_valset: list[list[tuple[int, list[Prediction]]]] | None = None

    # Optimization metadata
    total_metric_calls: int | None = None
    num_full_val_evals: int | None = None
    log_dir: str | None = None
    seed: int | None = None

    @property
    def best_idx(self) -> int:
        scores = self.val_aggregate_scores
        return max(range(len(scores)), key=lambda i: scores[i])

    @property
    def best_candidate(self) -> dict[str, str]:
        return self.candidates[self.best_idx]

    @property
    def highest_score_achieved_per_val_task(self) -> list[float]:
        return [
            self.val_subscores[list(self.per_val_instance_best_candidates[val_idx])[0]][val_idx]
            for val_idx in range(len(self.val_subscores[0]))
        ]

    def to_dict(self) -> dict[str, Any]:
        cands = [
            {k: v for k, v in cand.items()}
            for cand in self.candidates
        ]

        return dict(
            candidates=cands,
            parents=self.parents,
            val_aggregate_scores=self.val_aggregate_scores,
            best_outputs_valset=self.best_outputs_valset,
            val_subscores=self.val_subscores,
            per_val_instance_best_candidates=[list(s) for s in self.per_val_instance_best_candidates],
            discovery_eval_counts=self.discovery_eval_counts,
            total_metric_calls=self.total_metric_calls,
            num_full_val_evals=self.num_full_val_evals,
            log_dir=self.log_dir,
            seed=self.seed,
            best_idx=self.best_idx,
        )

    def get_iteration_info(self) -> dict:
        """
        Get iteration-level tracking information.
        Delegates to the underlying SIFTAResult if available.
        """
        if hasattr(self, '_sifta_result'):
            return self._sifta_result.get_iteration_info()
        return {
            "candidate_to_iteration": {0: -1},
            "iteration_to_candidate": {},
            "rejected_iterations": [],
            "num_iterations": 0,
        }

    @staticmethod
    def from_sifta_result(sifta_result: "SIFTAResult", adapter: "DspyAdapter") -> "DspySIFTAResult":
        dspy_result = DspySIFTAResult(
            candidates=[adapter.build_program(c) for c in sifta_result.candidates],
            parents=sifta_result.parents,
            val_aggregate_scores=sifta_result.val_aggregate_scores,
            best_outputs_valset=sifta_result.best_outputs_valset,
            val_subscores=sifta_result.val_subscores,
            per_val_instance_best_candidates=sifta_result.per_val_instance_best_candidates,
            discovery_eval_counts=sifta_result.discovery_eval_counts,
            total_metric_calls=sifta_result.total_metric_calls,
            num_full_val_evals=sifta_result.num_full_val_evals,
            log_dir=sifta_result.run_dir,
            seed=sifta_result.seed,
        )
        # Store reference to original sifta_result for iteration tracking
        object.__setattr__(dspy_result, '_sifta_result', sifta_result)
        return dspy_result


class SIFTA(Teleprompter):
    def __init__(
            self,
            metric: SIFTAFeedbackMetric,
            *,
            # Budget configuration
            auto: Literal["light", "medium", "heavy"] | None = None,
            max_full_evals: int | None = None,
            max_metric_calls: int | None = None,
            # Reflection based configuration
            reflection_minibatch_size: int = 3,
            candidate_selection_strategy: Literal["pareto", "current_best"] = "pareto",
            reflection_lm: LM | None = None,
            skip_perfect_score: bool = True,
            add_format_failure_as_feedback: bool = False,
            use_global_proposer: bool = False,
            # Evaluation configuration
            num_threads: int | None = None,
            failure_score: float = 0.0,
            perfect_score: float = 1.0,
            # Logging
            log_dir: str = None,
            track_stats: bool = False,
            use_wandb: bool = False,
            wandb_api_key: str | None = None,
            wandb_init_kwargs: dict[str, Any] | None = None,
            track_best_outputs: bool = False,
            # Reproducibility
            seed: int | None = 0,
            # Custom batch sampler configuration
            workflow=None,
            llm_name: str | None = None,
            batch_sampler_temperature: float = 0.5,
            batch_sampler_strategy: Literal["best_first", "worst_first", "extreme_first", "random"] = "random",
    ):
        try:
            inspect.signature(metric).bind(None, None, None, None, None)
        except TypeError as e:
            raise TypeError(
                "SIFTA metric must accept five arguments: (gold, pred, trace, pred_name, pred_trace). "
            ) from e

        self.metric_fn = metric

        # Budget configuration
        assert (
                (max_metric_calls is not None) +
                (max_full_evals is not None) +
                (auto is not None)
                == 1
        ), (
            "Exactly one of max_metric_calls, max_full_evals, auto must be set. "
            f"You set max_metric_calls={max_metric_calls}, "
            f"max_full_evals={max_full_evals}, "
            f"auto={auto}."
        )
        self.auto = auto
        self.max_full_evals = max_full_evals
        self.max_metric_calls = max_metric_calls

        # Reflection based configuration
        self.reflection_minibatch_size = reflection_minibatch_size
        self.candidate_selection_strategy = candidate_selection_strategy
        # self.reflection_lm = reflection_lm
        assert reflection_lm is not None, "SIFTA requires a reflection language model to be provided. Typically, you can use `dspy.LM(model='gpt-5', temperature=1.0, max_tokens=32000)` to get a good reflection model. Reflection LM is used by SIFTA to reflect on the behavior of the program and propose new instructions, and will benefit from a strong model."
        self.reflection_lm = lambda x: reflection_lm(x)[0]
        self.skip_perfect_score = skip_perfect_score
        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        self.use_global_proposer = use_global_proposer

        # Evaluation Configuration
        self.num_threads = num_threads
        self.failure_score = failure_score
        self.perfect_score = perfect_score

        # Logging configuration
        self.log_dir = log_dir
        self.track_stats = track_stats
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key
        self.wandb_init_kwargs = wandb_init_kwargs

        if track_best_outputs:
            assert track_stats, "track_stats must be True if track_best_outputs is True."
        self.track_best_outputs = track_best_outputs

        # Reproducibility
        self.seed = seed

        # Custom batch sampler configuration
        self.workflow = workflow
        self.llm_name = llm_name
        self.batch_sampler_temperature = batch_sampler_temperature
        self.batch_sampler_strategy = batch_sampler_strategy

    def auto_budget(self, num_preds, num_candidates, valset_size: int, minibatch_size: int = 35,
                    full_eval_steps: int = 5) -> int:
        import numpy as np
        num_trials = int(max(2 * (num_preds * 2) * np.log2(num_candidates), 1.5 * num_candidates))
        if num_trials < 0 or valset_size < 0 or minibatch_size < 0:
            raise ValueError("num_trials, valset_size, and minibatch_size must be >= 0.")
        if full_eval_steps < 1:
            raise ValueError("full_eval_steps must be >= 1.")

        V = valset_size
        N = num_trials
        M = minibatch_size
        m = full_eval_steps

        # Initial full evaluation on the default program
        total = V

        # Assume upto 5 trials for bootstrapping each candidate
        total += num_candidates * 5

        # N minibatch evaluations
        total += N * M
        if N == 0:
            return total  # no periodic/full evals inside the loop
        # Periodic full evals occur when trial_num % (m+1) == 0, where trial_num runs 2..N+1
        periodic_fulls = (N + 1) // (m) + 1
        # If 1 <= N < m, the code triggers one final full eval at the end
        extra_final = 1 if N < m else 0

        total += (periodic_fulls + extra_final) * V
        return total

    def compile(
            self,
            student: Module,
            *,
            trainset: list[Example],
            teacher: Module | None = None,
            valset: list[Example] | None = None,
    ) -> Module:
        """
        SIFTA uses the trainset to perform reflective updates to the prompt, but uses the valset for tracking Pareto scores.
        If no valset is provided, SIFTA will use the trainset for both.

        Parameters:
        - student: The student module to optimize.
        - trainset: The training set to use for reflective updates.
        - valset: The validation set to use for tracking Pareto scores. If not provided, SIFTA will use the trainset for both.
        """
        from sifta.api import optimize
        from sifta import SIFTAResult

        from sifta.dspy_sifta.teleprompt.sifta.sifta_utils import DspyAdapter, LoggerAdapter

        assert trainset is not None and len(trainset) > 0, "Trainset must be provided and non-empty"
        assert teacher is None, "Teacher is not supported."

        if self.auto is not None:
            self.max_metric_calls = self.auto_budget(
                num_preds=len(student.predictors()),
                num_candidates=AUTO_RUN_SETTINGS[self.auto]["n"],
                valset_size=len(valset) if valset is not None else len(trainset),
            )
        elif self.max_full_evals is not None:
            self.max_metric_calls = self.max_full_evals * (len(trainset) + (len(valset) if valset is not None else 0))
        else:
            assert self.max_metric_calls is not None, "Either auto, max_full_evals, or max_metric_calls must be set."

        logger.info(
            f"Running SIFTA for approx {self.max_metric_calls} metric calls of the program. This amounts to {self.max_metric_calls / len(trainset) if valset is None else self.max_metric_calls / (len(trainset) + len(valset)):.2f} full evals on the {'train' if valset is None else 'train+val'} set.")

        valset = valset or trainset
        logger.info(
            f"Using {len(valset)} examples for tracking Pareto scores. You can consider using a smaller sample of the valset to allow SIFTA to explore more diverse solutions within the same budget.")

        rng = random.Random(self.seed)

        def feedback_fn_creator(pred_name: str, predictor) -> "PredictorFeedbackFn":
            def feedback_fn(
                    predictor_output: dict[str, Any],
                    predictor_inputs: dict[str, Any],
                    module_inputs: Example,
                    module_outputs: Prediction,
                    captured_trace: "DSPyTrace",
            ) -> "ScoreWithFeedback":
                trace_for_pred = [(predictor, predictor_inputs, predictor_output)]
                o = self.metric_fn(
                    module_inputs,
                    module_outputs,
                    captured_trace,
                    pred_name,
                    trace_for_pred,
                )
                if hasattr(o, "feedback"):
                    if o["feedback"] is None:
                        o["feedback"] = f"This trajectory got a score of {o['score']}."
                    return o
                else:
                    return dict(score=o, feedback=f"This trajectory got a score of {o}.")

            return feedback_fn

        feedback_map = {
            k: feedback_fn_creator(k, v)
            for k, v in student.named_predictors()
        }

        # Build the DSPy adapter that encapsulates evaluation, trace capture, feedback extraction, and instruction proposal
        adapter = DspyAdapter(
            student_module=student,
            metric_fn=self.metric_fn,
            feedback_map=feedback_map,
            failure_score=self.failure_score,
            num_threads=self.num_threads,
            add_format_failure_as_feedback=self.add_format_failure_as_feedback,
            rng=rng,
        )

        reflection_lm = self.reflection_lm

        # Instantiate SIFTA with the simpler adapter-based API
        base_program = {name: pred.signature.instructions for name, pred in student.named_predictors()}
        sifta_result: SIFTAResult = optimize(
            seed_candidate=base_program,
            trainset=trainset,
            valset=valset,
            adapter=adapter,

            # Reflection-based configuration
            reflection_lm=reflection_lm,
            candidate_selection_strategy=self.candidate_selection_strategy,
            skip_perfect_score=self.skip_perfect_score,
            reflection_minibatch_size=self.reflection_minibatch_size,
            use_global_proposer=self.use_global_proposer,

            perfect_score=self.perfect_score,
            # Budget
            max_metric_calls=self.max_metric_calls,

            # Logging
            logger=LoggerAdapter(logger),
            run_dir=self.log_dir,
            use_wandb=self.use_wandb,
            wandb_api_key=self.wandb_api_key,
            wandb_init_kwargs=self.wandb_init_kwargs,
            track_best_outputs=self.track_best_outputs,
            display_progress_bar=True,
            raise_on_exception=True,

            # Reproducibility
            seed=self.seed,

            # Custom batch sampler configuration
            workflow=self.workflow,
            llm_name=self.llm_name,
            batch_sampler_temperature=self.batch_sampler_temperature,
            batch_sampler_strategy=self.batch_sampler_strategy,
        )

        new_prog = adapter.build_program(sifta_result.best_candidate)

        if self.track_stats:
            dspy_sifta_result = DspySIFTAResult.from_sifta_result(sifta_result, adapter)
            new_prog.detailed_results = dspy_sifta_result

        return new_prog
