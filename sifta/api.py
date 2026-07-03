import random
from typing import Any

from sifta.adapters.default_adapter.default_adapter import DefaultAdapter
from sifta.core.adapter import DataInst, SIFTAAdapter, RolloutOutput, Trajectory
from sifta.core.engine import GEPAEngine
from sifta.core.result import SIFTAResult
from sifta.logging.logger import LoggerProtocol, StdOutLogger
from sifta.proposer.reflective_mutation.base import LanguageModel
from sifta.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from sifta.strategies.batch_sampler import EpochShuffledBatchSampler
from sifta.strategies.candidate_selector import CurrentBestCandidateSelector, ParetoCandidateSelector
from sifta.strategies.component_selector import RoundRobinReflectionComponentSelector


def optimize(
        seed_candidate: dict[str, str],
        trainset: list[DataInst],
        valset: list[DataInst] | None = None,
        adapter: SIFTAAdapter[DataInst, Trajectory, RolloutOutput] | None = None,
        task_lm: str | None = None,
        # Reflection-based configuration
        reflection_lm: LanguageModel | str | None = None,
        candidate_selection_strategy: str = "pareto",
        skip_perfect_score=True,
        reflection_minibatch_size=3,
        perfect_score=1,
        use_global_proposer: bool = False,
        # Budget
        max_metric_calls=None,
        # Logging
        logger: LoggerProtocol | None = None,
        run_dir: str | None = None,
        use_wandb: bool = False,
        wandb_api_key: str | None = None,
        wandb_init_kwargs: dict[str, Any] | None = None,
        track_best_outputs: bool = False,
        display_progress_bar: bool = False,
        # Reproducibility
        seed: int = 0,
        raise_on_exception: bool = True,
        # Custom batch sampler configuration
        workflow=None,
        llm_name: str | None = None,
        batch_sampler_temperature: float = 0.5,
        batch_sampler_strategy: str = "random",
):
    """
    SIFTA can be applied to optimize any system that uses text components (e.g., prompts in a AI system, code snippets/code files/functions/classes in a codebase, etc.).
    In order for SIFTA to plug into your system's environment, SIFTA requires an adapter, `GEPAAdapter` to be implemented. The adapter is responsible for:
    1. Evaluating a proposed candidate on a batch of inputs.
       - The adapter receives a candidate proposed by SIFTA, along with a batch of inputs selected from the training/validation set.
       - The adapter instantiates the system with the texts proposed in the candidate.
       - The adapter then evaluates the candidate on the batch of inputs, and returns the scores.
       - The adapter should also capture relevant information from the execution of the candidate, like system and evaluation traces.
    2. Identifying textual information relevant to a component of the candidate
       - Given the trajectories captured during the execution of the candidate, SIFTA selects a component of the candidate to update.
       - The adapter receives the candidate, the batch of inputs, and the trajectories captured during the execution of the candidate.
       - The adapter is responsible for identifying the textual information relevant to the component to update.
       - This information is used by SIFTA to reflect on the performnace of the component, and propose new component texts.

    At each iteration, SIFTA proposes a new candidate using one of the following strategies:
    1. Reflective mutation: SIFTA proposes a new candidate by mutating the current candidate, leveraging rich textual feedback.
    2. Merge: SIFTA proposes a new candidate by merging 2 candidates that are on the Pareto frontier.

    SIFTA also tracks the Pareto frontier of performance achieved by different candidates on the validation set. This way, it can leverage candidates that
    work well on a subset of inputs to improve the system's performance on the entire validation set, by evolving from the Pareto frontier.

    Parameters:
    - seed_candidate: The initial candidate to start with.
    - trainset: The training set to use for reflective updates.
    - valset: The validation set to use for tracking Pareto scores. If not provided, SIFTA will use the trainset for both.
    - adapter: A `GEPAAdapter` instance that implements the adapter interface. This allows SIFTA to plug into your system's environment. If not provided, SIFTA will use a default adapter: `gepa.adapters.default_adapter.default_adapter.DefaultAdapter`, with model defined by `task_lm`.
    - task_lm: Optional. The model to use for the task. This is only used if `adapter` is not provided, and is used to initialize the default adapter.

    # Reflection-based configuration
    - reflection_lm: A `LanguageModel` instance that is used to reflect on the performance of the candidate program.
    - candidate_selection_strategy: The strategy to use for selecting the candidate to update.
    - skip_perfect_score: Whether to skip updating the candidate if it achieves a perfect score on the minibatch.
    - reflection_minibatch_size: The number of examples to use for reflection in each proposal step.
    - perfect_score: The perfect score to achieve.

    # Merge-based configuration
    - use_merge: Whether to use the merge strategy.
    - max_merge_invocations: The maximum number of merge invocations to perform.

    # Budget
    - max_metric_calls: The maximum number of metric calls to perform.

    # Logging
    - logger: A `LoggerProtocol` instance that is used to log the progress of the optimization.
    - run_dir: The directory to save the results to.
    - use_wandb: Whether to use Weights and Biases to log the progress of the optimization.
    - wandb_api_key: The API key to use for Weights and Biases.
    - wandb_init_kwargs: Additional keyword arguments to pass to the Weights and Biases initialization.
    - track_best_outputs: Whether to track the best outputs on the validation set. If True, GEPAResult will contain the best outputs obtained for each task in the validation set.

    # Reproducibility
    - seed: The seed to use for the random number generator.

    # Custom batch sampler configuration (PrismaDV-specific)
    - workflow: OptimizationWorkflow instance for calculating fail precision scores.
    - llm_name: LLM name used for trajectory creation in fail precision calculation.
    - batch_sampler_temperature: Temperature for softmax sampling (0.0 = deterministic, higher = more uniform).
    - batch_sampler_strategy: Strategy for sampling training examples:
        - "best_first": Prioritize highest-scoring examples
        - "worst_first": Prioritize lowest-scoring examples (default)
        - "extreme_first": Prioritize most extreme scores (furthest from 0.5)
        - "random": Uniform random sampling
    """
    if adapter is None:
        assert task_lm is not None, (
            "Since no adapter is provided, SIFTA requires a task LM to be provided. Please set the `task_lm` parameter."
        )
        adapter = DefaultAdapter(model=task_lm)
    else:
        assert task_lm is None, (
            "Since an adapter is provided, SIFTA does not require a task LM to be provided. Please set the `task_lm` parameter to None."
        )

    assert max_metric_calls is not None, "max_metric_calls must be set"
    assert reflection_lm is not None, (
        "SIFTA currently requires a reflection LM to be provided. We will soon support simpler application without specifying a reflection LM."
    )

    if isinstance(reflection_lm, str):
        import litellm

        reflection_lm_name = reflection_lm
        reflection_lm = (
            lambda prompt: litellm.completion(model=reflection_lm_name, messages=[{"role": "user", "content": prompt}])
            .choices[0]
            .message.content
        )

    if logger is None:
        logger = StdOutLogger()

    if valset is None:
        valset = trainset

    rng = random.Random(seed)
    candidate_selector = (
        ParetoCandidateSelector(rng=rng) if candidate_selection_strategy == "pareto" else CurrentBestCandidateSelector()
    )
    module_selector = RoundRobinReflectionComponentSelector()

    # Use custom FailPrecisionBatchSampler if workflow and llm_name provided
    if workflow is not None and llm_name is not None:
        from sifta.samplers.fail_precision_batch_sampler import FailPrecisionBatchSampler
        batch_sampler = FailPrecisionBatchSampler(
            trainset=trainset,
            workflow=workflow,
            llm_name=llm_name,
            minibatch_size=reflection_minibatch_size,
            rng=rng,
            seed_candidate=seed_candidate,
            precompute_scores=True,
            temperature=batch_sampler_temperature,
            sampling_strategy=batch_sampler_strategy,
        )
    else:
        # Fallback to original sampler if workflow/llm_name not provided
        batch_sampler = EpochShuffledBatchSampler(minibatch_size=reflection_minibatch_size, rng=rng)

    reflective_proposer = ReflectiveMutationProposer(
        logger=logger,
        trainset=trainset,
        adapter=adapter,
        candidate_selector=candidate_selector,
        module_selector=module_selector,
        batch_sampler=batch_sampler,
        perfect_score=perfect_score,
        skip_perfect_score=skip_perfect_score,
        use_wandb=use_wandb,
        reflection_lm=reflection_lm,
        use_global_proposer=use_global_proposer,
    )

    def evaluator(inputs, prog):
        eval_out = adapter.evaluate(inputs, prog, capture_traces=False)
        return eval_out.outputs, eval_out.scores

    engine = GEPAEngine(
        run_dir=run_dir,
        evaluator=evaluator,
        valset=valset,
        seed_candidate=seed_candidate,
        max_metric_calls=max_metric_calls,
        perfect_score=perfect_score,
        seed=seed,
        reflective_proposer=reflective_proposer,
        logger=logger,
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        wandb_init_kwargs=wandb_init_kwargs,
        track_best_outputs=track_best_outputs,
        display_progress_bar=display_progress_bar,
        raise_on_exception=raise_on_exception,
    )
    state = engine.run()
    result = SIFTAResult.from_state(state)
    return result
