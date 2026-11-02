"""The optimization budget: K validation passes of search per arm.

One evaluation is one run of an arm's program on one example of its own granularity: the column program
(ConstraintGenerationModule or HumanDesignedConstraintGenerationModule) on one (script, column) example, or the
script-level GEPA program on one script. An arm may spend K x |its validation set| evaluations on search, so every arm
gets the same number of validation passes whatever its granularity.

Charged are the evaluations an optimizer engine makes while it runs (engine.run): its evaluation of the start program,
the minibatches and the full validations of the search loop. Evaluations outside the engine are counted and recorded
but not charged: the drivers' initial evaluation and, for SIFTA, the condense pass, the evaluation of the current
program per round and the batch sampler's precompute.

An optimizer with several rounds (SIFTA) has one engine run per round; start_round(passes) gives the next engine run
its own cap. The total of K passes applies on top.

The count comes from a DSPy callback, which sees every module call in every thread and in every copy of the program.
An engine starts an iteration only if it fits: both engines (upstream GEPA 0.0.7 and SIFTA's fork of it) test
`state.total_num_evals < self.max_metric_calls` before each iteration, and the engine class installed here turns
`max_metric_calls` into 0 when the most expensive iteration so far would no longer fit (before the first iteration of
an engine run: one accepted iteration, i.e. two minibatches and the val set). A run ends at most one iteration below
its cap, and above it only if an iteration costs more than every earlier one; the summary records the exact counts.

Usage in a driver, once its validation set is known and before any evaluation of the optimization stage:
    budget = install(passes, len(valset))   # counts from now on; stops the engines (passes None: count only)
"""
from __future__ import annotations

import inspect
import threading

import dspy
from dspy.utils.callback import BaseCallback

from prismadv.llm.dspy.models.column_wise_module import (ConstraintGenerationModule,
                                                         HumanDesignedConstraintGenerationModule)

LOOP_TEST = "while state.total_num_evals < self.max_metric_calls:"
# The metric-call cap handed to an engine whose run this budget ends instead.
UNBOUNDED_METRIC_CALLS = 10 ** 9


class EvaluationBudget(BaseCallback):
    def __init__(self, passes: int | None, val_size: int, minibatch_size: int = 3):
        self.passes = passes
        self.val_size = val_size
        self.limit = None if passes is None else passes * val_size   # on charged evaluations
        self.used = 0                  # all evaluations
        self.charged = 0               # evaluations made while an engine runs
        # An accepted iteration evaluates two minibatches (parent and candidate) and then the val set: the estimate
        # of an iteration before any has run.
        self.first_step = val_size + 2 * minibatch_size
        self.engine_runs: list[dict] = []
        self._next_run_passes = None
        self._active = False
        self._run_base = 0
        self._run_limit = None
        self.largest_step = 0
        self._last_boundary = None
        self._lock = threading.Lock()

    @property
    def exhausted(self) -> bool:
        return self.limit is not None and self.charged >= self.limit

    def start_round(self, passes: int | None) -> None:
        """Cap the next engine run at `passes` validation passes."""
        self._next_run_passes = passes

    def begin_engine_run(self) -> None:
        self._active = True
        self._run_base = self.charged
        p = self._next_run_passes
        self._run_limit = None if p is None else self._run_base + p * self.val_size
        self._next_run_passes = None
        self.largest_step, self._last_boundary = 0, None

    def end_engine_run(self) -> None:
        self._active = False
        self.engine_runs.append({"charged": self.charged - self._run_base,
                                 "cap": None if self._run_limit is None else self._run_limit - self._run_base})

    def stop_before_iteration(self) -> bool:
        """Called where an engine decides whether to start another iteration: start it only if the most expensive
        step so far still fits in the total budget and in this engine run's cap."""
        if self._last_boundary is None:
            self.largest_step = self.first_step
        else:
            self.largest_step = max(self.largest_step, self.charged - self._last_boundary)
        self._last_boundary = self.charged
        limits = [x for x in (self.limit, self._run_limit) if x is not None]
        return bool(limits) and self.charged + self.largest_step > min(limits)

    def on_module_start(self, call_id, instance, inputs):
        from workflow_sifta.baselines.gepa.script_level_module import ScriptLevelConstraintGenerationModule

        if isinstance(instance, (ConstraintGenerationModule, HumanDesignedConstraintGenerationModule,
                                 ScriptLevelConstraintGenerationModule)):
            with self._lock:
                self.used += 1
                if self._active:
                    self.charged += 1


_BUDGET: EvaluationBudget | None = None


def _budgeted(engine_cls):
    """The engine class whose run is charged, with `max_metric_calls` read as 0 once the next iteration would not
    fit."""
    if LOOP_TEST not in inspect.getsource(engine_cls.run):
        raise RuntimeError(f"{engine_cls.__module__}.{engine_cls.__name__}.run no longer tests '{LOOP_TEST}'; "
                           f"the budget cannot stop it")

    class Budgeted(engine_cls):
        def run(self):
            if _BUDGET is None:
                return super().run()
            _BUDGET.begin_engine_run()
            try:
                return super().run()
            finally:
                _BUDGET.end_engine_run()

        # Read only where the engine decides to start an iteration (the loop test, and once for the progress bar
        # before the loop), so each read is an iteration boundary.
        @property
        def max_metric_calls(self):
            stop = _BUDGET is not None and _BUDGET.stop_before_iteration()
            return 0 if stop else self._max_metric_calls

        @max_metric_calls.setter
        def max_metric_calls(self, value):
            self._max_metric_calls = value

    Budgeted.__name__ = f"Budgeted{engine_cls.__name__}"
    return Budgeted


def install(passes: int | None, val_size: int) -> EvaluationBudget:
    """Start counting evaluations and stop GEPA and SIFTA engines at passes x val_size charged evaluations
    (passes None: count only)."""
    global _BUDGET
    import gepa.api
    import sifta.api

    _BUDGET = EvaluationBudget(passes, val_size)
    for api in (gepa.api, sifta.api):
        if not api.GEPAEngine.__name__.startswith("Budgeted"):
            api.GEPAEngine = _budgeted(api.GEPAEngine)
    others = [c for c in dspy.settings.callbacks if not isinstance(c, EvaluationBudget)]
    dspy.settings.configure(callbacks=[*others, _BUDGET])
    return _BUDGET


def current() -> EvaluationBudget | None:
    return _BUDGET


def summary_fields() -> dict:
    """What a run's summary records about its budget."""
    b = _BUDGET
    if b is None:
        return {"budget_validation_passes": None}
    passes = (lambda n: round(n / b.val_size, 3)) if b.val_size else (lambda n: None)
    return {"budget_validation_passes": b.passes, "val_size": b.val_size, "budget_evaluations": b.limit,
            "evaluations_charged": b.charged, "validation_passes_charged": passes(b.charged),
            "evaluations_total": b.used, "evaluations_not_charged": b.used - b.charged,
            "engine_runs": b.engine_runs}


def engine_budget_kwargs(passes: int | None, default: dict) -> dict:
    """An optimizer's budget arguments: under this budget the engine gets no metric-call cap of its own."""
    return {"max_metric_calls": UNBOUNDED_METRIC_CALLS} if passes is not None else default
