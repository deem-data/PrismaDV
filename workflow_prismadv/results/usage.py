"""Token usage, LLM cost, generation time and LLM calls of the LLM-based methods on EIDBench-synth and
EIDBench-real: one record per generated script, then the same averaging as the F1 numbers.
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from prismadv.llm_backend.llm_pricing_info import price_tokens
from workflow_prismadv.results.alarm import load
from workflow_prismadv.results.benchmarks import real, synth
from workflow_prismadv.results.methods import prismadv, prompting
from workflow_prismadv.results.methods._common import REPLICATES
from workflow_prismadv.utils.replicates import replicate_of_artifact

# row key -> (artifact prefix, model in the artifact name, synth glob in general/constraints/<task>/)
ARMS = {k: (p, m, f"{p}--{m}--*") for k, (p, m) in {**prompting.ARTIFACT, **prismadv.ARTIFACT}.items()}
ARMS["codex"] = ("codex", "gpt-6-sol", "codex--gpt-6-sol--medium*")
FIELDS = ["input", "cached", "output", "reasoning", "calls", "usd_uncached", "usd_cached", "seconds"]
LOGGED_COST_TOLERANCE = 0.001   # relative
BENCHMARKS = ("synth", "real")


def _need(value, what: str, p: Path):
    if value is None:
        raise ValueError(f"{p} has no {what}")
    return value


def _price(model: str, inp: int, out: int, cached: int, p: Path) -> float:
    return _need(price_tokens(model, inp, out, cached), f"price for {model!r}", p)


def _zero() -> dict:
    return dict(input=0, cached=0, output=0, reasoning=0, calls=0, usd_uncached=0.0, usd_cached=0.0)


def _add_phases(r: dict, phases: dict, model: str, p: Path) -> None:
    """cost_summary phases (PrismaDV, zero-shot, few-shot): tokens and requests, cost repriced."""
    for name, c in _need(phases, "cost_summary", p).items():
        inp = _need(c.get("prompt_tokens"), f"{name}.prompt_tokens", p)
        cached = _need(c.get("cached_prompt_tokens"), f"{name}.cached_prompt_tokens", p)
        out = _need(c.get("completion_tokens"), f"{name}.completion_tokens", p)
        r["input"] += inp
        r["cached"] += cached
        r["output"] += out
        r["reasoning"] += _need(c.get("reasoning_tokens"), f"{name}.reasoning_tokens", p)
        r["calls"] += _need(c.get("requests"), f"{name}.requests", p)
        r["usd_uncached"] += _price(model, inp, out, 0, p)
        r["usd_cached"] += _price(model, inp, out, cached, p)


def _add_attempt(r: dict, t: dict, model: str, p: Path, what: str) -> None:
    """One codex run (token_accounting) or one failed attempt: tokens and calls, cost repriced."""
    inp = _need(t.get("input_tokens"), f"{what}.input_tokens", p)
    cached = _need(t.get("cached_input_tokens"), f"{what}.cached_input_tokens", p)
    out = _need(t.get("output_tokens"), f"{what}.output_tokens", p)
    r["input"] += inp
    r["cached"] += cached
    r["output"] += out
    r["reasoning"] += _need(t.get("reasoning_tokens"), f"{what}.reasoning_tokens", p)
    r["calls"] += _need(t.get("llm_calls"), f"{what}.llm_calls", p)
    r["usd_uncached"] += _price(model, inp, out, 0, p)
    r["usd_cached"] += _price(model, inp, out, cached, p)


def check_logged_cost(r: dict, uncached: float, cached: float, p: Path) -> None:
    """Use the logged cost, after checking that the repriced tokens give the same number."""
    for name, logged in (("usd_uncached", uncached), ("usd_cached", cached)):
        if abs(r[name] - logged) > LOGGED_COST_TOLERANCE * max(logged, 1e-9):
            raise ValueError(f"{p}: {name} repriced {r[name]:.6f} != logged {logged:.6f}")
        r[name] = logged


def synth_record(p: Path, a: dict, prefix: str, model: str) -> dict:
    r = _zero()
    if prefix == "codex":
        _add_attempt(r, _need(a.get("token_accounting"), "token_accounting", p), model, p, "token_accounting")
    else:
        _add_phases(r, a.get("cost_summary"), model, p)
    for i, f in enumerate(a.get("failed_attempts") or []):
        if prefix == "prismadv":
            raise ValueError(f"{p}: PrismaDV artifact with failed attempts (no reader for them)")
        _add_attempt(r, f, model, p, f"failed_attempts[{i}]")
    if prefix != "prismadv":
        c = _need(a.get("cost_all_attempts"), "cost_all_attempts", p)
        check_logged_cost(r, _need(c.get("usd_without_cache"), "cost_all_attempts.usd_without_cache", p),
                          _need(c.get("usd"), "cost_all_attempts.usd", p), p)
    r["seconds"] = _need(a.get("wall_clock_seconds"), "wall_clock_seconds", p)
    return r


def real_record(p: Path, a: dict, prefix: str, model: str) -> dict:
    r = _zero()
    if prefix == "codex":
        t = _need(a.get("trajectory"), "trajectory", p)
        _add_attempt(r, _need(t.get("token_accounting"), "trajectory.token_accounting", p), model, p,
                     "trajectory.token_accounting")
        failed = t.get("failed_attempts") or []
        for i, f in enumerate(failed):
            _add_attempt(r, f, model, p, f"trajectory.failed_attempts[{i}]")
        cached = _need(t.get("cost", {}).get("usd"), "trajectory.cost.usd", p)
        uncached = _need(t.get("cost_without_cache_usd"), "trajectory.cost_without_cache_usd", p)
        for i, f in enumerate(failed):
            cached += _need(f.get("usd"), f"trajectory.failed_attempts[{i}].usd", p)
            uncached += _need(f.get("usd_without_cache"), f"trajectory.failed_attempts[{i}].usd_without_cache", p)
        check_logged_cost(r, uncached, cached, p)
        r["seconds"] = _need(t.get("wall_clock_seconds_including_retries"),
                             "trajectory.wall_clock_seconds_including_retries", p)
        return r
    _add_phases(r, a.get("cost_summary"), model, p)
    for i, f in enumerate(a.get("failed_attempts") or []):
        if prefix == "prismadv":
            raise ValueError(f"{p}: PrismaDV artifact with failed attempts (no reader for them)")
        _add_attempt(r, f, model, p, f"failed_attempts[{i}]")
    if prefix == "prismadv":
        r["seconds"] = _need(a.get("wall_clock_seconds"), "wall_clock_seconds", p)
    else:  # zero-shot, few-shot: only the LLM call is timed on real
        r["seconds"] = _need(a["cost_summary"].get("constraint_generation", {}).get("elapsed_s"),
                             "cost_summary.constraint_generation.elapsed_s", p)
    return r


def _synth_dataset(ds: str) -> list[dict]:
    rows = []
    for t in synth.tasks(ds):
        for key, (prefix, model, glob) in ARMS.items():
            reps = {}
            for p in sorted((synth.general(ds) / "constraints" / t).glob(f"{glob}.yaml")):
                a = load(p)
                r = replicate_of_artifact(p, a)
                if r in REPLICATES:
                    reps[r] = (p, a)    # sorted by name, so the newest timestamp wins (as in methods._common)
            for r, (p, a) in reps.items():
                rows.append(dict(benchmark=synth.NAME, dataset=ds, unit=t, arm=key, rep=r, artifact=p.name,
                                 **synth_record(p, a, prefix, model)))
    return rows


def synth_records() -> pd.DataFrame:
    with Pool(len(synth.DATASETS)) as pool:
        parts = pool.map(_synth_dataset, synth.DATASETS)
    return pd.DataFrame([r for part in parts for r in part])


def real_records() -> pd.DataFrame:
    rows = []
    for ds in real.DATASETS:
        for key, (prefix, model, _) in ARMS.items():
            for r in REPLICATES:
                p = real.artifact(ds, prefix, model, r)
                if p is not None:
                    rows.append(dict(benchmark=real.NAME, dataset=ds, unit="project", arm=key, rep=r,
                                     artifact=p.name, **real_record(p, load(p), prefix, model)))
    return pd.DataFrame(rows)


def records(benchmarks=BENCHMARKS) -> pd.DataFrame:
    """One row per generated script; raises if an arm, replicate and dataset misses a script."""
    readers = {"synth": synth_records, "real": real_records}
    out = pd.concat([readers[b]() for b in benchmarks], ignore_index=True)
    check_complete(out, benchmarks)
    return out


def check_complete(rec: pd.DataFrame, benchmarks=BENCHMARKS) -> None:
    """Every arm and replicate needs a script for every unit, so no mean silently uses fewer scripts."""
    expected = {"synth": {ds: len(synth.tasks(ds)) for ds in synth.DATASETS},
                "real": {ds: 1 for ds in real.DATASETS}}
    got = rec.groupby(["benchmark", "arm", "rep", "dataset"]).size()
    problems = [f"{b} {k} r{r} {ds}: {got.get((b, k, r, ds), 0)}/{n} scripts"
                for b in benchmarks for k in ARMS for r in REPLICATES for ds, n in expected[b].items()
                if got.get((b, k, r, ds), 0) != n]
    if problems:
        raise ValueError("incomplete scripts:\n" + "\n".join(problems))


def per_replicate(rec: pd.DataFrame) -> pd.DataFrame:
    """Per (benchmark, arm, replicate): the mean per dataset, then the mean over the datasets; plus output_text."""
    rec = rec.assign(output_text=rec["output"] - rec["reasoning"])
    return (rec.groupby(["benchmark", "arm", "rep", "dataset"])[FIELDS + ["output_text"]].mean()
            .groupby(["benchmark", "arm", "rep"]).mean())


def summary(rec: pd.DataFrame) -> pd.DataFrame:
    """Per (benchmark, arm): mean and sample SD over replicates of every field, the replicate count, and
    cached_share = cached / input of the means."""
    g = per_replicate(rec).groupby(["benchmark", "arm"])
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    out["reps"] = g.size()
    out["cached_share"] = out["cached_mean"] / out["input_mean"]
    return out
