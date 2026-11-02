"""Cost with the provider's prompt cache, runtime and LLM calls vs macro F1 on EIDBench-synth.
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workflow_prismadv.results import methods, usage
from workflow_prismadv.results.metrics import per_replicate

ARMS = {"P-6sol": "PrismaDV", "codex": "Codex", "single-6sol": "Zero-shot", "few-6sol": "Few-shot"}
FAMILIES = [methods.prismadv, methods.codex, methods.prompting]
SHAPES = {"PrismaDV": "X", "Codex": "o", "Zero-shot": "s", "Few-shot": "^"}
FILL, EDGE, BAR = "#2a78d6", "#104281", "#8f8e88"
YLIM = (0.63, 0.78)
XLABEL_SIZE = 16
matplotlib.rcdefaults()
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def points() -> pd.DataFrame:
    """Per arm: mean and SD over replicates of macro F1, cost, runtime and LLM calls."""
    out = methods.outcomes(FAMILIES, benchmarks=("synth",))
    f1 = per_replicate(out[out.arm.isin(ARMS)]).loc["synth"][["f1"]].reset_index()
    rec = usage.records(("synth",))
    x = (usage.per_replicate(rec[rec.arm.isin(ARMS)]).loc["synth"][["usd_cached", "seconds", "calls"]]
         .rename(columns={"usd_cached": "usd"}).reset_index())
    g = f1.merge(x, on=["arm", "rep"], validate="one_to_one").groupby("arm")[["f1", "usd", "seconds", "calls"]]
    p = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).loc[list(ARMS)]
    p["method"] = [ARMS[k] for k in p.index]
    return p


def draw(ax, p: pd.DataFrame, col: str) -> None:
    for r in p.itertuples():
        x, xs = getattr(r, f"{col}_mean"), getattr(r, f"{col}_sd")
        ax.errorbar(x, r.f1_mean, xerr=[[min(xs, x * 0.95)], [xs]], yerr=r.f1_sd,
                    fmt="none", ecolor=BAR, elinewidth=1.6, capsize=4, capthick=1.6, zorder=1)
        ax.scatter(x, r.f1_mean, s=300, linewidths=2.6, zorder=3, marker=SHAPES[r.method], facecolor=FILL,
                   edgecolor=EDGE)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.grid(True, linestyle=(0, (5, 4)), color="#bdbdbd", linewidth=1.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=18)


def log_limits(p: pd.DataFrame, col: str, pad: float = 1.8) -> tuple[float, float]:
    lo = (p[f"{col}_mean"] - p[f"{col}_sd"]).clip(lower=p[f"{col}_mean"] * 0.05).min()
    hi = (p[f"{col}_mean"] + p[f"{col}_sd"]).max()
    return lo / pad, hi * pad


def plot(p: pd.DataFrame, out: Path | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    labels = {"usd": "← Cost per task, cached (USD)", "seconds": "← Generation time per task (s)",
              "calls": "← LLM calls per task"}
    for ax, (col, label) in zip(axes, labels.items()):
        draw(ax, p, col)
        ax.set_ylim(YLIM)
        ax.set_xlim(*log_limits(p, col))
        fmt = (lambda v, _: f"${v:g}") if col == "usd" else (lambda v, _: f"{v:g}")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
        ax.set_xlabel(label, fontsize=XLABEL_SIZE)
    axes[0].set_ylabel("F1 score →", fontsize=20)
    handles = [plt.Line2D([], [], linestyle="", markersize=13, markeredgewidth=2.4, marker=mk, markerfacecolor=FILL,
                          markeredgecolor=EDGE, label=m) for m, mk in SHAPES.items()]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=16, bbox_to_anchor=(0.52, 1.0),
               columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    if out:
        fig.savefig(out, dpi=300)
    return fig


if __name__ == "__main__":
    plot(points(), Path(__file__).with_suffix(".pdf"))
