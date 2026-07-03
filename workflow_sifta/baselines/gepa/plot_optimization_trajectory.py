"""
Plot GEPA optimization trajectory from logged data.

Usage:
    python workflow_sifta/baselines/gepa/plot_optimization_trajectory.py \
        --run_dir optimization_runs/baseline_run_20260108_120000
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_round_data(run_dir: Path) -> list[dict]:
    """Load all round data from a run directory."""
    round_files = sorted(run_dir.glob("round_*.json"))
    rounds = []
    for round_file in round_files:
        with open(round_file, 'r') as f:
            rounds.append(json.load(f))
    return rounds


def plot_all_rounds_combined(run_dir: Path, save_path: Path = None):
    """Plot trajectories for all rounds in a single combined plot."""
    rounds = load_round_data(run_dir)

    if not rounds:
        print(f"No round data found in {run_dir}")
        return

    fig, ax = plt.subplots(figsize=(16, 8))

    # Color scheme for different rounds
    colors = plt.cm.tab10(np.linspace(0, 1, len(rounds)))

    # Track global iteration offset across rounds
    global_iteration_offset = 0
    all_global_iterations = []
    all_scores = []
    round_boundaries = []

    for round_idx, round_data in enumerate(rounds):
        round_num = round_data.get("round", round_idx)
        gepa_details = round_data.get("gepa_details", {})

        if not gepa_details or "history" not in gepa_details:
            print(f"Warning: No GEPA details for round {round_num}, skipping")
            continue

        color = colors[round_idx]

        # Extract data from GEPA history
        history = gepa_details["history"]
        scores = [entry.get("score") for entry in history if entry.get("score") is not None]

        if not scores:
            print(f"Warning: No scores in GEPA history for round {round_num}, skipping")
            continue

        # Create local iterations
        local_iterations = list(range(len(scores)))

        # Compute current best score at each iteration
        current_best_scores = []
        current_best = -float('inf')
        for score in scores:
            if score > current_best:
                current_best = score
            current_best_scores.append(current_best)

        # Convert to global iterations
        global_iterations = [it + global_iteration_offset for it in local_iterations]
        all_global_iterations.extend(global_iterations)
        all_scores.extend(current_best_scores)

        # Plot current best line for this round
        ax.plot(global_iterations, current_best_scores, '-', linewidth=2.5,
                color=color, label=f'Round {round_num + 1}', marker='o', markersize=3)

        # Plot actual scores as scatter
        ax.scatter(global_iterations, scores, c=[color], s=80, alpha=0.5, zorder=5)

        # Mark round boundary
        if round_idx < len(rounds) - 1:
            boundary_x = global_iterations[-1] + 0.5
            round_boundaries.append(boundary_x)

        # Update offset for next round
        global_iteration_offset += len(scores) + 2  # +2 for spacing between rounds

    # Add vertical lines at round boundaries
    for boundary_x in round_boundaries:
        ax.axvline(x=boundary_x, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)

    # Formatting
    ax.set_xlabel('Iteration (Across All Rounds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation Score', fontsize=14, fontweight='bold')
    ax.set_title('GEPA Optimization Trajectory - All Rounds', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add custom legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='gray', linewidth=2.5, marker='o', markersize=8, label='Current Best'),
        Line2D([0], [0], color='gray', linewidth=0, marker='o', markersize=8, alpha=0.5, label='Iteration Score'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Round Boundary'),
    ]

    # Combine with round legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + custom_lines, labels + [line.get_label() for line in custom_lines],
              loc='best', fontsize=10, ncol=2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined plot to {save_path}")
    else:
        plt.show()


def print_summary(run_dir: Path):
    """Print a summary of the optimization run."""
    rounds = load_round_data(run_dir)

    print(f"\n{'='*80}")
    print(f"GEPA Optimization Summary: {run_dir.name}")
    print(f"{'='*80}")

    for round_data in rounds:
        round_num = round_data.get("round", "?")
        score_before = round_data.get("score", 0.0)
        gepa_details = round_data.get("gepa_details", {})

        if gepa_details and "history" in gepa_details:
            num_iterations = len(gepa_details["history"])
            best_score = gepa_details.get("best_score", score_before)
            best_iteration = gepa_details.get("best_iteration", "?")

            print(f"\nRound {round_num + 1}:")
            print(f"  Initial score: {score_before:.4f}")
            print(f"  GEPA iterations: {num_iterations}")
            print(f"  Best iteration: {best_iteration}")
            print(f"  Best score: {best_score:.4f}")
            print(f"  Improvement: {100*(best_score - score_before):+.2f}%")
        else:
            print(f"\nRound {round_num + 1}: No GEPA details available")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Plot GEPA optimization trajectory")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to optimization run directory")
    parser.add_argument("--save", type=str, help="Path to save plot (if not specified, will display)")
    parser.add_argument("--summary", action="store_true", help="Print summary only (no plot)")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        return

    print_summary(run_dir)

    if not args.summary:
        save_path = Path(args.save) if args.save else None
        plot_all_rounds_combined(run_dir, save_path=save_path)


if __name__ == "__main__":
    main()
