import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def performance_chart(
    results: Dict[str, Dict],
    title: str = "Performance Chart",
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for idx, (solver, data) in enumerate(results.items()):
        avg_gap = data.get("mean_gap", 0)
        avg_time = data.get("mean_time", 0)
        ax.scatter(
            avg_time,
            avg_gap,
            marker=markers[idx % len(markers)],
            color=colors[idx],
            s=100,
            label=solver,
            zorder=5,
        )
        if data.get("gaps") and data.get("times"):
            gaps, times = np.array(data["gaps"]), np.array(data["times"])
            if gaps.size == times.size:
                ax.scatter(
                    times,
                    gaps,
                    marker=markers[idx % len(markers)],
                    color=colors[idx],
                    s=20,
                    alpha=0.3,
                    zorder=3,
                )
    ax.set_xlabel("Average Normalized Time (PassMark)")
    ax.set_ylabel("Average Gap (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def convergence_profile(
    convergence_data: Dict[str, List[float]],
    title: str = "Convergence Profile",
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    for solver, gaps_over_time in convergence_data.items():
        times = np.arange(len(gaps_over_time))
        ax.plot(times, gaps_over_time, label=solver, linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Average Gap (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def gap_distribution(
    results: Dict[str, List[float]],
    title: str = "Gap Distribution",
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(results))
    bp = ax.boxplot(list(results.values()), positions=positions, patch_artist=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(list(results.keys()), rotation=45, ha="right")
    ax.set_ylabel("Gap (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_report(
    results: Dict[str, Dict],
    statistical_tests: Dict,
    benchmark_name: str,
    output_dir: str = "reports",
):
    os.makedirs(output_dir, exist_ok=True)
    performance_chart(
        results,
        title=f"{benchmark_name} - Performance Chart",
        save_path=os.path.join(output_dir, f"{benchmark_name}_performance.png"),
    )
    gap_dict = {s: d.get("gaps", []) for s, d in results.items()}
    if gap_dict:
        gap_distribution(
            gap_dict,
            title=f"{benchmark_name} - Gap Distribution",
            save_path=os.path.join(output_dir, f"{benchmark_name}_gaps.png"),
        )
    summary_path = os.path.join(output_dir, f"{benchmark_name}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Benchmark: {benchmark_name}\n")
        f.write("=" * 60 + "\n\n")
        for solver, data in results.items():
            f.write(f"Solver: {solver}\n")
            f.write(f"  Mean Gap: {data.get('mean_gap', 'N/A'):.4f}%\n")
            f.write(f"  Std Gap:  {data.get('std_gap', 'N/A'):.4f}%\n")
            f.write(f"  Mean Time: {data.get('mean_time', 'N/A'):.2f}s\n")
            f.write("\n")
        if statistical_tests:
            f.write("Statistical Tests\n")
            f.write("-" * 40 + "\n")
            ref = statistical_tests.get("reference_solver", "")
            f.write(f"Reference: {ref}\n\n")
            for solver, comp in statistical_tests.get("comparisons", {}).items():
                p = comp.get("p_value", "N/A")
                sig = comp.get("significant", "N/A")
                p_corr = comp.get("p_value_corrected", "N/A")
                sig_corr = comp.get("significant_corrected", "N/A")
                f.write(f"{solver} vs {ref}:\n")
                f.write(f"  p-value: {p}\n")
                f.write(f"  Significant (α=0.05): {sig}\n")
                f.write(f"  Bonferroni corrected p: {p_corr}\n")
                f.write(f"  Significant (corrected): {sig_corr}\n\n")
