import numpy as np
from scipy.stats import wilcoxon, norm
from typing import Dict, List, Tuple, Optional


def compute_passmark_factor(
    reference_pm: float = 2277, target_pm: float = 2277
) -> float:
    return reference_pm / target_pm


def normalize_time(runtime: float, passmark_factor: float) -> float:
    return runtime * passmark_factor


def compute_gap(cost: float, best_known: float) -> float:
    return 100.0 * (cost - best_known) / best_known


def compute_passmark_gap(
    cost: float, best_known: float, runtime: float, passmark_factor: float
) -> Tuple[float, float]:
    gap = compute_gap(cost, best_known)
    norm_time = normalize_time(runtime, passmark_factor)
    return gap, norm_time


def wilcoxon_signed_rank(
    results_a: List[float], results_b: List[float], alternative: str = "less"
) -> Dict:
    if len(results_a) != len(results_b):
        raise ValueError("Results must have the same length")
    if len(results_a) < 6:
        return {
            "statistic": None,
            "p_value": None,
            "significant": None,
            "error": "Sample size too small for Wilcoxon test (n < 6)",
        }
    try:
        stat, p = wilcoxon(results_a, results_b, alternative=alternative)
        return {
            "statistic": float(stat),
            "p_value": float(p),
            "significant": bool(p < 0.05),
            "n": len(results_a),
        }
    except ValueError as e:
        return {
            "statistic": None,
            "p_value": None,
            "significant": None,
            "error": str(e),
        }


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> np.ndarray:
    m = len(p_values)
    corrected = np.array(p_values) * m
    corrected = np.clip(corrected, 0, 1.0)
    return corrected


def evaluate_solver_set(
    results: Dict[str, Dict[str, List[float]]],
    reference_solver: str,
    alpha: float = 0.05,
) -> Dict:
    ref_costs = results[reference_solver]["costs"]
    comparisons = {}
    for solver, data in results.items():
        if solver == reference_solver:
            continue
        comp = wilcoxon_signed_rank(ref_costs, data["costs"], alternative="less")
        comparisons[solver] = comp
    raw_p = [
        comp["p_value"] for comp in comparisons.values() if comp["p_value"] is not None
    ]
    if raw_p:
        corrected = bonferroni_correction(raw_p, alpha)
        idx = 0
        for solver, comp in comparisons.items():
            if comp["p_value"] is not None:
                comp["p_value_corrected"] = float(corrected[idx])
                comp["significant_corrected"] = bool(corrected[idx] < alpha)
                idx += 1
    return {
        "reference_solver": reference_solver,
        "comparisons": comparisons,
        "alpha": alpha,
    }


def compute_summary(results: Dict[str, List[float]], best_known: float) -> Dict:
    costs = np.array(results)
    gaps = compute_gap(costs, best_known)
    return {
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "min_cost": float(np.min(costs)),
        "max_cost": float(np.max(costs)),
        "mean_gap": float(np.mean(gaps)),
        "std_gap": float(np.std(gaps)),
        "best_gap": float(np.min(gaps)),
        "worst_gap": float(np.max(gaps)),
    }
