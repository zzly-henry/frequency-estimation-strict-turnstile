"""
Metrics computation for frequency estimation evaluation.

Computes:
  - Mean / max / median relative error: |f̂(x) - f(x)| / F1
  - Mean / max absolute error: |f̂(x) - f(x)|
  - Weighted mean relative error (weighted by true frequency)
  - Heavy hitter precision / recall at threshold phi
  - NRMSE (Normalized Root Mean Square Error)
  
"""

import math
import numpy as np
from typing import Dict, Optional, Set


def compute_metrics(
    algo,
    true_freq: Dict,
    F1: int,
    candidates: Optional[Set] = None,
    phi: float = 0.01,
) -> dict:
    """
    Compute all error metrics for an algorithm against ground truth.

    Parameters
    ----------
    algo : object
        Algorithm instance with .query(item) and .heavy_hitters(phi) methods.
    true_freq : dict
        Ground truth {item: true_frequency}.
    F1 : int
        Total weight (sum of true frequencies).
    candidates : set or None
        Candidate set for sketch-based algorithms. If None, uses true_freq keys.
    phi : float
        Heavy hitter threshold fraction.

    Returns
    -------
    dict with metric keys.
    """
    if F1 <= 0:
        return _empty_metrics()

    items = list(true_freq.keys())
    if not items:
        return _empty_metrics()

    abs_errors = []
    rel_errors = []
    weighted_errors = []

    for item in items:
        true_f = true_freq[item]
        est_f = algo.query(item)
        abs_err = abs(est_f - true_f)
        rel_err = abs_err / F1

        abs_errors.append(abs_err)
        rel_errors.append(rel_err)
        weighted_errors.append(abs_err * true_f)

    abs_errors = np.array(abs_errors, dtype=np.float64)
    rel_errors = np.array(rel_errors, dtype=np.float64)

    # NRMSE
    mse = np.mean(abs_errors ** 2)
    nrmse = math.sqrt(mse) / F1 if F1 > 0 else 0.0

    # Heavy hitter precision / recall
    true_hh = {item for item, f in true_freq.items() if f >= phi * F1}
    try:
        if candidates is not None:
            est_hh_list = algo.heavy_hitters(phi, candidates) if _takes_candidates(algo) \
                else algo.heavy_hitters(phi)
        else:
            est_hh_list = algo.heavy_hitters(phi)
        est_hh = {item for item, _ in est_hh_list}
    except TypeError:
        est_hh = set()

    if len(est_hh) > 0:
        precision = len(true_hh & est_hh) / len(est_hh)
    else:
        precision = 1.0 if len(true_hh) == 0 else 0.0

    if len(true_hh) > 0:
        recall = len(true_hh & est_hh) / len(true_hh)
    else:
        recall = 1.0

    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "mean_absolute_error": float(np.mean(abs_errors)),
        "max_absolute_error": float(np.max(abs_errors)),
        "median_absolute_error": float(np.median(abs_errors)),
        "mean_relative_error": float(np.mean(rel_errors)),
        "max_relative_error": float(np.max(rel_errors)),
        "median_relative_error": float(np.median(rel_errors)),
        "nrmse": nrmse,
        "weighted_mean_rel_error": float(sum(weighted_errors)) / (F1 * F1) if F1 > 0 else 0.0,
        "hh_precision": precision,
        "hh_recall": recall,
        "hh_f1": f1_score,
        "num_true_hh": len(true_hh),
        "num_est_hh": len(est_hh),
    }


def _takes_candidates(algo) -> bool:
    """Check if algo.heavy_hitters accepts a candidates parameter."""
    import inspect
    try:
        sig = inspect.signature(algo.heavy_hitters)
        return "candidates" in sig.parameters
    except (ValueError, TypeError):
        return False


def _empty_metrics() -> dict:
    """Return zeroed metrics dict."""
    return {
        "mean_absolute_error": 0.0,
        "max_absolute_error": 0.0,
        "median_absolute_error": 0.0,
        "mean_relative_error": 0.0,
        "max_relative_error": 0.0,
        "median_relative_error": 0.0,
        "nrmse": 0.0,
        "weighted_mean_rel_error": 0.0,
        "hh_precision": 1.0,
        "hh_recall": 1.0,
        "hh_f1": 1.0,
        "num_true_hh": 0,
        "num_est_hh": 0,
    }


def compute_alpha_actual(stream) -> float:
    """Compute the actual α of a stream: sum|v_t| / sum(f_e)."""
    from collections import Counter
    freq = Counter()
    total_abs = 0
    for item, delta in stream:
        freq[item] += delta
        total_abs += abs(delta)
    F1 = sum(v for v in freq.values() if v > 0)
    return total_abs / F1 if F1 > 0 else float("inf")