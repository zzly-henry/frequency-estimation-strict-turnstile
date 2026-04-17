"""
Synthetic Uniform/Balanced stream generator for Strict Turnstile + α-Bounded Deletion.

Generates insertions from a uniform (or binomial) distribution, then samples
deletions uniformly at random from previously inserted items such that:
  1. f_e >= 0 for all e at all times (strict turnstile)
  2. D <= (1 - 1/α) * I  (α-bounded deletion property)
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import Counter


def generate_uniform_stream(
    N: int = 100000,
    alpha: float = 2.0,
    universe_size: int = 10000,
    seed: int = 42,
) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
    """
    Generate a strict-turnstile α-bounded deletion stream with uniform inserts.

    Parameters
    ----------
    N : int
        Total number of operations.
    alpha : float
        α-bounded deletion parameter (>= 1).
    universe_size : int
        Number of distinct elements.
    seed : int
        Random seed.

    Returns
    -------
    stream : list of (element, delta)
    true_freq : dict {element: final_frequency}
    """
    rng = np.random.RandomState(seed)

    if alpha <= 1.0:
        num_inserts = N
        num_deletes = 0
    else:
        deletion_ratio = 1.0 - 1.0 / alpha
        num_inserts = int(N / (1.0 + deletion_ratio))
        num_deletes = N - num_inserts
        num_deletes = min(num_deletes, int(deletion_ratio * num_inserts))

    # Uniform insertions
    insert_items = rng.randint(1, universe_size + 1, size=num_inserts)

    stream: List[Tuple[int, int]] = []
    current_freq: Counter = Counter()

    insert_idx = 0
    deletes_remaining = num_deletes
    batch_size = max(1, num_inserts // max(num_deletes, 1) + 1)

    while insert_idx < num_inserts or deletes_remaining > 0:
        # Insert batch
        batch_end = min(insert_idx + batch_size, num_inserts)
        for i in range(insert_idx, batch_end):
            item = int(insert_items[i])
            stream.append((item, 1))
            current_freq[item] += 1
        insert_idx = batch_end

        # Delete some
        if deletes_remaining > 0:
            pos_items = [it for it, c in current_freq.items() if c > 0]
            if pos_items:
                pos_counts = np.array([current_freq[it] for it in pos_items], dtype=np.float64)
                pos_probs = pos_counts / pos_counts.sum()
                n_del = min(deletes_remaining, len(pos_items), batch_size)
                del_items = rng.choice(pos_items, size=n_del, p=pos_probs, replace=True)
                for item in del_items:
                    item = int(item)
                    if current_freq[item] > 0 and deletes_remaining > 0:
                        stream.append((item, -1))
                        current_freq[item] -= 1
                        deletes_remaining -= 1

        # Drain remaining deletes after all inserts
        if insert_idx >= num_inserts and deletes_remaining > 0:
            while deletes_remaining > 0:
                pos_items = [it for it, c in current_freq.items() if c > 0]
                if not pos_items:
                    break
                pos_counts = np.array([current_freq[it] for it in pos_items], dtype=np.float64)
                pos_probs = pos_counts / pos_counts.sum()
                n_del = min(deletes_remaining, len(pos_items))
                del_items = rng.choice(pos_items, size=n_del, p=pos_probs, replace=True)
                for item in del_items:
                    item = int(item)
                    if current_freq[item] > 0 and deletes_remaining > 0:
                        stream.append((item, -1))
                        current_freq[item] -= 1
                        deletes_remaining -= 1
                if deletes_remaining <= 0:
                    break

    true_freq = {item: cnt for item, cnt in current_freq.items() if cnt > 0}
    return stream, true_freq


def generate_binomial_stream(
    N: int = 100000,
    alpha: float = 2.0,
    universe_size: int = 10000,
    n_trials: int = 20,
    p_success: float = 0.5,
    seed: int = 42,
) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
    """
    Generate a strict-turnstile α-bounded deletion stream with binomial-distributed inserts.

    Items are drawn as (binomial(n_trials, p_success) % universe_size) + 1.
    """
    rng = np.random.RandomState(seed)

    if alpha <= 1.0:
        num_inserts = N
        num_deletes = 0
    else:
        deletion_ratio = 1.0 - 1.0 / alpha
        num_inserts = int(N / (1.0 + deletion_ratio))
        num_deletes = N - num_inserts
        num_deletes = min(num_deletes, int(deletion_ratio * num_inserts))

    raw = rng.binomial(n_trials, p_success, size=num_inserts)
    insert_items = (raw % universe_size) + 1

    stream: List[Tuple[int, int]] = []
    current_freq: Counter = Counter()

    insert_idx = 0
    deletes_remaining = num_deletes
    batch_size = max(1, num_inserts // max(num_deletes, 1) + 1)

    while insert_idx < num_inserts or deletes_remaining > 0:
        batch_end = min(insert_idx + batch_size, num_inserts)
        for i in range(insert_idx, batch_end):
            item = int(insert_items[i])
            stream.append((item, 1))
            current_freq[item] += 1
        insert_idx = batch_end

        if deletes_remaining > 0:
            pos_items = [it for it, c in current_freq.items() if c > 0]
            if pos_items:
                pos_counts = np.array([current_freq[it] for it in pos_items], dtype=np.float64)
                pos_probs = pos_counts / pos_counts.sum()
                n_del = min(deletes_remaining, len(pos_items), batch_size)
                del_items = rng.choice(pos_items, size=n_del, p=pos_probs, replace=True)
                for item in del_items:
                    item = int(item)
                    if current_freq[item] > 0 and deletes_remaining > 0:
                        stream.append((item, -1))
                        current_freq[item] -= 1
                        deletes_remaining -= 1

        if insert_idx >= num_inserts and deletes_remaining > 0:
            while deletes_remaining > 0:
                pos_items = [it for it, c in current_freq.items() if c > 0]
                if not pos_items:
                    break
                pos_counts = np.array([current_freq[it] for it in pos_items], dtype=np.float64)
                pos_probs = pos_counts / pos_counts.sum()
                n_del = min(deletes_remaining, len(pos_items))
                del_items = rng.choice(pos_items, size=n_del, p=pos_probs, replace=True)
                for item in del_items:
                    item = int(item)
                    if current_freq[item] > 0 and deletes_remaining > 0:
                        stream.append((item, -1))
                        current_freq[item] -= 1
                        deletes_remaining -= 1
                if deletes_remaining <= 0:
                    break

    true_freq = {item: cnt for item, cnt in current_freq.items() if cnt > 0}
    return stream, true_freq


def get_stream_stats(stream: List[Tuple[int, int]], true_freq: Dict[int, int]) -> dict:
    """Return summary statistics of a generated stream."""
    inserts = sum(1 for _, d in stream if d > 0)
    deletes = sum(1 for _, d in stream if d < 0)
    F1 = sum(true_freq.values())
    distinct = len(true_freq)
    return {
        "total_ops": len(stream),
        "inserts": inserts,
        "deletes": deletes,
        "F1": F1,
        "distinct_elements": distinct,
        "alpha_actual": (inserts + deletes) / F1 if F1 > 0 else float("inf"),
    }