"""
Synthetic Zipfian stream generator for Strict Turnstile + α-Bounded Deletion.

Generates insertions from a Zipfian distribution, then samples deletions
uniformly at random from previously inserted items such that:
  1. f_e >= 0 for all e at all times (strict turnstile)
  2. D <= (1 - 1/α) * I  (α-bounded deletion property)

Parameters:
  - N: total number of stream operations (inserts + deletes)
  - alpha: α parameter (>= 1)
  - s: Zipfian exponent (skewness)
  - universe_size: number of distinct elements
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import Counter


def generate_zipf_stream(
    N: int = 100000,
    alpha: float = 2.0,
    s: float = 1.0,
    universe_size: int = 10000,
    seed: int = 42,
) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
    """
    Generate a strict-turnstile α-bounded deletion stream with Zipfian inserts.

    Parameters
    ----------
    N : int
        Total number of operations (insertions + deletions).
    alpha : float
        α-bounded deletion parameter (>= 1). α=1 means no deletions.
    s : float
        Zipfian exponent. Higher = more skewed.
    universe_size : int
        Number of distinct elements in the universe.
    seed : int
        Random seed.

    Returns
    -------
    stream : list of (element, delta) tuples
        The generated stream. delta=+1 for insert, delta=-1 for delete.
    true_freq : dict
        True final frequencies {element: frequency}.
    """
    rng = np.random.RandomState(seed)

    # Compute number of insertions and deletions from α
    # D <= (1 - 1/α) * I  and  I + D = N
    # D = (1 - 1/α) * I  =>  I + (1-1/α)*I = N  =>  I*(2-1/α) = N
    # I = N / (2 - 1/α)
    if alpha <= 1.0:
        # No deletions
        num_inserts = N
        num_deletes = 0
    else:
        deletion_ratio = 1.0 - 1.0 / alpha
        num_inserts = int(N / (1.0 + deletion_ratio))
        num_deletes = N - num_inserts
        # Verify: num_deletes <= deletion_ratio * num_inserts
        num_deletes = min(num_deletes, int(deletion_ratio * num_inserts))

    # Generate Zipfian insertions
    # Zipf weights for elements 1..universe_size
    ranks = np.arange(1, universe_size + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, s)
    weights /= weights.sum()

    insert_items = rng.choice(universe_size, size=num_inserts, p=weights) + 1  # 1-indexed

    # Build the stream: first generate all inserts, then interleave deletes
    # To ensure strict turnstile (f_e >= 0 always), we interleave carefully:
    # Process inserts and schedule deletes from items that have positive freq.

    stream: List[Tuple[int, int]] = []
    current_freq: Counter = Counter()

    # Create a pool: all inserts placed first, deletes drawn from accumulated items
    insert_idx = 0
    deletes_remaining = num_deletes

    # Strategy: process in rounds. Each round: batch of inserts, then some deletes.
    batch_size = max(1, num_inserts // max(num_deletes, 1) + 1)

    while insert_idx < num_inserts or deletes_remaining > 0:
        # Insert a batch
        batch_end = min(insert_idx + batch_size, num_inserts)
        for i in range(insert_idx, batch_end):
            item = int(insert_items[i])
            stream.append((item, 1))
            current_freq[item] += 1
        insert_idx = batch_end

        # Delete some items (proportional to batch)
        if deletes_remaining > 0 and sum(current_freq.values()) > 0:
            n_del_now = min(
                deletes_remaining,
                max(1, batch_end - (batch_end - batch_size)),
                int(sum(v for v in current_freq.values() if v > 0))
            )
            # Sample deletions uniformly from items with positive frequency
            pos_items = [item for item, cnt in current_freq.items() if cnt > 0]
            if pos_items and n_del_now > 0:
                pos_counts = np.array([current_freq[item] for item in pos_items], dtype=np.float64)
                pos_probs = pos_counts / pos_counts.sum()
                del_items = rng.choice(pos_items, size=min(n_del_now, len(pos_items)),
                                       p=pos_probs, replace=True)
                for item in del_items:
                    item = int(item)
                    if current_freq[item] > 0:
                        stream.append((item, -1))
                        current_freq[item] -= 1
                        deletes_remaining -= 1
                        if deletes_remaining <= 0:
                            break

        # Safety: if no inserts left but deletes remain and no positive freq, stop
        if insert_idx >= num_inserts and deletes_remaining > 0:
            pos_items = [item for item, cnt in current_freq.items() if cnt > 0]
            if not pos_items:
                break
            pos_counts = np.array([current_freq[item] for item in pos_items], dtype=np.float64)
            pos_probs = pos_counts / pos_counts.sum()
            while deletes_remaining > 0 and any(current_freq[i] > 0 for i in pos_items):
                del_items = rng.choice(pos_items, size=min(deletes_remaining, len(pos_items)),
                                       p=pos_probs, replace=True)
                for item in del_items:
                    item = int(item)
                    if current_freq[item] > 0:
                        stream.append((item, -1))
                        current_freq[item] -= 1
                        deletes_remaining -= 1
                        if deletes_remaining <= 0:
                            break
                # Recompute
                pos_items = [item for item, cnt in current_freq.items() if cnt > 0]
                if not pos_items:
                    break
                pos_counts = np.array([current_freq[item] for item in pos_items], dtype=np.float64)
                pos_probs = pos_counts / pos_counts.sum()

    # True frequencies
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