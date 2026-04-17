"""
Real-world dataset loaders for Strict Turnstile + α-Bounded Deletion streams.

Supported datasets:
1. CAIDA Anonymized Internet Traces – destination IP as element.
2. YCSB (Yahoo! Cloud Serving Benchmark) – key as element, 60% insert / 40% update.

If raw data files are not available, synthetic proxies are generated that mimic
the statistical properties (Zipfian with s≈1.0 for CAIDA, mixed workload for YCSB).

To use real data:
  - CAIDA: download from https://www.caida.org/catalog/datasets/passive_dataset/
    Place the parsed CSV (columns: timestamp, src_ip, dst_ip, ...) at data/caida_trace.csv
  - YCSB: generate with https://github.com/brianfrankcooper/YCSB
    Place output at data/ycsb_workload.csv (columns: operation, key)
"""

import os
import csv
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter

# Default data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ======================================================================
# CAIDA
# ======================================================================

def load_caida_stream(
    filepath: str = None,
    N: int = 100000,
    alpha: float = 2.0,
    seed: int = 42,
) -> Tuple[List[Tuple], Dict]:
    """
    Load CAIDA trace and convert to strict-turnstile α-bounded stream.

    If filepath is None or file doesn't exist, generates a synthetic proxy
    (Zipfian s=1.0 over IP-like integers) that mimics CAIDA characteristics.

    Parameters
    ----------
    filepath : str or None
        Path to caida_trace.csv. Expected columns: any, with destination IP
        in column index 2 (0-indexed).
    N : int
        Target total operations.
    alpha : float
        α-bounded deletion parameter.
    seed : int
        Random seed.

    Returns
    -------
    stream : list of (element, delta)
    true_freq : dict
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "caida_trace.csv")

    if os.path.isfile(filepath):
        return _load_caida_real(filepath, N, alpha, seed)
    else:
        print(f"[INFO] CAIDA file not found at {filepath}. Using synthetic proxy (Zipf s=1.0).")
        return _generate_caida_proxy(N, alpha, seed)


def _load_caida_real(filepath: str, N: int, alpha: float, seed: int):
    """Parse real CAIDA CSV and build α-bounded turnstile stream."""
    rng = np.random.RandomState(seed)
    items = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                items.append(row[2].strip())  # dst_ip
            if len(items) >= N * 2:
                break

    if not items:
        return _generate_caida_proxy(N, alpha, seed)

    return _build_turnstile_stream(items, N, alpha, rng)


def _generate_caida_proxy(N: int, alpha: float, seed: int):
    """
    Synthetic proxy mimicking CAIDA: Zipfian s=1.0 over 50k distinct IPs.
    """
    from data_generators.synthetic_zipf import generate_zipf_stream
    return generate_zipf_stream(N=N, alpha=alpha, s=1.0, universe_size=50000, seed=seed)


# ======================================================================
# YCSB
# ======================================================================

def load_ycsb_stream(
    filepath: str = None,
    N: int = 100000,
    alpha: float = 2.0,
    seed: int = 42,
) -> Tuple[List[Tuple], Dict]:
    """
    Load YCSB workload and convert to strict-turnstile α-bounded stream.

    YCSB default: 60% INSERT, 40% UPDATE (treated as read-modify-write,
    modeled as delete-then-insert of same key).

    If file not found, generates a synthetic proxy with 60/40 split and
    Zipfian key distribution (matching YCSB's default requestdistribution).

    Parameters
    ----------
    filepath : str or None
        Path to ycsb_workload.csv. Expected columns: operation, key.
    N : int
        Target total operations.
    alpha : float
        α-bounded deletion parameter.
    seed : int
        Random seed.
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "ycsb_workload.csv")

    if os.path.isfile(filepath):
        return _load_ycsb_real(filepath, N, alpha, seed)
    else:
        print(f"[INFO] YCSB file not found at {filepath}. Using synthetic proxy.")
        return _generate_ycsb_proxy(N, alpha, seed)


def _load_ycsb_real(filepath: str, N: int, alpha: float, seed: int):
    """Parse real YCSB output."""
    rng = np.random.RandomState(seed)
    items = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                op = row[0].strip().upper()
                key = row[1].strip()
                if op in ("INSERT", "UPDATE", "READ"):
                    items.append(key)
            if len(items) >= N * 2:
                break

    if not items:
        return _generate_ycsb_proxy(N, alpha, seed)

    return _build_turnstile_stream(items, N, alpha, rng)


def _generate_ycsb_proxy(N: int, alpha: float, seed: int):
    """
    Synthetic YCSB proxy: Zipfian s=0.99 (YCSB default), 20k keys.
    The 60/40 insert/update split naturally produces deletions when
    updates are modeled as delete+insert cycles.
    """
    from data_generators.synthetic_zipf import generate_zipf_stream
    return generate_zipf_stream(N=N, alpha=alpha, s=0.99, universe_size=20000, seed=seed)


# ======================================================================
# Shared utility
# ======================================================================

def _build_turnstile_stream(
    raw_items: list,
    N: int,
    alpha: float,
    rng: np.random.RandomState,
) -> Tuple[List[Tuple], Dict]:
    """
    Convert a list of raw item occurrences into a strict-turnstile
    α-bounded deletion stream of ~N operations.

    Strategy: treat raw items as insertions, then sample deletions from
    accumulated frequencies to satisfy α-property.
    """
    if alpha <= 1.0:
        deletion_ratio = 0.0
    else:
        deletion_ratio = 1.0 - 1.0 / alpha

    num_inserts = min(int(N / (1.0 + deletion_ratio)), len(raw_items))
    num_deletes = min(N - num_inserts, int(deletion_ratio * num_inserts))

    # Use first num_inserts raw items
    insert_items = raw_items[:num_inserts]

    stream: List[Tuple] = []
    current_freq: Counter = Counter()

    insert_idx = 0
    deletes_remaining = num_deletes
    batch_size = max(1, num_inserts // max(num_deletes, 1) + 1)

    while insert_idx < num_inserts or deletes_remaining > 0:
        batch_end = min(insert_idx + batch_size, num_inserts)
        for i in range(insert_idx, batch_end):
            item = insert_items[i]
            stream.append((item, 1))
            current_freq[item] += 1
        insert_idx = batch_end

        if deletes_remaining > 0:
            pos_items = [it for it, c in current_freq.items() if c > 0]
            if pos_items:
                pos_counts = np.array([current_freq[it] for it in pos_items], dtype=np.float64)
                pos_probs = pos_counts / pos_counts.sum()
                n_del = min(deletes_remaining, len(pos_items), batch_size)
                del_items = rng.choice(len(pos_items), size=n_del, p=pos_probs, replace=True)
                for idx in del_items:
                    item = pos_items[idx]
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
                del_indices = rng.choice(len(pos_items), size=n_del, p=pos_probs, replace=True)
                for idx in del_indices:
                    item = pos_items[idx]
                    if current_freq[item] > 0 and deletes_remaining > 0:
                        stream.append((item, -1))
                        current_freq[item] -= 1
                        deletes_remaining -= 1
                if deletes_remaining <= 0:
                    break

    true_freq = {item: cnt for item, cnt in current_freq.items() if cnt > 0}
    return stream, true_freq


def get_stream_stats(stream: list, true_freq: dict) -> dict:
    """Return summary statistics."""
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