"""
Diverse Dataset Evaluation: test all algorithms on Zipfian, Uniform, Binomial,
CAIDA, and YCSB datasets across multiple α values.

Produces CSV results and comparison plots.
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.misra_gries import MisraGries
from algorithms.space_saving_plus import (
    LazySpaceSaving, SpaceSavingPlus, DoubleSpaceSaving, IntegratedSpaceSaving
)
from algorithms.count_min import CountMinSketch, AlphaCountMinSketch
from algorithms.count_sketch import CountSketch, AlphaCountSketch
from algorithms.learned_integrated_ss import LearnedIntegratedSpaceSaving

from data_generators.synthetic_zipf import generate_zipf_stream
from data_generators.synthetic_uniform import generate_uniform_stream, generate_binomial_stream
from data_generators.real_dataset_loader import load_caida_stream, load_ycsb_stream

from utils.metrics import compute_metrics
from utils.plotter import plot_dataset_results

# ======================================================================
# Configuration
# ======================================================================

N_DEFAULT = 500_000
ALPHAS = [1.5, 2.0, 4.0, 8.0]
EPSILON = 0.01
DELTA = 0.01
SEED = 42
UNIVERSE_SIZE = 50000
ZIPF_EXPONENTS = [1.0, 1.5, 2.0]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "dataset")


# ======================================================================
# Dataset generators registry
# ======================================================================

def get_datasets(alpha: float, N: int = N_DEFAULT):
    """Return dict of {dataset_name: (stream, true_freq)}."""
    datasets = {}

    # Zipfian with different exponents
    for s in ZIPF_EXPONENTS:
        name = f"Zipf_s{s}"
        stream, freq = generate_zipf_stream(
            N=N, alpha=alpha, s=s, universe_size=UNIVERSE_SIZE, seed=SEED
        )
        datasets[name] = (stream, freq)

    # Uniform
    stream, freq = generate_uniform_stream(
        N=N, alpha=alpha, universe_size=UNIVERSE_SIZE, seed=SEED
    )
    datasets["Uniform"] = (stream, freq)

    # Binomial
    stream, freq = generate_binomial_stream(
        N=N, alpha=alpha, universe_size=UNIVERSE_SIZE, seed=SEED
    )
    datasets["Binomial"] = (stream, freq)

    # CAIDA (real or proxy)
    stream, freq = load_caida_stream(N=N, alpha=alpha, seed=SEED)
    datasets["CAIDA"] = (stream, freq)

    # YCSB (real or proxy)
    stream, freq = load_ycsb_stream(N=N, alpha=alpha, seed=SEED)
    datasets["YCSB"] = (stream, freq)

    return datasets


def build_algorithms(alpha: float):
    """Instantiate all algorithms."""
    k = math.ceil(alpha / EPSILON)
    return {
        "MisraGries": MisraGries(k=k, alpha=alpha),
        "LazySS±": LazySpaceSaving(k=k, alpha=alpha),
        "SS±": SpaceSavingPlus(k=k, alpha=alpha),
        "DoubleSS±": DoubleSpaceSaving(k=k, alpha=alpha),
        "IntegratedSS±": IntegratedSpaceSaving(k=k, alpha=alpha),
        "CountMin": CountMinSketch.from_epsilon_delta(EPSILON, DELTA, seed=SEED),
        "α-CountMin": AlphaCountMinSketch.from_epsilon_delta_alpha(EPSILON, DELTA, alpha, seed=SEED),
        "CountSketch": CountSketch.from_epsilon_delta(EPSILON, DELTA, seed=SEED),
        "α-CountSketch": AlphaCountSketch.from_epsilon_delta_alpha(EPSILON, DELTA, alpha, seed=SEED),
        "LearnedISS±": LearnedIntegratedSpaceSaving(k=k, alpha=alpha),
    }


def run_on_dataset(dataset_name: str, stream, true_freq, alpha: float) -> list:
    """Run all algorithms on one dataset. Returns list of result dicts."""
    F1 = sum(true_freq.values())
    candidates = set(true_freq.keys())
    algos = build_algorithms(alpha)
    results = []

    for algo_name, algo in algos.items():
        t_start = time.perf_counter()
        for item, delta in stream:
            algo.update(item, delta)
        t_elapsed = time.perf_counter() - t_start

        metrics = compute_metrics(algo, true_freq, F1, candidates=candidates)

        results.append({
            "dataset": dataset_name,
            "algorithm": algo_name,
            "alpha": alpha,
            "N": len(stream),
            "F1": F1,
            "distinct": len(true_freq),
            "space": algo.get_space(),
            "max_space": algo.get_max_space(),
            "update_time_s": t_elapsed,
            "time_per_update_us": (t_elapsed / len(stream)) * 1e6,
            **metrics,
        })
        print(f"    {algo_name:20s}  err={metrics['mean_relative_error']:.6f}  "
              f"space={algo.get_space()}")

    return results


def run_dataset_evaluation():
    """Run full dataset evaluation across all α values and datasets."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []
    for alpha in ALPHAS:
        print(f"\n{'='*60}")
        print(f"α = {alpha}")
        print(f"{'='*60}")

        datasets = get_datasets(alpha)
        for ds_name, (stream, true_freq) in datasets.items():
            print(f"\n  Dataset: {ds_name}  |stream|={len(stream)}  "
                  f"F1={sum(true_freq.values())}  distinct={len(true_freq)}")
            results = run_on_dataset(ds_name, stream, true_freq, alpha)
            all_results.extend(results)

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "dataset_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    plot_dataset_results(df, OUTPUT_DIR)
    print(f"Plots saved to {OUTPUT_DIR}")

    return df


if __name__ == "__main__":
    run_dataset_evaluation()