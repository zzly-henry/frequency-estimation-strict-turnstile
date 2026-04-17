"""
Plotting utilities for frequency estimation experiments.

Generates publication-quality plots using matplotlib and seaborn.
All functions save figures as PNG and PDF to the specified output directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Style
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("tab10", 12)
FIGSIZE = (10, 6)
DPI = 150


def _save_fig(fig, output_dir: str, name: str):
    """Save figure as PNG and PDF."""
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Parametric evaluation plots
# ======================================================================

def plot_parametric_results(df: pd.DataFrame, output_dir: str):
    """Generate all parametric evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)

    _plot_error_vs_N(df, output_dir)
    _plot_error_vs_alpha(df, output_dir)
    _plot_space_vs_alpha(df, output_dir)
    _plot_time_vs_N(df, output_dir)
    _plot_heatmap_error(df, output_dir)


def _plot_error_vs_N(df: pd.DataFrame, output_dir: str):
    """Mean relative error vs stream length N, one line per algorithm."""
    fig, axes = plt.subplots(1, len(df["alpha"].unique()), figsize=(5 * len(df["alpha"].unique()), 5),
                              sharey=True, squeeze=False)
    for idx, alpha in enumerate(sorted(df["alpha"].unique())):
        ax = axes[0, idx]
        sub = df[df["alpha"] == alpha]
        for algo in sub["algorithm"].unique():
            algo_df = sub[sub["algorithm"] == algo].sort_values("N")
            ax.plot(algo_df["N"], algo_df["mean_relative_error"],
                    marker="o", markersize=4, label=algo)
        ax.set_xlabel("Stream Length N")
        ax.set_ylabel("Mean Relative Error")
        ax.set_title(f"α = {alpha}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if idx == 0:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("Relative Error vs Stream Length", y=1.02)
    _save_fig(fig, output_dir, "error_vs_N")


def _plot_error_vs_alpha(df: pd.DataFrame, output_dir: str):
    """Mean relative error vs α, one line per algorithm. Averaged over N."""
    agg = df.groupby(["algorithm", "alpha"])["mean_relative_error"].mean().reset_index()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for algo in agg["algorithm"].unique():
        algo_df = agg[agg["algorithm"] == algo].sort_values("alpha")
        ax.plot(algo_df["alpha"], algo_df["mean_relative_error"],
                marker="s", markersize=5, label=algo)
    ax.set_xlabel("α (Bounded Deletion Parameter)")
    ax.set_ylabel("Mean Relative Error")
    ax.set_title("Relative Error vs α")
    ax.set_yscale("log")
    ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    _save_fig(fig, output_dir, "error_vs_alpha")


def _plot_space_vs_alpha(df: pd.DataFrame, output_dir: str):
    """Space consumption vs α."""
    # Take one N value for clarity
    N_val = sorted(df["N"].unique())[0]
    sub = df[df["N"] == N_val]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for algo in sub["algorithm"].unique():
        algo_df = sub[sub["algorithm"] == algo].sort_values("alpha")
        ax.plot(algo_df["alpha"], algo_df["space"],
                marker="^", markersize=5, label=algo)
    ax.set_xlabel("α (Bounded Deletion Parameter)")
    ax.set_ylabel("Space (counters / cells)")
    ax.set_title(f"Space vs α (N={N_val})")
    ax.set_yscale("log")
    ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    _save_fig(fig, output_dir, "space_vs_alpha")


def _plot_time_vs_N(df: pd.DataFrame, output_dir: str):
    """Update time per operation vs N."""
    agg = df.groupby(["algorithm", "N"])["time_per_update_us"].mean().reset_index()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for algo in agg["algorithm"].unique():
        algo_df = agg[agg["algorithm"] == algo].sort_values("N")
        ax.plot(algo_df["N"], algo_df["time_per_update_us"],
                marker="d", markersize=4, label=algo)
    ax.set_xlabel("Stream Length N")
    ax.set_ylabel("Time per Update (μs)")
    ax.set_title("Update Time vs Stream Length")
    ax.set_xscale("log")
    ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    _save_fig(fig, output_dir, "time_vs_N")


def _plot_heatmap_error(df: pd.DataFrame, output_dir: str):
    """Heatmap: algorithm × α showing mean relative error (averaged over N)."""
    pivot = df.groupby(["algorithm", "alpha"])["mean_relative_error"].mean().reset_index()
    pivot_table = pivot.pivot(index="algorithm", columns="alpha", values="mean_relative_error")

    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot_table) * 0.5)))
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Mean Relative Error"})
    ax.set_title("Error Heatmap: Algorithm × α")
    ax.set_ylabel("Algorithm")
    ax.set_xlabel("α")
    _save_fig(fig, output_dir, "heatmap_error")


# ======================================================================
# Dataset evaluation plots
# ======================================================================

def plot_dataset_results(df: pd.DataFrame, output_dir: str):
    """Generate all dataset evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)

    _plot_error_by_dataset(df, output_dir)
    _plot_error_by_dataset_alpha(df, output_dir)
    _plot_hh_performance(df, output_dir)


def _plot_error_by_dataset(df: pd.DataFrame, output_dir: str):
    """Grouped bar chart: mean relative error per algorithm, grouped by dataset."""
    # Average over α for summary view
    agg = df.groupby(["dataset", "algorithm"])["mean_relative_error"].mean().reset_index()

    datasets = sorted(agg["dataset"].unique())
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(4 * n_ds, 6), sharey=True, squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[0, idx]
        sub = agg[agg["dataset"] == ds].sort_values("mean_relative_error")
        bars = ax.barh(sub["algorithm"], sub["mean_relative_error"], color=PALETTE[:len(sub)])
        ax.set_xlabel("Mean Relative Error")
        ax.set_title(ds)
        ax.set_xscale("log")
    fig.suptitle("Error by Dataset (averaged over α)", y=1.02)
    _save_fig(fig, output_dir, "error_by_dataset")


def _plot_error_by_dataset_alpha(df: pd.DataFrame, output_dir: str):
    """Line plot: error vs α for each dataset, faceted."""
    datasets = sorted(df["dataset"].unique())
    n_ds = len(datasets)
    fig, axes = plt.subplots(2, (n_ds + 1) // 2, figsize=(6 * ((n_ds + 1) // 2), 10),
                              squeeze=False)
    axes_flat = axes.flatten()

    for idx, ds in enumerate(datasets):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        sub = df[df["dataset"] == ds]
        agg = sub.groupby(["algorithm", "alpha"])["mean_relative_error"].mean().reset_index()
        for algo in agg["algorithm"].unique():
            a_df = agg[agg["algorithm"] == algo].sort_values("alpha")
            ax.plot(a_df["alpha"], a_df["mean_relative_error"],
                    marker="o", markersize=4, label=algo)
        ax.set_xlabel("α")
        ax.set_ylabel("Mean Relative Error")
        ax.set_title(ds)
        ax.set_yscale("log")
        if idx == 0:
            ax.legend(fontsize=6, loc="best")

    # Hide unused axes
    for idx in range(n_ds, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Error vs α by Dataset", y=1.02)
    _save_fig(fig, output_dir, "error_vs_alpha_by_dataset")


def _plot_hh_performance(df: pd.DataFrame, output_dir: str):
    """Heavy hitter precision and recall by algorithm."""
    agg = df.groupby("algorithm")[["hh_precision", "hh_recall", "hh_f1"]].mean().reset_index()
    agg = agg.sort_values("hh_f1", ascending=False)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(agg))
    w = 0.25
    ax.bar(x - w, agg["hh_precision"], width=w, label="Precision", color=PALETTE[0])
    ax.bar(x, agg["hh_recall"], width=w, label="Recall", color=PALETTE[1])
    ax.bar(x + w, agg["hh_f1"], width=w, label="F1", color=PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(agg["algorithm"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Heavy Hitter Detection Performance")
    ax.legend()
    ax.set_ylim(0, 1.05)
    _save_fig(fig, output_dir, "hh_performance")


# ======================================================================
# Advanced evaluation plots
# ======================================================================

def plot_advanced_results(df: pd.DataFrame, output_dir: str):
    """Generate all advanced evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Experiment 1: baseline comparison
    baseline_df = df[df["experiment"] == "baseline_comparison"]
    if len(baseline_df) > 0:
        _plot_advanced_baseline(baseline_df, output_dir)

    # Experiment 2: fixed ratio sweep
    fr_df = df[df["experiment"] == "fixed_ratio_sweep"]
    if len(fr_df) > 0:
        _plot_fixed_ratio(fr_df, output_dir)

    # Experiment 3: window size sweep
    ws_df = df[df["experiment"] == "window_size_sweep"]
    if len(ws_df) > 0:
        _plot_window_size(ws_df, output_dir)

    # Summary table
    _save_summary_table(df, output_dir)


def _plot_advanced_baseline(df: pd.DataFrame, output_dir: str):
    """LearnedISS± vs baselines across Zipf exponents."""
    fig, axes = plt.subplots(1, len(df["zipf_s"].unique()),
                              figsize=(5 * len(df["zipf_s"].unique()), 5),
                              sharey=True, squeeze=False)
    for idx, s in enumerate(sorted(df["zipf_s"].unique())):
        ax = axes[0, idx]
        sub = df[df["zipf_s"] == s]
        agg = sub.groupby(["algorithm", "alpha"])["mean_relative_error"].mean().reset_index()
        for algo in agg["algorithm"].unique():
            a_df = agg[agg["algorithm"] == algo].sort_values("alpha")
            ax.plot(a_df["alpha"], a_df["mean_relative_error"],
                    marker="o", markersize=5, label=algo)
        ax.set_xlabel("α")
        ax.set_ylabel("Mean Relative Error")
        ax.set_title(f"Zipf s={s}")
        ax.set_yscale("log")
        if idx == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Learned ISS± vs Baselines", y=1.02)
    _save_fig(fig, output_dir, "advanced_baseline_comparison")


def _plot_fixed_ratio(df: pd.DataFrame, output_dir: str):
    """Error vs fixed_ratio parameter."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    learned = df[df["algorithm"] == "LearnedISS±"].sort_values("fixed_ratio")
    baseline = df[df["algorithm"] != "LearnedISS±"]

    ax.plot(learned["fixed_ratio"], learned["mean_relative_error"],
            marker="o", markersize=6, label="LearnedISS±", color=PALETTE[0], linewidth=2)

    # Baseline horizontal line
    if len(baseline) > 0:
        bl_err = baseline["mean_relative_error"].values[0]
        ax.axhline(y=bl_err, color=PALETTE[1], linestyle="--",
                    label="IntegratedSS± baseline", linewidth=2)

    ax.set_xlabel("Fixed Ratio (fraction of counters for predicted hot items)")
    ax.set_ylabel("Mean Relative Error")
    ax.set_title("Impact of fixed_ratio on Learned ISS±")
    ax.legend()
    _save_fig(fig, output_dir, "fixed_ratio_sweep")


def _plot_window_size(df: pd.DataFrame, output_dir: str):
    """Error vs window_size parameter."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    df_sorted = df.sort_values("window_size")
    ax.plot(df_sorted["window_size"], df_sorted["mean_relative_error"],
            marker="s", markersize=6, color=PALETTE[0], linewidth=2)
    ax.set_xlabel("Window Size")
    ax.set_ylabel("Mean Relative Error")
    ax.set_title("Impact of Window Size on Learned ISS±")
    ax.set_xscale("log")
    _save_fig(fig, output_dir, "window_size_sweep")


def _save_summary_table(df: pd.DataFrame, output_dir: str):
    """Save a summary CSV table of all advanced results."""
    summary_cols = ["experiment", "algorithm", "alpha", "mean_relative_error",
                    "max_relative_error", "space", "update_time_s"]
    existing = [c for c in summary_cols if c in df.columns]
    summary = df[existing].copy()
    summary.to_csv(os.path.join(output_dir, "advanced_summary.csv"), index=False)