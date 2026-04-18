# Frequency Estimation under Strict Turnstile Model with α-Bounded Deletion

**SEEM5020 Course Project**

This repository implements and evaluates classic frequency estimation algorithms extended to handle the **Strict Turnstile Model** with the **α-Bounded Deletion** property.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Algorithms](#algorithms)
- [Quick Start](#quick-start)
- [Reproducing All Results](#reproducing-all-results)
- [Datasets](#datasets)
- [Dependencies](#dependencies)
- [Citations](#citations)

---

## Overview

**Strict Turnstile Model**: Stream updates are pairs `(e_t, v_t)` where `v_t` can be positive (insert) or negative (delete), but the true frequency `f_e` of any element never drops below zero at any point.

**α-Bounded Deletion Property (L1 α-property)**: The total absolute volume of all updates satisfies:

```
sum|v_t| <= alpha * sum|f_e|    where alpha >= 1
```

Equivalently, the number of deletions `D <= (1 - 1/alpha) * I` where `I` is the total insertions.

All algorithms guarantee deterministic error `|f_hat(x) - f(x)| <= epsilon * F1` using `O(alpha/epsilon)` space.

---

## Repository Structure

```
frequency-estimation-strict-turnstile/
├── algorithms/
│   ├── misra_gries.py              # Misra-Gries extended for turnstile
│   ├── space_saving_plus.py        # 4 variants: Lazy, SS±, Double, Integrated
│   ├── count_min.py                # Standard + α-optimized Count-Min Sketch
│   ├── count_sketch.py             # Standard + α-optimized Count-Sketch
│   └── learned_integrated_ss.py    # Advanced: Learned-Integrated SpaceSaving±
├── data_generators/
│   ├── synthetic_zipf.py           # Zipfian streams (s = 1.0, 1.5, 2.0)
│   ├── synthetic_uniform.py        # Uniform and Binomial streams
│   └── real_dataset_loader.py      # MAWI pcap loader
├── experiments/
│   ├── parametric_eval.py          # Vary N and α
│   ├── dataset_eval.py             # Diverse dataset evaluation
│   └── advanced_eval.py            # Learned-ISS± evaluation
├── utils/
│   ├── metrics.py                  # Error metrics, HH precision/recall
│   └── plotter.py                  # Publication-quality plots
├── data/                           # Place 202506181400.pcap here
│   └── 202506181400.pcap
├── results/                        # Auto-generated output directory
├── README.md
├── requirements.txt
├── run_all_experiments.sh
└── report.pdf                      # Auto-generated report with figures
```

---

## Algorithms

### Counter-Based (extended from insertion-only to strict turnstile)

| Algorithm | Description | Space |
|-----------|-------------|-------|
| **Misra-Gries** | Classic decrement-all; deletions spread to min-counter | O(α/ε) |
| **Lazy SpaceSaving±** | Ignores deletions of unmonitored items | O(α/ε) |
| **SpaceSaving±** | Decrements max-error counter on unmonitored deletions | O(α/ε) |
| **Double SpaceSaving±** | Separate summaries for inserts and deletes | O(α/ε) |
| **Integrated SpaceSaving±** | Single summary tracking insert+delete counts; mergeable | O(α/ε) |

### Sketch-Based (natively supports turnstile)

| Algorithm | Description | Space |
|-----------|-------------|-------|
| **Count-Min Sketch** | Standard turnstile CMS, min-query | O((1/ε) log(1/δ)) |
| **α-Count-Min Sketch** | Reduced depth using α-bounded property | optimized |
| **Count-Sketch** | Median estimator with sign hashing | O((1/ε²) log(1/δ)) |
| **α-Count-Sketch** | Reduced depth using α-bounded property | optimized |

### Advanced Design

| Algorithm | Description |
|-----------|-------------|
| **Learned-Integrated SpaceSaving±** | Combines Integrated SS± with a sliding-window frequency predictor. Allocates fixed counters to predicted hot items, reducing eviction error on heavy hitters under skewed distributions. |

---

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Place the MAWI trace file

Downloaded the MAWI Working Group Traffic Archive trace file: **202506181400.pcap**

Place it in the `data/` folder:

```bash
mkdir -p data
cp /path/to/202506181400.pcap data/
```

If the file is not present, the dataset evaluation will skip the MAWI dataset and run only on synthetic data.

### 3. Run all experiments

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

Or run individually:

```bash
# Parametric evaluation (vary N, α)
python -m experiments.parametric_eval

# Dataset evaluation — all datasets (synthetic + MAWI)
python -m experiments.dataset_eval

# Dataset evaluation — MAWI only with specific α
python -m experiments.dataset_eval --dataset mawi --alpha 2.0

# Dataset evaluation — synthetic only
python -m experiments.dataset_eval --dataset synthetic

# Advanced evaluation (Learned-ISS± analysis)
python -m experiments.advanced_eval
```

### 4. View results

Results are saved to `results/` with CSV files and PNG/PDF plots:

```
results/
├── parametric/
│   ├── parametric_results.csv
│   ├── error_vs_N.png
│   ├── error_vs_alpha.png
│   ├── space_vs_alpha.png
│   ├── time_vs_N.png
│   └── heatmap_error.png
├── dataset/
│   ├── dataset_results.csv
│   ├── error_by_dataset.png
│   ├── error_vs_alpha_by_dataset.png
│   └── hh_performance.png
└── advanced/
    ├── advanced_results.csv
    ├── advanced_baseline_comparison.png
    ├── fixed_ratio_sweep.png
    └── window_size_sweep.png
```

---

## Reproducing All Results

```bash
# Clone the repository
git clone https://github.com/your-username/frequency-estimation-strict-turnstile.git
cd frequency-estimation-strict-turnstile

# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Place MAWI trace in data/
mkdir -p data
cp /path/to/202506181400.pcap data/

# Run everything
./run_all_experiments.sh

# Or step by step:
python -m experiments.parametric_eval
python -m experiments.dataset_eval
python -m experiments.advanced_eval
```

**Estimated total runtime**: ~30-60 minutes depending on hardware (the N=2x10^6 configurations are the slowest).

---

## Datasets

### Synthetic

- **Zipfian (Skewed)**: Generated via `data_generators/synthetic_zipf.py` with exponents s in {1.0, 1.5, 2.0}. Insertions drawn from Zipf distribution, deletions sampled uniformly from previously inserted items to satisfy α-property and strict turnstile.
- **Uniform (Balanced)**: Generated via `data_generators/synthetic_uniform.py`. Same deletion sampling rule.
- **Binomial (Balanced)**: Generated via `data_generators/synthetic_uniform.py` (binomial variant). Same deletion sampling rule.

### Real-World

- **MAWI Working Group Traffic Archive**
  - File: `202506181400.pcap`
  - Source: https://mawi.wide.ad.jp/mawi/samplepoint-F/2025/202506181400.html
  - Element: destination IP address extracted from each packet
  - Each packet is treated as a +1 insertion. Deletions are then randomly sampled from previously seen items so that the stream exactly satisfies strict turnstile (f_e >= 0 always) and the chosen α-bounded deletion property.
  - **Setup**: Place `202506181400.pcap` in the `data/` folder (or specify the full path via `--mawi-path`).
  - Requires `dpkt` (recommended) or `scapy` to parse pcap. Install via `pip install dpkt` or `pip install scapy`.

---

## Dependencies

- Python 3.10+
- numpy
- pandas
- matplotlib
- seaborn
- dpkt (recommended for pcap parsing) or scapy (fallback)

Install via:
```bash
pip install -r requirements.txt
pip install dpkt   # for MAWI pcap parsing
```

---

## References

[1] Misra, J., & Gries, D. (1982). Finding repeated elements. Science of Computer Programming,
    2(2), 143–152.

[2] Metwally, A., Agrawal, D., & El Abbadi, A. (2005). Efficient computation of frequent and top-k
    elements in data streams. International Conference on Database Theory (ICDT).

[3] Cormode, G., & Muthukrishnan, S. (2005). An improved data stream summary: the count-min
    sketch and its applications. Journal of Algorithms, 55(1), 58–75.

[4] Charikar, M., Chen, K., & Farach-Colton, M. (2004). Finding frequent items in data streams.
    Theoretical Computer Science, 312(1), 3–15.

[5] Berinde, R., Indyk, P., Cormode, G., & Strauss, M. J. (2010). Space-optimal heavy hitters with
    strong error bounds. ACM Transactions on Database Systems (TODS), 35(4), 1–28.

[6] MAWI Working Group. MAWI Traffic Archive, Samplepoint-F (2025-06-18).
    https://mawi.wide.ad.jp/mawi/

[7] Jayaram, R., & Woodruff, D. P. (2018). Data streams with bounded deletions. Proceedings of the
    37th ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems (PODS),
    341–354. (arXiv:1803.08777)

[8] Zhao, F., Agrawal, D., El Abbadi, A., & Metwally, A. (2021). SpaceSaving±: An Optimal
    Algorithm for Frequency Estimation and Frequent items in the Bounded Deletion Model.
    Proceedings of the VLDB Endowment (PVLDB). (arXiv:2112.03462)

[9] Shahout, R., & Mitzenmacher, M. (2024). Learning-Based Heavy Hitters and Flow Frequency
    Estimation in Streams. arXiv preprint arXiv:2406.16270.

[10] Zhao, F., Agrawal, D., El Abbadi, A., Mathieu, C., Metwally, A., & de Rougemont, M.
     (2023). The SpaceSaving± Family of Algorithms for Data Streams with Bounded Deletions.
     Proceedings of the ACM on Management of Data, 1(2), 1–26. (arXiv:2309.12623)


## License

This project is for academic purposes (SEEM5020 course project). All rights reserved.