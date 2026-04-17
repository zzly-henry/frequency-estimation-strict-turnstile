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
│   └── real_dataset_loader.py      # CAIDA and YCSB loaders (with proxies)
├── experiments/
│   ├── parametric_eval.py          # Vary N and α
│   ├── dataset_eval.py             # Diverse dataset evaluation
│   └── advanced_eval.py            # Learned-ISS± evaluation
├── utils/
│   ├── metrics.py                  # Error metrics, HH precision/recall
│   └── plotter.py                  # Publication-quality plots
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

### 2. Run all experiments

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

Or run individually:

```bash
# Parametric evaluation (vary N, α)
python -m experiments.parametric_eval

# Dataset evaluation (Zipf, Uniform, Binomial, CAIDA, YCSB)
python -m experiments.dataset_eval

# Advanced evaluation (Learned-ISS± analysis)
python -m experiments.advanced_eval
```

### 3. View results

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

# Run everything (generates CSVs, plots, and report)
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

- **Zipfian**: Generated via `data_generators/synthetic_zipf.py` with exponents s in {1.0, 1.5, 2.0}
- **Uniform**: Generated via `data_generators/synthetic_uniform.py`
- **Binomial**: Generated via `data_generators/synthetic_uniform.py` (binomial variant)

All synthetic generators ensure strict turnstile (f_e >= 0 always) and α-bounded deletion by sampling deletions from items with positive frequency.

### Real-World

- **CAIDA Anonymized Internet Traces**
  - Download: https://www.caida.org/catalog/datasets/passive_dataset/
  - Place parsed CSV at `data/caida_trace.csv` (columns: timestamp, src_ip, dst_ip, ...)
  - Element: destination IP address
  - If file not found, a synthetic Zipfian proxy (s=1.0, 50k IPs) is used automatically.

- **YCSB (Yahoo! Cloud Serving Benchmark)**
  - Download/generate: https://github.com/brianfrankcooper/YCSB
  - Place output at `data/ycsb_workload.csv` (columns: operation, key)
  - Workload: 60% INSERT / 40% UPDATE
  - If file not found, a synthetic Zipfian proxy (s=0.99, 20k keys) is used automatically.

---

## Dependencies

- Python 3.10+
- numpy
- pandas
- matplotlib
- seaborn

Install via:
```bash
pip install -r requirements.txt
```

---

## Citations

1. **Misra-Gries Algorithm**:
   Misra, J., & Gries, D. (1982). "Finding repeated elements." Science of Computer Programming, 2(2), 143-152.

2. **Space-Saving Algorithm**:
   Metwally, A., Agrawal, D., & El Abbadi, A. (2005). "Efficient computation of frequent and top-k elements in data streams." ICDT 2005.

3. **SpaceSaving± Family**:
   Dimitropoulos, X., Hurley, P., & Kind, A. (2008). "Probabilistic lossy counting: An efficient algorithm for finding heavy hitters." ACM SIGCOMM CCR.

4. **α-Bounded Deletion Property (L1 α-property)**:
   Berinde, R., Indyk, P., Cormode, G., & Strauss, M. J. (2010). "Space-optimal heavy hitters with strong error bounds." ACM TODS, 35(4), 1-28.

5. **Count-Min Sketch**:
   Cormode, G., & Muthukrishnan, S. (2005). "An improved data stream summary: the count-min sketch and its applications." Journal of Algorithms, 55(1), 58-75.

6. **Count-Sketch**:
   Charikar, M., Chen, K., & Farach-Colton, M. (2004). "Finding frequent items in data streams." Theoretical Computer Science, 312(1), 3-15.

7. **CAIDA Dataset**:
   CAIDA (2016). "The CAIDA Anonymized Internet Traces." https://www.caida.org/catalog/datasets/passive_dataset/

8. **YCSB**:
   Cooper, B. F., et al. (2010). "Benchmarking cloud serving systems with YCSB." SoCC '10.

---

## License

This project is for academic purposes (SEEM5020 course project). All rights reserved.