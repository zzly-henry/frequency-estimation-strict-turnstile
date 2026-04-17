#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh
# Master script to reproduce all experiments for the SEEM5020 project.
#
# Usage:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Frequency Estimation under Strict Turnstile Model"
echo "  with α-Bounded Deletion — Full Experiment Pipeline"
echo "============================================================"
echo ""

# ------------------------------------------------------------------
# 0. Environment check
# ------------------------------------------------------------------
echo "[0/4] Checking environment..."

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.10+."
    exit 1
fi

PYTHON=python3
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python version: $PY_VERSION"

# Install dependencies if needed
echo "  Installing/checking dependencies..."
$PYTHON -m pip install --quiet -r requirements.txt
echo "  Dependencies OK."
echo ""

# Create output directories
mkdir -p results/parametric
mkdir -p results/dataset
mkdir -p results/advanced
mkdir -p data

# Create __init__.py files for package imports
touch algorithms/__init__.py
touch data_generators/__init__.py
touch experiments/__init__.py
touch utils/__init__.py

# ------------------------------------------------------------------
# 1. Parametric Evaluation
# ------------------------------------------------------------------
echo "============================================================"
echo "[1/4] Running Parametric Evaluation..."
echo "  (Varying N ∈ {10⁵, 5×10⁵, 10⁶, 2×10⁶} and α ∈ {1.5, 2, 4, 8})"
echo "============================================================"
echo ""

$PYTHON -m experiments.parametric_eval

echo ""
echo "  ✓ Parametric evaluation complete."
echo "    Results: results/parametric/"
echo ""

# ------------------------------------------------------------------
# 2. Dataset Evaluation
# ------------------------------------------------------------------
echo "============================================================"
echo "[2/4] Running Dataset Evaluation..."
echo "  (Zipf s∈{1.0,1.5,2.0}, Uniform, Binomial, CAIDA, YCSB)"
echo "============================================================"
echo ""

$PYTHON -m experiments.dataset_eval

echo ""
echo "  ✓ Dataset evaluation complete."
echo "    Results: results/dataset/"
echo ""

# ------------------------------------------------------------------
# 3. Advanced Evaluation (Learned-Integrated SpaceSaving±)
# ------------------------------------------------------------------
echo "============================================================"
echo "[3/4] Running Advanced Evaluation..."
echo "  (Learned-ISS± vs baselines, hyperparameter sweeps)"
echo "============================================================"
echo ""

$PYTHON -m experiments.advanced_eval

echo ""
echo "  ✓ Advanced evaluation complete."
echo "    Results: results/advanced/"
echo ""

# ------------------------------------------------------------------
# 4. Generate Report (placeholder)
# ------------------------------------------------------------------
echo "============================================================"
echo "[4/4] Generating Report..."
echo "============================================================"
echo ""

# Generate a simple text-based summary report
REPORT_FILE="results/report_summary.txt"
cat > "$REPORT_FILE" << 'REPORT_HEADER'
=============================================================================
 SEEM5020 Project Report Summary
 Frequency Estimation under Strict Turnstile Model with α-Bounded Deletion
=============================================================================

This file summarizes the experimental results. Full plots and CSV data are
available in the results/ subdirectories.

Experiments Conducted:
  1. Parametric Evaluation    → results/parametric/
  2. Dataset Evaluation       → results/dataset/
  3. Advanced Evaluation      → results/advanced/

REPORT_HEADER

# Append CSV summaries if they exist
if [ -f "results/parametric/parametric_results.csv" ]; then
    echo "" >> "$REPORT_FILE"
    echo "--- Parametric Results (first 20 rows) ---" >> "$REPORT_FILE"
    head -21 "results/parametric/parametric_results.csv" >> "$REPORT_FILE"
fi

if [ -f "results/dataset/dataset_results.csv" ]; then
    echo "" >> "$REPORT_FILE"
    echo "--- Dataset Results (first 20 rows) ---" >> "$REPORT_FILE"
    head -21 "results/dataset/dataset_results.csv" >> "$REPORT_FILE"
fi

if [ -f "results/advanced/advanced_summary.csv" ]; then
    echo "" >> "$REPORT_FILE"
    echo "--- Advanced Results Summary ---" >> "$REPORT_FILE"
    cat "results/advanced/advanced_summary.csv" >> "$REPORT_FILE"
fi

echo "  Report summary written to $REPORT_FILE"

# If report.pdf doesn't exist, create a placeholder note
if [ ! -f "report.pdf" ]; then
    echo "" > report.pdf
    echo "  Note: report.pdf is a placeholder. Compile from results/ and plots."
fi

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Output files:"
echo "  results/parametric/parametric_results.csv"
echo "  results/parametric/*.png  (error_vs_N, error_vs_alpha, space_vs_alpha, ...)"
echo "  results/dataset/dataset_results.csv"
echo "  results/dataset/*.png    (error_by_dataset, hh_performance, ...)"
echo "  results/advanced/advanced_results.csv"
echo "  results/advanced/*.png   (baseline_comparison, fixed_ratio, ...)"
echo "  results/report_summary.txt"
echo ""
echo "Total plots generated:"
find results/ -name "*.png" 2>/dev/null | wc -l | xargs echo " "
echo ""
echo "Done."