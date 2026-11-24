#!/usr/bin/env bash
# Reproduce every Milestone 2 result, table, and figure.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/m2 figures
python src/experiments/m2_e0_data_audit.py     # downloads once to data/, audits data and error generator
python src/experiments/m2_e1_cv.py             # RQ1: 5x5 fund-level cross-validation
python src/experiments/m2_e3_unseen_types.py   # RQ3: leave-one-error-type-out diagnosis
python src/experiments/m2_e4_holdout.py        # RQ2: time-forward unseen-fund holdout (scored once)
python src/experiments/m2_tables_figures.py    # tables in results/m2/tables, figures in figures/
