# NAV Anomaly Detection for Mutual Funds and ETFs

ANLY 530 (Principles and Applications of Machine Learning), Harrisburg University, Fall 2026.
Conference project by Agastya Chowdhary.

## Research question

Can a simple, interpretable pre-release model rank each day's mutual fund and ETF net asset values (NAVs) so that a fixed 5% analyst review budget catches more NAV errors than the industry's 50 basis point rule? And do those gains hold on funds, time periods, and error types the model never saw in training?

## Repository layout

```
nav-anomaly-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── 01_exploration.ipynb      # builds the data and target, shows baselines at a 5% budget
├── src/
│   ├── nav_data.py               # download, error injection, point-in-time features, helpers
│   └── experiments/              # weekly ANLY 530 experiments (self-contained scripts)
│       ├── nav_target_test.py            # Week 4: target and baselines
│       ├── week5_eval.py                 # Week 5: variability of the estimate, row vs fund vs time splits
│       ├── week6_representation.py       # Week 6: Lasso and feature stability
│       ├── week7_neighbors_vs_bayes.py   # Week 7: kNN and Naive Bayes vs logistic
│       └── week8_trees_ensembles.py      # Week 8: trees and ensembles, leave-one-error-type-out
└── figures/
    └── target_distribution.png
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_exploration.ipynb
# or run any experiment directly, for example:
python src/experiments/week8_trees_ensembles.py
```

The notebook takes a few minutes; most of the time is the data download. All randomness uses seed 42.

## Data

- **Source:** daily closes, dividends, and capital gains for 26 U.S. mutual funds and ETFs (U.S. equity, bond, international) plus three benchmark ETFs (SPY, AGG, EFA), May 2021 to August 2026, from Yahoo Finance through the `yfinance` package.
- **No data is committed.** Everything is downloaded at run time. Yahoo revises adjusted histories, so exact counts can shift slightly between downloads.
- **Labels are synthetic.** Real NAV errors are not public, so four error types (stale price, missed split, missed dividend, expense error) are injected at three severities on about 1% of fund-days. The first correct day after each error is excluded from training and scoring.
- **No proprietary data.** No fund administrator, client, or employer data is used at any stage.

## Evaluation design

- Unit: one fund on one trading day. Target: `is_anomaly`.
- Splits hold out whole funds (repeated 5-fold, stratified by asset class); a time-forward holdout of unseen funds is the final check.
- Primary metric: event recall when an analyst reviews exactly 5% of fund-days. Secondary: PR-AUC, calibration.
- Scaling and tuning happen inside the training folds only.
