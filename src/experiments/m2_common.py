"""Shared helpers for the Milestone 2 experiment scripts."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import nav_m2 as m
import m2_models as mm


def development_set(d):
    """Development data: development funds, development period (before Dec 2024)."""
    return d[(~d["holdout_fund"]) & (d["period"] == "past")].reset_index(drop=True)


def score_all(train, test, seed, exclude_types=(), hybrid_w=None, log=None):
    """Scores for every method on `test` after fitting on `train`.

    exclude_types: error types whose erroneous days are removed from training
    (the unseen-type test). Rules and the isolation forest never use labels.
    """
    tr = mm.drop_types(train, list(exclude_types))
    scores = dict(mm.rule_scores(test))
    timing = {}
    for name, (est, feats) in mm.make_supervised(seed).items():
        t0 = time.time()
        scores[name] = mm.fit_score(est, feats, tr, test)
        timing[name] = time.time() - t0
    t0 = time.time()
    s_if = mm.iforest_scores(tr, test, seed)
    scores["Isolation forest (no labels)"] = s_if
    if hybrid_w is None:
        hybrid_w, _ = mm.choose_hybrid_weight(tr, seed)
    scores["Hybrid (logistic + isolation forest)"] = mm.hybrid_scores(
        tr, test, hybrid_w, seed, s_lr=scores["Logistic, full features"], s_if=s_if)
    timing["hybrid+iforest"] = time.time() - t0
    if log:
        log(f"    fit times: " + ", ".join(f"{k.split(',')[0][:12]} {v:.0f}s" for k, v in timing.items()))
    return scores, hybrid_w


def event_table(frame):
    """One row per event present in `frame`: type, severity, size, asset class."""
    e = frame[frame["is_anomaly"] == 1].groupby("event_id").agg(
        atype=("atype", "first"), sev=("sev", "first"), size_bps=("size_bps", "max"),
        asset_class=("asset_class", "first"), ticker=("ticker", "first"), days=("is_anomaly", "size"))
    return e


def log_to(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(path, "a") as fh:
            fh.write(line + "\n")
    return log
