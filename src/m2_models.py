"""
m2_models.py: rules, course models, the hybrid, and the fold machinery for Milestone 2.

Every model is fit on training funds only. Scaling sits inside each pipeline.
Hyperparameters for the tree models are fixed at the values chosen by nested
cross-validation in the Week 8 experiment (src/experiments/week8_trees_ensembles.py).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest

from nav_m2 import (M1_FEATURES, FULL_FEATURES, ERROR_TYPES, MAIN_BUDGET,
                    review_top, event_hits, pct_rank)

HYBRID_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]

# ----------------------------------------------------------------------------- rules
FLAT_MOVE = 0.002     # benchmark moved at least 20 bps while the NAV did not move


def rule_scores(d):
    """Label-free rule scores. Higher means review first."""
    flat = (d["zero_move"] == 1) & (d["bench_abs"] >= FLAT_MOVE)
    ex_gap = (d["is_ex"] == 1) & (d["ex_ratio"] > 0.5)   # NAV did not fall by the distribution
    return {
        "Rule: raw move": d["abs_ret_obs"].to_numpy(),
        "Rule: benchmark-adjusted move": d["abs_adj_ret"].to_numpy(),
        "Rule: expert checks": (flat | ex_gap).astype(float).to_numpy() + d["abs_adj_ret"].to_numpy(),
    }


def rule_natural_flags(d, thresh=0.005):
    """Rules at their own 50 bps threshold, with no review budget."""
    flat = (d["zero_move"] == 1) & (d["bench_abs"] >= FLAT_MOVE)
    ex_gap = (d["is_ex"] == 1) & (d["ex_ratio"] > 0.5)
    return {
        "Rule: raw move > 50 bps": (d["abs_ret_obs"] > thresh).to_numpy(),
        "Rule: benchmark-adjusted move > 50 bps": (d["abs_adj_ret"] > thresh).to_numpy(),
        "Rule: expert checks (flat NAV, ex-date gap, or > 50 bps)":
            (flat | ex_gap | (d["abs_adj_ret"] > thresh)).to_numpy(),
    }


# ----------------------------------------------------------------------------- models
def make_supervised(seed=0, C=1.0):
    """name -> (estimator, feature list)."""
    return {
        "Logistic, M1 features": (make_pipeline(StandardScaler(), LogisticRegression(
            C=C, class_weight="balanced", max_iter=2000)), M1_FEATURES),
        "Logistic, full features": (make_pipeline(StandardScaler(), LogisticRegression(
            C=C, class_weight="balanced", max_iter=2000)), FULL_FEATURES),
        "Gaussian Naive Bayes": (GaussianNB(), FULL_FEATURES),
        "Decision tree": (DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                 class_weight="balanced", random_state=seed), FULL_FEATURES),
        "Random forest": (RandomForestClassifier(n_estimators=150, min_samples_leaf=20,
                                                 class_weight="balanced_subsample", n_jobs=-1,
                                                 random_state=seed), FULL_FEATURES),
        "Gradient boosting": (HistGradientBoostingClassifier(learning_rate=0.03, max_leaf_nodes=7,
                                                             max_iter=200, class_weight="balanced",
                                                             early_stopping=False, random_state=seed),
                              FULL_FEATURES),
    }


def fit_score(est, feats, train, test):
    est.fit(train[feats].to_numpy(), train["is_anomaly"].to_numpy())
    return est.predict_proba(test[feats].to_numpy())[:, 1]


def iforest_scores(train, test, seed=0):
    """Isolation forest fit without labels on the training funds."""
    m = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
    m.fit(train[FULL_FEATURES].to_numpy())
    return -m.score_samples(test[FULL_FEATURES].to_numpy())


def lr_full(C=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000))


def drop_types(frame, types):
    """Remove the erroneous days of the given error types (used for unseen-type tests)."""
    if not types:
        return frame
    return frame[~frame["atype"].isin(types)]


# ----------------------------------------------------------------------------- folds
def group_folds(d, n_splits, seed):
    """Fold assignment over portfolio groups, stratified by asset class."""
    groups = d[["group", "asset_class"]].drop_duplicates("group").sort_values("group")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of = {}
    for f, (_, te) in enumerate(skf.split(groups["group"], groups["asset_class"])):
        for g in groups["group"].iloc[te]:
            fold_of[g] = f
    return d["group"].map(fold_of).to_numpy()


def recall_on(frame, flag, types=None):
    hit = event_hits(frame, flag)
    if types is not None:
        t = frame.loc[frame["is_anomaly"] == 1].groupby("event_id")["atype"].first()
        hit = hit[t.reindex(hit.index).isin(types).to_numpy()]
    return float(hit.mean()) if len(hit) else np.nan


# ----------------------------------------------------------------------------- hybrid
def choose_hybrid_weight(train, seed=0, budget=MAIN_BUDGET, inner_splits=3):
    """Pick the logistic weight w for  w*rank(logistic) + (1-w)*rank(isolation forest).

    Chosen inside the training funds only. Objective, fixed in advance: the average of
    (a) event recall on inner validation funds and (b) event recall on each error
    type when that type is removed from inner training (an inner leave-one-type-out).
    Part (b) rewards robustness to error types the model has not seen.
    """
    present = [k for k in ERROR_TYPES if (train["atype"] == k).any()]
    fold = group_folds(train, inner_splits, seed + 100)
    seen = {w: [] for w in HYBRID_WEIGHTS}
    unseen = {w: [] for w in HYBRID_WEIGHTS}
    for f in range(inner_splits):
        tr, va = train[fold != f], train[fold == f]
        r_if = pct_rank(iforest_scores(tr, va, seed))
        r_lr = pct_rank(fit_score(lr_full(), FULL_FEATURES, tr, va))
        for w in HYBRID_WEIGHTS:
            seen[w].append(recall_on(va, review_top(w * r_lr + (1 - w) * r_if, budget, seed)))
        for k in present:
            r_k = pct_rank(fit_score(lr_full(), FULL_FEATURES, drop_types(tr, [k]), va))
            for w in HYBRID_WEIGHTS:
                unseen[w].append(recall_on(va, review_top(w * r_k + (1 - w) * r_if, budget, seed), [k]))
    obj = {w: 0.5 * np.nanmean(seen[w]) + 0.5 * np.nanmean(unseen[w]) for w in HYBRID_WEIGHTS}
    best = max(HYBRID_WEIGHTS, key=lambda w: (round(obj[w], 6), w))
    return best, obj


def hybrid_scores(train, test, w, seed=0, s_lr=None, s_if=None):
    if s_lr is None:
        s_lr = fit_score(lr_full(), FULL_FEATURES, train, test)
    if s_if is None:
        s_if = iforest_scores(train, test, seed)
    return w * pct_rank(s_lr) + (1 - w) * pct_rank(s_if)


# ----------------------------------------------------------------------------- statistics
def nadeau_bengio(diffs, test_train_ratio):
    """Corrected resampled t-test (Nadeau & Bengio, 2003) for J paired fold differences."""
    from scipy import stats
    d = np.asarray(diffs, float)
    J = len(d)
    var = d.var(ddof=1)
    se_corr = np.sqrt((1 / J + test_train_ratio) * var)
    se_naive = np.sqrt(var / J)
    t = d.mean() / se_corr if se_corr > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), J - 1)
    return dict(mean_diff=d.mean(), se_naive=se_naive, se_corrected=se_corr, t=t, p=p, J=J)
