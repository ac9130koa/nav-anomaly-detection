"""
ANLY 530 - Week 8, TEST: does tree complexity earn its place?
Same data, injected target, three Week 6 features, and 25 fund-level folds
as Weeks 5-7. Compared on identical folds: logistic baseline (untuned, as in
Week 7), a single decision tree, a random forest, and histogram gradient
boosting, each tree model tuned on training funds only (3 inner group folds).
Metric: event recall when the analyst reviews exactly 5% of fund-days, plus
PR-AUC, Brier, model size, impurity vs held-out permutation importance, and a
leave-one-error-type-out check (train without a type, test on it).

(Week 6 header kept below for reference.)
ANLY 530 - Week 6, Part B (TEST): what should the model learn from?
Same data, target, and fund-level CV as Weeks 4-5. One procedure: L1 (Lasso)
logistic regression, with C tuned on training funds only (nested), fitted on
three feature sets. Question: do the overlapping "move size" features add
anything, and are Lasso's selections stable enough to interpret?

(Week 5 header kept below for reference.)
ANLY 530 - Week 5, Part B (TEST): does the estimate move?
Same data and injected target as Week 4. One modeling procedure
(gradient-boosted classifier), evaluated three ways:
  1. one fund-level 80/20 split
  2. repeated fund-level cross-validation (5 folds x 5 repeats)
  3. the same CV with random row-level folds (leakage-prone contrast)
  4. fund-level CV that also holds out time (train < 2025, test >= 2025)
Primary metric: event recall when the review budget is 5% of fund-days.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score

SEED = 42
START, END = "2021-05-01", "2026-09-01"
INJECT_RATE = 0.01          # 1% of eligible fund-days
THRESH = 0.005              # 50 bps, the industry materiality convention

# Preliminary universe: 26 funds, 3 asset classes, mutual funds + ETFs
BENCH = {"us_equity": "SPY", "bond": "AGG", "intl": "EFA"}
FUNDS = {
    "us_equity": ["VFIAX", "FXAIX", "VTSAX", "AGTHX", "PRBLX", "IVV", "VTI", "QQQ", "IWM", "DIA"],
    "bond":      ["VBTLX", "PTTRX", "DODIX", "VWITX", "BND", "LQD", "TLT", "TIP"],
    "intl":      ["VTIAX", "DODFX", "VTMGX", "VEIEX", "VEA", "IEFA", "VWO", "EEM"],
}
fund_class = {t: c for c, ts in FUNDS.items() for t in ts}

# ---------------------------------------------------------------- 1. Load
tickers = list(fund_class) + list(BENCH.values())
raw = yf.download(tickers, start=START, end=END, auto_adjust=False,
                  actions=True, progress=False)
close = raw["Close"]
dist = raw["Dividends"].fillna(0)
if "Capital Gains" in raw.columns.get_level_values(0):
    dist = dist + raw["Capital Gains"].reindex(columns=close.columns).fillna(0)

bench_ret = {c: close[b].pct_change() for c, b in BENCH.items()}

# ------------------------------------------------ 2. Build target by injection
rng = np.random.default_rng(SEED)
rows = []
for t, cls in fund_class.items():
    df = pd.DataFrame({"close": close[t], "dist": dist[t]}).dropna(subset=["close"])
    df["ticker"], df["asset_class"] = t, cls
    df["bench_ret"] = bench_ret[cls].reindex(df.index)
    rows.append(df.iloc[1:].assign(prev_close=df["close"].shift(1).iloc[1:]))
panel = pd.concat(rows).reset_index(names="date")

panel["obs"] = panel["close"]                      # released NAV
panel["is_anomaly"], panel["echo"] = 0, 0
panel["atype"], panel["sev"] = "none", 0
panel["event_id"] = -1

n_events = int(INJECT_RATE * len(panel) / 1.25)    # stale events span >1 row
types = ["stale_price", "missed_split", "missed_dividend", "expense_error"]
split_err = {1: -0.0025, 2: -0.0067, 3: +0.04}     # split ignored on a small position
exp_err = {1: -0.0005, 2: -0.0015, 3: -0.0050}     # expense accrual overstated

used = np.zeros(len(panel), bool)
idx_by_fund = panel.groupby("ticker").indices
done = {k: 0 for k in types}
for k in types:
    target = n_events // 4
    for _ in range(50_000):
        if done[k] >= target:
            break
        fund = rng.choice(list(idx_by_fund))
        full = idx_by_fund[fund]
        ix = full
        if k == "missed_dividend":                 # only real ex-dates qualify
            ix = ix[panel.loc[ix, "dist"].values > 0]
            if len(ix) == 0:
                continue
        i = int(rng.choice(ix))
        if k == "missed_dividend":
            y = panel.at[i, "dist"] / panel.at[i, "prev_close"]
            sev = 1 if y < 0.002 else (2 if y < 0.01 else 3)
        else:
            sev = int(rng.integers(1, 4))
        span = sev if k == "stale_price" else 1
        lo, hi = i - 1, i + span + 1                # buffer + one clean echo day
        if lo < full.min() or hi > full.max() or used[lo:hi + 1].any():
            continue
        for j in range(i, i + span):
            if k == "stale_price":
                panel.at[j, "obs"] = panel.at[i - 1, "close"]
            elif k == "missed_split":
                panel.at[j, "obs"] = panel.at[j, "close"] * (1 + split_err[sev])
            elif k == "missed_dividend":
                panel.at[j, "obs"] = panel.at[j, "close"] + panel.at[j, "dist"]
            else:
                panel.at[j, "obs"] = panel.at[j, "close"] * (1 + exp_err[sev])
            panel.at[j, "is_anomaly"] = int(panel.at[j, "obs"] != panel.at[j, "close"])
            panel.at[j, "atype"], panel.at[j, "sev"] = k, sev
            panel.at[j, "event_id"] = sum(done.values())
        panel.at[i + span, "echo"] = 1             # first correct day after the error
        used[lo:hi + 1] = True
        done[k] += 1

# Observed total return the control sees (distribution calendar is known)
prev_obs = panel.groupby("ticker")["obs"].shift(1).fillna(panel["prev_close"])
panel["ret_obs"] = (panel["obs"] + panel["dist"]) / prev_obs - 1
panel["adj_ret"] = panel["ret_obs"] - panel["bench_ret"]
panel = panel.dropna(subset=["adj_ret"])

# ---------------------------------------- 3. Point-in-time features per fund
g = panel.groupby("ticker")
past = lambda s, f: g[s].transform(lambda x: getattr(x.shift(1).rolling(60), f)())
panel["z60"] = (panel["adj_ret"] - past("adj_ret", "mean")) / past("adj_ret", "std")
panel["abs_z60"] = panel["z60"].abs()
panel["zero_move"] = (panel["ret_obs"] - panel["dist"] / prev_obs.loc[panel.index]).abs().lt(1e-9).astype(int)
panel["stale_run"] = panel.groupby(["ticker", (panel["zero_move"] == 0).cumsum()])["zero_move"].cumsum()
panel["bench_abs"] = panel["bench_ret"].abs()
panel["is_ex"] = (panel["dist"] > 0).astype(int)
panel["prev_adj"] = g["adj_ret"].shift(1)
panel["vs_class_med"] = panel["ret_obs"] - panel.groupby(["date", "asset_class"])["ret_obs"].transform("median")
panel["cls"] = panel["asset_class"].astype("category").cat.codes
FEATS = ["ret_obs", "adj_ret", "z60", "abs_z60", "zero_move", "stale_run", "bench_abs",
         "is_ex", "prev_adj", "vs_class_med", "cls"]

# Drop the 60-day warm-up; correction ("echo") days are left out of training and scoring
data = panel.dropna(subset=["z60"]).query("echo == 0").reset_index(drop=True)
X, y, fund = data[FEATS], data["is_anomaly"], data["ticker"]
print(f"Modeling rows: {len(data):,} | funds: {fund.nunique()} | positives: {y.sum()} "
      f"({y.mean():.2%}) | events: {data.loc[y == 1, 'event_id'].nunique()}")


# ---------------------------------------- 4. Candidate representation
d = data
for c in ["ret_obs", "adj_ret", "vs_class_med", "prev_adj"]:
    d["abs_" + c] = d[c].abs()                                 # errors move NAV either way
d["zero_x_bench"] = d["zero_move"] * d["bench_abs"]            # flat NAV on a day the market moved
d["flat_rate_60"] = (d.groupby("ticker")["zero_move"]          # how often this fund is flat (past only)
                      .transform(lambda x: x.shift(1).rolling(60, min_periods=20).mean()).fillna(0))
FULL = ["abs_ret_obs", "abs_adj_ret", "abs_vs_class_med", "abs_z60", "abs_prev_adj", "bench_abs",
        "zero_move", "stale_run", "zero_x_bench", "flat_rate_60", "is_ex"]
REDUCED = ["abs_z60", "abs_vs_class_med", "zero_move", "zero_x_bench", "stale_run", "is_ex"]
MINIMAL = ["abs_z60", "zero_x_bench", "is_ex"]

clean = d[d.is_anomaly == 0]
print(f"Clean fund-days with an unchanged NAV: {clean.zero_move.sum():,} of {len(clean):,} "
      f"({clean.zero_move.mean():.1%}); top funds: "
      f"{clean[clean.zero_move == 1].ticker.value_counts().head(4).to_dict()}")
rho = d[["abs_adj_ret", "abs_z60", "abs_vs_class_med"]].corr(method="spearman")
print(f"Spearman: |adj_ret| vs |z60| = {rho.iloc[0, 1]:.2f}, |adj_ret| vs |vs_class_med| = {rho.iloc[0, 2]:.2f}")


# ======================================================================
# Week 8: does tree complexity earn its place?
# ======================================================================
import os, time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss
from sklearn.base import clone

F = ["abs_z60", "zero_x_bench", "is_ex"]                     # Week 6 representation
X, y, fund = d[F], d["is_anomaly"], d["ticker"]
QUICK = os.environ.get("W8_QUICK") == "1"                    # dev switch: 1 repeat only
REPS = 1 if QUICK else 5

def review_top(s, budget=0.05, seed=0):
    """Flag exactly 5% of fund-days; ties in the score are broken at random."""
    n = int(np.ceil(budget * len(s)))
    order = np.lexsort((np.random.default_rng(seed).random(len(s)), -s))
    flag = np.zeros(len(s), bool); flag[order[:n]] = True
    return flag

# Candidate models. Logistic stays untuned, as in Week 7. Each tree model gets a
# small grid, chosen on training funds only (3 inner group folds, PR-AUC).
CANDIDATES = {
    "Logistic (baseline)": (Pipeline([("scale", StandardScaler()),
                                       ("lr", LogisticRegression(class_weight="balanced", max_iter=2000))]),
                            [{}]),
    "Decision tree": (DecisionTreeClassifier(class_weight="balanced", random_state=0),
                      [{"max_depth": md, "min_samples_leaf": ml} for md in [2, 3, 4, 6] for ml in [20, 100]]),
    "Random forest": (RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample",
                                             n_jobs=-1, random_state=0),
                      [{"min_samples_leaf": ml} for ml in [20, 100]]),
    "Gradient boosting": (HistGradientBoostingClassifier(class_weight="balanced", max_iter=200,
                                                         early_stopping=False, random_state=0),
                          [{"learning_rate": lr, "max_leaf_nodes": ln} for lr in [0.03, 0.1] for ln in [7, 15]]),
}

def leaves(m):
    """Model size: coefficients for logistic, total leaves for trees."""
    if isinstance(m, Pipeline):
        return m.named_steps["lr"].coef_.size + 1
    if isinstance(m, DecisionTreeClassifier):
        return m.get_n_leaves()
    if isinstance(m, RandomForestClassifier):
        return sum(t.get_n_leaves() for t in m.estimators_)
    return sum(p.get_n_leaf_nodes() for it in m._predictors for p in it)   # HistGB

def fit_tuned(base, grid, tr):
    if len(grid) == 1:
        return clone(base).set_params(**grid[0]).fit(X.iloc[tr], y.iloc[tr]), grid[0]
    inner = GroupKFold(n_splits=3)
    scores = []
    for params in grid:
        sc = [average_precision_score(y.iloc[tr[b]],
                  clone(base).set_params(**params).fit(X.iloc[tr[a]], y.iloc[tr[a]])
                  .predict_proba(X.iloc[tr[b]])[:, 1])
              for a, b in inner.split(tr, groups=fund.iloc[tr])]
        scores.append(np.mean(sc))
    best = grid[int(np.argmax(scores))]
    return clone(base).set_params(**best).fit(X.iloc[tr], y.iloc[tr]), best

TYPES = ["stale_price", "missed_split", "missed_dividend", "expense_error"]
rows, imp_rows, chosen = [], [], []
t0 = time.time()
for rep in range(REPS):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
    for k, (tr, te) in enumerate(outer.split(X, d["asset_class"], groups=fund)):
        yt = y.iloc[te].values
        for name, (base, grid) in CANDIDATES.items():
            m, params = fit_tuned(base, grid, tr)
            chosen.append({"model": name, **params})
            s = m.predict_proba(X.iloc[te])[:, 1]
            s_tr = m.predict_proba(X.iloc[tr])[:, 1]
            flag = review_top(s, seed=rep * 10 + k)
            ev = d.loc[te].assign(flag=flag).query("is_anomaly == 1")
            by_type = ev.groupby(["atype", "event_id"])["flag"].any().groupby("atype").mean()
            rows.append({"model": name, "rep": rep, "fold": k,
                         "recall": ev.groupby("event_id")["flag"].any().mean(),
                         "pr_auc": average_precision_score(yt, s),
                         "train_pr_auc": average_precision_score(y.iloc[tr], s_tr),
                         "brier": brier_score_loss(yt, s),
                         "size": leaves(m),
                         "p_claimed": s[flag].mean(), "p_actual": yt[flag].mean(),
                         **{f"recall_{t}": by_type.get(t, np.nan) for t in TYPES}})
            # held-out permutation importance (PR-AUC drop on the test funds)
            pi = permutation_importance(m, X.iloc[te], yt, scoring="average_precision",
                                        n_repeats=5, random_state=rep * 10 + k)
            imp = {"model": name, "rep": rep, "fold": k, "kind": "permutation (held-out)",
                   **dict(zip(F, pi.importances_mean))}
            imp_rows.append(imp)
            if hasattr(m, "feature_importances_"):
                imp_rows.append({"model": name, "rep": rep, "fold": k, "kind": "impurity (training)",
                                 **dict(zip(F, m.feature_importances_))})
        print(f"  outer fold {rep}-{k} done ({time.time() - t0:.0f}s)", flush=True)

res, imp = pd.DataFrame(rows), pd.DataFrame(imp_rows)
order = list(CANDIDATES)

print(f"\n{len(res) // len(order)} fund-level folds; exact 5% review budget "
      f"(observed error rate = {y.mean():.1%}):")
summ = (res.groupby("model")
          .agg(recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
               prauc_mean=("pr_auc", "mean"), prauc_sd=("pr_auc", "std"),
               train_prauc=("train_pr_auc", "mean"),
               size_median=("size", "median"), brier=("brier", "mean"),
               p_claimed=("p_claimed", "mean"), p_actual=("p_actual", "mean"))
          .loc[order].round(3))
print(summ.to_string())

print("\nRecall at the 5% budget, by error type (mean over folds):")
print(res.groupby("model")[[f"recall_{t}" for t in TYPES]].mean().loc[order].round(2).to_string())

# paired comparison against the baseline on the same folds
base_r = res[res.model == "Logistic (baseline)"].set_index(["rep", "fold"])["recall"]
for name in order[1:]:
    diff = res[res.model == name].set_index(["rep", "fold"])["recall"] - base_r
    print(f"{name:18s} minus baseline: mean {diff.mean():+.3f}, SD {diff.std():.3f}, "
          f"better in {(diff > 0).sum()}/{len(diff)} folds, worse in {(diff < 0).sum()}")

print("\nHyperparameters chosen on training funds:")
ch = pd.DataFrame(chosen)
for name in order[1:]:
    sub = ch[ch.model == name].drop(columns="model").dropna(axis=1, how="all")
    print(f"  {name}: {sub.astype(str).agg(', '.join, axis=1).value_counts().to_dict()}")

print("\nFeature importance (mean over folds; permutation = PR-AUC drop on held-out funds):")
imp_s = imp.groupby(["model", "kind"])[F].mean().round(3)
print(imp_s.to_string())
rank_flip = (imp.query("kind == 'permutation (held-out)'")
                .assign(top=lambda t: t[F].idxmax(axis=1))
                .groupby("model")["top"].value_counts().unstack(fill_value=0))
print("\nTop-ranked feature by permutation importance (count of folds):")
print(rank_flip.loc[order].to_string())

# ---------------------------------------- Leave-one-error-type-out check
# Train without one error type (its rows removed from training), test on held-out
# funds, and score recall on that unseen type only. Hyperparameters fixed at the
# setting chosen most often above.
modal = {name: (ch[ch.model == name].drop(columns="model").dropna(axis=1, how="all")
                  .astype(object).mode().iloc[0].to_dict() if len(CANDIDATES[name][1]) > 1 else {})
         for name in order}
for name, p in modal.items():
    for key, v in p.items():
        if isinstance(v, float) and v.is_integer() and key != "learning_rate":
            p[key] = int(v)
loto = []
for rep in range(REPS):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
    for k, (tr, te) in enumerate(outer.split(X, d["asset_class"], groups=fund)):
        for t in TYPES:
            keep = tr[~((d["atype"].values[tr] == t) & (y.values[tr] == 1))]
            for name in order:
                m = clone(CANDIDATES[name][0]).set_params(**modal[name]).fit(X.iloc[keep], y.iloc[keep])
                s = m.predict_proba(X.iloc[te])[:, 1]
                ev = d.loc[te].assign(flag=review_top(s, seed=rep * 10 + k)).query("is_anomaly == 1 and atype == @t")
                loto.append({"model": name, "type": t, "rep": rep, "fold": k,
                             "recall_unseen": ev.groupby("event_id")["flag"].any().mean()})
loto = pd.DataFrame(loto)
seen = res.melt(id_vars="model", value_vars=[f"recall_{t}" for t in TYPES], var_name="type",
                value_name="recall_seen").assign(type=lambda t: t["type"].str.replace("recall_", ""))
cmp_ = (seen.groupby(["model", "type"])["recall_seen"].mean().to_frame()
            .join(loto.groupby(["model", "type"])["recall_unseen"].mean()).round(2))
print(f"\nLeave-one-error-type-out (fixed hyperparameters {modal}):")
print(cmp_.unstack("type").loc[order].to_string())

res.to_csv("week8_fold_results.csv", index=False)
imp.to_csv("week8_importance.csv", index=False)
loto.to_csv("week8_leave_one_type_out.csv", index=False)

# ---------------------------------------- Figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4.1), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
size = res.groupby("model")["size"].median()
labels = [f"{m}\n({int(size[m]):,} {'parameters' if m.startswith('Logistic') else 'leaves'})" for m in order]
rng_j = np.random.default_rng(1)
for i, mname in enumerate(order):
    v = res.loc[res.model == mname, "recall"]
    ax.scatter(v, i + rng_j.uniform(-0.12, 0.12, len(v)), alpha=0.55, s=24)
    ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color="black", lw=2)
ax.set_yticks(range(len(order))); ax.set_yticklabels(labels, fontsize=9); ax.invert_yaxis()
ax.set_xlabel("event recall when reviewing exactly 5% of fund-days")
ax.set_title("Same features, same 25 fund-level folds (bar = mean)", fontsize=11)
ax = axes[1]
rf = imp[imp.model == "Random forest"].groupby("kind")[F].mean()
share = rf.div(rf.sum(axis=1), axis=0).loc[["impurity (training)", "permutation (held-out)"]]
xpos = np.arange(len(F)); w = 0.38
ax.bar(xpos - w / 2, share.iloc[0], w, label="impurity (training data)", color="#9aa5b1")
ax.bar(xpos + w / 2, share.iloc[1], w, label="permutation (held-out funds)", color="#2e86c1")
for j in range(len(F)):
    for off, v in [(-w / 2, share.iloc[0, j]), (w / 2, share.iloc[1, j])]:
        ax.text(j + off, v + 0.01, f"{v:.0%}", ha="center", fontsize=8)
ax.set_xticks(xpos); ax.set_xticklabels(["|z-score of move|", "flat NAV on\nmoving-market day", "ex-date flag"], fontsize=9)
ax.set_ylabel("share of total importance"); ax.set_ylim(0, 0.72)
ax.set_title("Random forest: what 'important' depends on", fontsize=11)
ax.legend(fontsize=8, loc="upper right", frameon=False)
plt.tight_layout(); plt.savefig("week8_trees_vs_baseline.png", dpi=150)
print(f"\nTotal runtime {time.time() - t0:.0f}s")
