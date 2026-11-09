"""
ANLY 530 - Week 7, Part B (TEST): neighbors or probabilities?
Same data, target, three features (Week 6 decision), and 25 fund-level folds
as Weeks 5-6. Compared: kNN (k = 5, 25, 100, and k tuned on training funds only),
Gaussian Naive Bayes, and the logistic baseline. Metric: event recall when the analyst reviews exactly 5%
of fund-days (ties broken at random), plus PR-AUC and a calibration check.

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import brier_score_loss
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


# ---------------------------------------- 5. Week 7 comparison
F = ["abs_z60", "zero_x_bench", "is_ex"]                     # Week 6 decision
X, y, fund = d[F], d["is_anomaly"], d["ticker"]

print("\nWhy scaling is a decision for kNN (after StandardScaler):")
Zs = StandardScaler().fit_transform(X)
print(f"  an ex-date sits {Zs[d.is_ex.values == 1, 2].max():.1f} SD from a normal day; "
      f"largest |z60| = {Zs[:, 0].max():.0f} SD; largest flat-on-moving-day = {Zs[:, 1].max():.0f} SD")

print("\nIs NB's conditional independence plausible? Spearman WITHIN each class:")
for lab, sub in [("normal", d[y == 0]), ("error ", d[y == 1])]:
    c = sub[F].corr(method="spearman")
    print(f"  {lab}: z60~flat {c.iloc[0, 1]:+.2f} | z60~ex {c.iloc[0, 2]:+.2f} | flat~ex {c.iloc[1, 2]:+.2f}")

MODELS = {                                   # scaler sits inside the pipeline, refit per fold
    "kNN, k=5":   lambda: Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier(5))]),
    "kNN, k=25":  lambda: Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier(25))]),
    "kNN, k=100": lambda: Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier(100))]),
    "Gaussian NB": lambda: GaussianNB(),     # NB needs no scaling: it models each feature separately
    "Logistic (baseline)": lambda: Pipeline([("scale", StandardScaler()),
                                             ("lr", LogisticRegression(class_weight="balanced", max_iter=2000))]),
}

def review_top(s, budget=0.05, seed=0):
    """Flag exactly 5% of fund-days; ties in the score are broken at random."""
    n = int(np.ceil(budget * len(s)))
    tiebreak = np.random.default_rng(seed).random(len(s))
    order = np.lexsort((tiebreak, -s))                         # sort by score desc, then random
    flag = np.zeros(len(s), bool); flag[order[:n]] = True
    return flag

def event_recall(idx, s, seed=0):
    ev = d.loc[idx].assign(flag=review_top(s, seed=seed)).query("is_anomaly == 1")
    return ev.groupby("event_id")["flag"].any().mean()

K_GRID = [25, 100, 250, 500, 1000]
def knn(k): return Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier(k))])
def tuned_k(tr):                                   # choose k on training funds only (inner folds)
    inner = {k: np.mean([event_recall(tr[b], knn(k).fit(X.iloc[tr[a]], y.iloc[tr[a]])
                                      .predict_proba(X.iloc[tr[b]])[:, 1])
                         for a, b in GroupKFold(3).split(tr, groups=fund.iloc[tr])]) for k in K_GRID}
    return max(inner, key=inner.get)

MODELS = dict(list(MODELS.items())[:3] + [("kNN, k tuned", None)] + list(MODELS.items())[3:])
rows, chosen_k = [], []
for rep in range(5):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
    for k, (tr, te) in enumerate(outer.split(X, d["asset_class"], groups=fund)):
        yt = y.iloc[te].values
        for name, make in MODELS.items():
            if make is None:
                kb = tuned_k(tr); chosen_k.append(kb); make = lambda kb=kb: knn(kb)
            s = make().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
            flag = review_top(s, seed=rep * 10 + k)
            naive = s >= np.quantile(s, 0.95)                  # the usual "top 5%" cutoff, ties included
            ev = d.loc[te].assign(flag=flag).query("is_anomaly == 1")
            rows.append({"model": name,
                         "recall": ev.groupby("event_id")["flag"].any().mean(),
                         "pr_auc": average_precision_score(yt, s),
                         "distinct_scores": len(np.unique(s)),
                         "naive_flag_rate": naive.mean(),
                         "brier": brier_score_loss(yt, s),
                         "mean_p_flagged": s[flag].mean(),          # what the model claims for its flags
                         "actual_rate_flagged": yt[flag].mean()})   # what is actually there
res = pd.DataFrame(rows)

order = list(MODELS)
summ = (res.groupby("model")
          .agg(recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
               prauc_mean=("pr_auc", "mean"), prauc_sd=("pr_auc", "std"),
               distinct_scores=("distinct_scores", "median"),
               naive_flag_pct=("naive_flag_rate", lambda v: 100 * v.mean()),
               brier=("brier", "mean"),
               p_claimed=("mean_p_flagged", "mean"), p_actual=("actual_rate_flagged", "mean"))
          .loc[order].round(3))
print("\n25 fund-level folds; exact 5% review budget (observed error rate = "
      f"{y.mean():.1%}):")
print(summ.to_string())
print(f"k chosen on training funds: {pd.Series(chosen_k).value_counts().to_dict()}")
print(f"Brier score of always predicting the 1.0% base rate: {y.mean() * (1 - y.mean()):.4f}")
res.to_csv("week7_fold_results.csv", index=False)

# Figure: recall at the 5% budget, 25 folds per model
fig, ax = plt.subplots(figsize=(7.5, 3.8))
rng_j = np.random.default_rng(1)
for i, m in enumerate(order):
    v = res.loc[res.model == m, "recall"]
    ax.scatter(v, i + rng_j.uniform(-0.12, 0.12, len(v)), alpha=0.55, s=26)
    ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color="black", lw=2)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
ax.set_xlabel("event recall when reviewing exactly 5% of fund-days")
ax.set_title("Same features, same 25 fund-level folds (black bar = mean)")
plt.tight_layout(); plt.savefig("week7_model_comparison.png", dpi=150)
