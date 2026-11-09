"""
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

# ---------------------------------------- 5. Lasso inside the fund-level CV
CS = [0.001, 0.003, 0.01, 0.03, 0.1, 1.0]
def lasso(C):                                  # scaler is refit inside every training fold
    return Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                               class_weight="balanced"))])

def event_recall(te, s, budget=0.05):
    flag = s >= np.quantile(s, 1 - budget)
    ev = d.loc[te].assign(flag=flag).query("is_anomaly == 1")
    return ev.groupby("event_id")["flag"].any().mean()

def run(feats):
    X, y, fund, rows = d[feats], d["is_anomaly"], d["ticker"], []
    for rep in range(5):
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
        for tr, te in outer.split(X, d["asset_class"], groups=fund):
            inner = {C: np.mean([average_precision_score(y.iloc[tr[b]],
                         lasso(C).fit(X.iloc[tr[a]], y.iloc[tr[a]]).predict_proba(X.iloc[tr[b]])[:, 1])
                         for a, b in GroupKFold(3).split(tr, groups=fund.iloc[tr])]) for C in CS}
            C = max(inner, key=inner.get)                      # C chosen on training funds only
            m = lasso(C).fit(X.iloc[tr], y.iloc[tr])
            s = m.predict_proba(X.iloc[te])[:, 1]
            rows.append({"C": C, "recall": event_recall(te, s),
                         "pr_auc": average_precision_score(y.iloc[te], s),
                         **dict(zip(feats, m.named_steps["lr"].coef_[0]))})
    return pd.DataFrame(rows)

res = {name: run(f) for name, f in [("Full (11)", FULL), ("Reduced (6)", REDUCED), ("Minimal (3)", MINIMAL)]}
print("\nLasso logistic, 25 fund-level folds (C tuned inside each training set):")
print(pd.DataFrame({k: {"recall@5% mean": v.recall.mean(), "recall SD": v.recall.std(),
                        "PR-AUC mean": v.pr_auc.mean(), "PR-AUC SD": v.pr_auc.std()}
                    for k, v in res.items()}).T.round(2).to_string())

f = res["Full (11)"]
print("\nFull set: how stable is each coefficient across the 25 folds?")
print(pd.DataFrame({"selected %": (f[FULL] != 0).mean() * 100,
                    "positive sign %": (f[FULL] > 0).mean() * 100,
                    "coef mean": f[FULL].mean(), "coef SD": f[FULL].std()})
        .round(2).sort_values("positive sign %").to_string())
print("C chosen across folds (full set):", f.C.value_counts().to_dict())

# Figure: standardized coefficients across 25 folds, full set
order = f[FULL].mean().sort_values().index
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.boxplot([f[c] for c in order], vert=False, labels=order)
ax.axvline(0, color="grey", lw=1, ls="--")
ax.set_xlabel("standardized L1 coefficient (25 fund-level folds)")
ax.set_title("Which coefficients keep their sign? Full feature set")
plt.tight_layout(); plt.savefig("week6_coef_stability.png", dpi=150)
pd.concat(res, names=["feature_set"]).to_csv("week6_lasso_folds.csv")
