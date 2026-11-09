"""
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
from sklearn.model_selection import StratifiedGroupKFold, KFold
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

# ---------------------------------------- 4. One procedure, scored on a test set
def evaluate(tr, te, budget=0.05):
    m = HistGradientBoostingClassifier(class_weight="balanced", random_state=0)
    m.fit(X.iloc[tr], y.iloc[tr])
    s = pd.Series(m.predict_proba(X.iloc[te])[:, 1], index=te)
    flag = s >= s.quantile(1 - budget)                    # review the top 5% of fund-days
    ev = data.loc[te].assign(flag=flag.values).query("is_anomaly == 1")
    return {"event_recall_5pct": ev.groupby("event_id")["flag"].any().mean(),
            "pr_auc": average_precision_score(y.iloc[te], s),
            "test_events": ev["event_id"].nunique(),
            "sev1_share": ev.groupby("event_id")["sev"].first().eq(1).mean()}

CUT = pd.Timestamp("2025-01-01")
before = data["date"].values < CUT
results = []
for rep in range(5):
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
    for k, (tr, te) in enumerate(sgk.split(X, data["asset_class"], groups=fund)):
        results.append({"design": "fund-level CV", "rep": rep, "fold": k, **evaluate(tr, te)})
        results.append({"design": "fund-level + time-forward", "rep": rep, "fold": k,
                        **evaluate(tr[before[tr]], te[~before[te]])})
    kf = KFold(n_splits=5, shuffle=True, random_state=rep)
    for k, (tr, te) in enumerate(kf.split(X)):
        results.append({"design": "row-level CV", "rep": rep, "fold": k, **evaluate(tr, te)})
res = pd.DataFrame(results)

one = res.query("design == 'fund-level CV' and rep == 0 and fold == 0").iloc[0]
print(f"\nOne fund-level 80/20 split: event recall@5% = {one.event_recall_5pct:.2f}, "
      f"PR-AUC = {one.pr_auc:.2f} ({int(one.test_events)} test events)")

order = ["row-level CV", "fund-level CV", "fund-level + time-forward"]
summ = (res.groupby("design")
          .agg(recall_mean=("event_recall_5pct", "mean"), recall_sd=("event_recall_5pct", "std"),
               recall_min=("event_recall_5pct", "min"), recall_max=("event_recall_5pct", "max"),
               prauc_mean=("pr_auc", "mean"), prauc_sd=("pr_auc", "std"),
               events_min=("test_events", "min"), events_max=("test_events", "max"))
          .loc[order].round(2))
print("\n25 estimates per design (5 folds x 5 repeats):")
print(summ.to_string())

fl = res.query("design == 'fund-level CV'")
p, n = fl.event_recall_5pct.mean(), fl.test_events.mean()
print(f"\nFund-level CV: counting noise alone (binomial SE at ~{n:.0f} events) = "
      f"{np.sqrt(p * (1 - p) / n):.3f} vs observed SD = {fl.event_recall_5pct.std():.3f}")

print(f"Fund-level CV: correlation of fold recall with share of subtle (severity 1) events = "
      f"{fl.event_recall_5pct.corr(fl.sev1_share):.2f}")

# Figure: the 25 estimates per design
fig, ax = plt.subplots(figsize=(7.5, 3.6))
rng_j = np.random.default_rng(1)
for i, d in enumerate(order):
    v = res.loc[res.design == d, "event_recall_5pct"]
    ax.scatter(v, i + rng_j.uniform(-0.1, 0.1, len(v)), alpha=0.6, s=30)
    ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color="black", lw=2)
ax.scatter([one.event_recall_5pct], [order.index("fund-level CV")], marker="*", s=260,
           color="#c0392b", zorder=5, label="the single 80/20 split")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
ax.set_xlabel("event recall at a 5% review budget"); ax.legend(loc="lower left")
ax.set_title("Same procedure, 25 evaluations per design (black bar = mean)")
plt.tight_layout(); plt.savefig("week5_estimate_spread.png", dpi=150)
res.to_csv("week5_cv_results.csv", index=False)
