"""
ANLY 530 - Week discussion, Part B (TEST)
Preliminary check of the NAV anomaly target: load data, build the target,
report its distribution, plot it, and score two simple baselines.

Target (one row = one fund on one trading day):
    is_anomaly = 1 if the NAV released that day differs from the true NAV
Labels come from synthetic error injection (real NAV errors are not public).
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

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
        panel.at[i + span, "echo"] = 1             # first correct day after the error
        used[lo:hi + 1] = True
        done[k] += 1

# Observed total return the control sees (distribution calendar is known)
prev_obs = panel.groupby("ticker")["obs"].shift(1).fillna(panel["prev_close"])
panel["ret_obs"] = (panel["obs"] + panel["dist"]) / prev_obs - 1
panel["adj_ret"] = panel["ret_obs"] - panel["bench_ret"]
panel = panel.dropna(subset=["adj_ret"])

# ---------------------------------------------- 3. Target distribution
y = panel["is_anomaly"]
print(f"Rows (fund-days): {len(panel):,} | funds: {panel.ticker.nunique()} "
      f"| dates: {panel.date.min():%Y-%m-%d} to {panel.date.max():%Y-%m-%d}")
print(f"Injected events: {sum(done.values())} -> anomalous rows: {y.sum()}")
print(y.value_counts().rename({0: "normal", 1: "anomaly"}).to_frame("rows")
      .assign(pct=lambda d: (100 * d.rows / d.rows.sum()).round(2)))
print("\nAnomalous rows by type x severity:")
print(pd.crosstab(panel.loc[y == 1, "atype"], panel.loc[y == 1, "sev"], margins=True))

# ---------------------------------------------- 4. Visualization
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
vc = y.value_counts().sort_index()
ax[0].bar(["normal (0)", "anomaly (1)"], vc.values, color=["#9aa5b1", "#c0392b"])
ax[0].set_yscale("log"); ax[0].set_ylabel("fund-days (log scale)")
ax[0].set_title("Target balance: is_anomaly")
for i, v in enumerate(vc.values):
    ax[0].text(i, v * 1.15, f"{v:,}\n({100 * v / vc.sum():.2f}%)", ha="center")
ax[0].set_ylim(10, vc.max() * 6)
ct = pd.crosstab(panel.loc[y == 1, "atype"], panel.loc[y == 1, "sev"])
ct.plot(kind="bar", ax=ax[1], color=["#f5b7b1", "#e74c3c", "#922b21"], rot=0)
ax[1].set_title("Positive class by error type and severity")
ax[1].set_xlabel(""); ax[1].set_ylabel("anomalous fund-days")
ax[1].legend(title="severity", labels=["1 subtle", "2 moderate", "3 obvious"])
plt.tight_layout(); plt.savefig("nav_target_distribution.png", dpi=150)

# ---------------------------------------------- 5. Baselines
def score(name, flag):
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    return {"baseline": name,
            "accuracy": round((flag == (y == 1)).mean(), 4),
            "recall": round(tp / (tp + fn), 2),
            "precision": round(tp / (tp + fp), 2) if tp + fp else 0.0,
            "flag_rate_%": round(100 * flag.mean(), 1)}

never = pd.Series(False, index=panel.index)
raw_rule = panel["ret_obs"].abs() > THRESH
adj_rule = panel["adj_ret"].abs() > THRESH
cap = panel["adj_ret"].abs().quantile(0.95)          # threshold that flags 5% (review capacity)
cap_rule = panel["adj_ret"].abs() > cap
print("\nBaselines:")
print(pd.DataFrame([score("Majority class (never flag)", never),
                    score("50 bps day-over-day, raw", raw_rule),
                    score("50 bps benchmark-adjusted", adj_rule),
                    score(f"Benchmark-adjusted at 5% cap ({cap*1e4:.0f} bps)", cap_rule)]).to_string(index=False))
prev = y.mean()
print(f"\nPrevalence {prev:.2%}: best possible precision at a 5% flag rate = {prev/0.05:.2f}; "
      f"flag rate needed for 0.80 recall at 0.50 precision = {0.80*prev/0.50:.1%}")

print("\nBenchmark-adjusted rule, recall by error type and severity:")
print(panel[y == 1].assign(hit=adj_rule[y == 1])
      .pivot_table(index="atype", columns="sev", values="hit", aggfunc="mean").round(2))

fp = adj_rule & (y == 0)
print(f"\nFalse alarms that are the day-after 'echo' of an injected error: "
      f"{int((fp & (panel.echo == 1)).sum())} of {int(fp.sum())}")
print("Flag rate on clean rows by asset class (%):")
print((100 * adj_rule[y == 0].groupby(panel.loc[y == 0, "asset_class"]).mean()).round(1))
