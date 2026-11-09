"""
nav_data.py: build the fund-day panel, the injected NAV-error target, and the
point-in-time features used in ANLY 530 Weeks 4-8.

Data: daily prices and distributions from Yahoo Finance via yfinance, downloaded
at run time. Nothing is stored in the repository.

Target: is_anomaly = 1 if the NAV released that day differs from the true NAV.
Real NAV errors are not public, so errors are injected into clean data:
four types (stale price, missed split, missed dividend, expense error) at three
severities, on about 1% of fund-days. Every injected event is logged
(event_id, atype, sev), and the first correct day after each error is marked
as an "echo" day and excluded from training and scoring.
"""
import numpy as np
import pandas as pd

SEED = 42
START, END = "2021-05-01", "2026-09-01"
INJECT_RATE = 0.01      # share of eligible fund-days
THRESH = 0.005          # 50 bps, the industry materiality convention

# Preliminary universe: 26 funds, 3 asset classes, mutual funds and ETFs
BENCH = {"us_equity": "SPY", "bond": "AGG", "intl": "EFA"}
FUNDS = {
    "us_equity": ["VFIAX", "FXAIX", "VTSAX", "AGTHX", "PRBLX", "IVV", "VTI", "QQQ", "IWM", "DIA"],
    "bond":      ["VBTLX", "PTTRX", "DODIX", "VWITX", "BND", "LQD", "TLT", "TIP"],
    "intl":      ["VTIAX", "DODFX", "VTMGX", "VEIEX", "VEA", "IEFA", "VWO", "EEM"],
}
ERROR_TYPES = ["stale_price", "missed_split", "missed_dividend", "expense_error"]
PRIMARY_FEATURES = ["abs_z60", "zero_x_bench", "is_ex"]   # Week 6 decision


def download(start=START, end=END):
    """Download closes and distributions for the funds and benchmarks."""
    import yfinance as yf
    fund_class = {t: c for c, ts in FUNDS.items() for t in ts}
    tickers = list(fund_class) + list(BENCH.values())
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False,
                      actions=True, progress=False)
    close = raw["Close"]
    dist = raw["Dividends"].fillna(0)
    if "Capital Gains" in raw.columns.get_level_values(0):
        dist = dist + raw["Capital Gains"].reindex(columns=close.columns).fillna(0)
    return close, dist, fund_class


def inject_errors(close, dist, fund_class, seed=SEED, rate=INJECT_RATE):
    """Long fund-day panel with injected NAV errors and an event log."""
    bench_ret = {c: close[b].pct_change() for c, b in BENCH.items()}
    rng = np.random.default_rng(seed)
    rows = []
    for t, cls in fund_class.items():
        df = pd.DataFrame({"close": close[t], "dist": dist[t]}).dropna(subset=["close"])
        df["ticker"], df["asset_class"] = t, cls
        df["bench_ret"] = bench_ret[cls].reindex(df.index)
        rows.append(df.iloc[1:].assign(prev_close=df["close"].shift(1).iloc[1:]))
    panel = pd.concat(rows).reset_index(names="date")

    panel["obs"] = panel["close"]                      # the NAV that would be released
    panel["is_anomaly"], panel["echo"] = 0, 0
    panel["atype"], panel["sev"], panel["event_id"] = "none", 0, -1

    n_events = int(rate * len(panel) / 1.25)           # stale events span more than one row
    split_err = {1: -0.0025, 2: -0.0067, 3: +0.04}     # split ignored on a small position
    exp_err = {1: -0.0005, 2: -0.0015, 3: -0.0050}     # expense accrual overstated

    used = np.zeros(len(panel), bool)
    idx_by_fund = panel.groupby("ticker").indices
    done = {k: 0 for k in ERROR_TYPES}
    for k in ERROR_TYPES:
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
                yld = panel.at[i, "dist"] / panel.at[i, "prev_close"]
                sev = 1 if yld < 0.002 else (2 if yld < 0.01 else 3)
            else:
                sev = int(rng.integers(1, 4))
            span = sev if k == "stale_price" else 1
            lo, hi = i - 1, i + span + 1                # buffer plus one clean echo day
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

    # Observed total return the control sees (the distribution calendar is known)
    prev_obs = panel.groupby("ticker")["obs"].shift(1).fillna(panel["prev_close"])
    panel["prev_obs"] = prev_obs
    panel["ret_obs"] = (panel["obs"] + panel["dist"]) / prev_obs - 1
    panel["adj_ret"] = panel["ret_obs"] - panel["bench_ret"]
    return panel.dropna(subset=["adj_ret"])


def add_features(panel):
    """Point-in-time features: each uses only data available at NAV strike."""
    g = panel.groupby("ticker")
    past = lambda s, f: g[s].transform(lambda x: getattr(x.shift(1).rolling(60), f)())
    panel["z60"] = (panel["adj_ret"] - past("adj_ret", "mean")) / past("adj_ret", "std")
    panel["abs_z60"] = panel["z60"].abs()
    panel["zero_move"] = (panel["ret_obs"] - panel["dist"] / panel["prev_obs"]).abs().lt(1e-9).astype(int)
    panel["stale_run"] = panel.groupby(["ticker", (panel["zero_move"] == 0).cumsum()])["zero_move"].cumsum()
    panel["bench_abs"] = panel["bench_ret"].abs()
    panel["is_ex"] = (panel["dist"] > 0).astype(int)
    panel["prev_adj"] = g["adj_ret"].shift(1)
    panel["vs_class_med"] = panel["ret_obs"] - panel.groupby(["date", "asset_class"])["ret_obs"].transform("median")

    # drop the 60-day warm-up and the correction ("echo") days
    d = panel.dropna(subset=["z60"]).query("echo == 0").reset_index(drop=True)
    for c in ["ret_obs", "adj_ret", "vs_class_med", "prev_adj"]:
        d["abs_" + c] = d[c].abs()
    d["zero_x_bench"] = d["zero_move"] * d["bench_abs"]    # flat NAV on a day the market moved
    return d


def build_dataset(start=START, end=END, seed=SEED):
    """Download, inject, and featurize. Returns the modeling frame."""
    close, dist, fund_class = download(start, end)
    return add_features(inject_errors(close, dist, fund_class, seed=seed))


def review_top(scores, budget=0.05, seed=0):
    """Flag exactly `budget` of rows by score; ties broken at random."""
    n = int(np.ceil(budget * len(scores)))
    order = np.lexsort((np.random.default_rng(seed).random(len(scores)), -np.asarray(scores)))
    flag = np.zeros(len(scores), bool)
    flag[order[:n]] = True
    return flag


def event_recall(frame, flag):
    """Share of injected error events with at least one flagged day."""
    ev = frame.assign(flag=flag).query("is_anomaly == 1")
    return ev.groupby("event_id")["flag"].any().mean()
