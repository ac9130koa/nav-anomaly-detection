"""
nav_m2.py: Milestone 2 data pipeline for the NAV error detection study.

What changed from Milestone 1 (src/nav_data.py, kept unchanged for the weekly scripts):
  1. Universe: 126 funds (mutual funds and ETFs) in three asset classes, up from 26.
  2. Benchmarks use total return (price plus distributions). M1 used price return,
     which put a false move on every benchmark ex-dividend day.
  3. Daily-accrual bond mutual funds are detected from the data: their NAV does not
     drop on Yahoo's distribution date, so distributions are left out of their
     returns and they are never given a missed-dividend error.
  4. Share classes of one portfolio (e.g., VTSAX and VTI) form one group, so a
     portfolio is never split between training and test.
  5. Every NAV is rounded to the cent, as published. An injected error exists only
     if it changes the published NAV; sub-cent errors vanish and are counted.
  6. Two realistic error variants are added next to the four M1 "recipe" types:
     partial stale pricing (only part of the portfolio is not repriced) and an
     expense-accrual drift that builds up over several days.
  7. New point-in-time features: beta-adjusted residual z-score, a 10-day
     cumulative residual (drift) z-score, and an ex-date ratio.

Data: Yahoo Finance via yfinance, downloaded at run time and cached in data/ (not committed).
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "m2"
FIGURES = ROOT / "figures"

SEED = 42
START, END = "2021-05-01", "2026-09-01"
INJECT_RATE = 0.01                 # share of fund-days that carry an error
BUDGETS = [0.01, 0.02, 0.05, 0.10]  # share of fund-days an analyst reviews
MAIN_BUDGET = 0.05
REPROCESS = 0.005                  # 0.5% of NAV: above this, public NAV error policies require reprocessing shareholder trades; $0.01/share is the materiality floor
DEV_END = "2024-12-01"             # development period ends; December 2024 is an embargo month
TEST_START = "2025-01-01"          # future period starts
HOLDOUT_SHARE = 0.20               # share of portfolio groups per asset class held out

BENCH = {"us_equity": "SPY", "bond": "AGG", "intl": "EFA"}
FUNDS = {
    "us_equity": [
        "VFIAX", "FXAIX", "VTSAX", "SWPPX", "SWTSX", "FSKAX", "VIGAX", "VVIAX", "VIMAX", "VSMAX",
        "AGTHX", "PRBLX", "FCNTX", "DODGX", "PRGFX", "TRBCX", "AMCPX", "AIVSX", "ANCFX", "FMAGX",
        "VDIGX", "VPMCX", "VWNFX", "FDGRX", "AWSHX",
        "IVV", "VOO", "VTI", "ITOT", "SCHB", "SCHX", "QQQ", "IWM", "DIA", "VTV", "VUG", "RSP",
        "MDY", "IJH", "IJR", "VB", "VO", "SCHD", "VIG", "DGRO", "USMV", "MTUM", "QUAL",
        "XLK", "XLF", "XLV", "XLE",
    ],
    "bond": [
        "VBTLX", "PTTRX", "DODIX", "VWITX", "FXNAX", "MWTRX", "PIMIX", "VFSTX", "VBIRX", "VWESX",
        "FBNDX", "VFIDX", "VWEHX", "VAIPX", "VWIUX",
        "BND", "LQD", "TLT", "TIP", "IEF", "SHY", "IEI", "GOVT", "MUB", "VCIT", "VCSH", "BSV",
        "BIV", "BLV", "SCHZ", "HYG", "JNK", "EMB", "MBB", "VGSH", "VGIT", "VGLT", "IUSB", "USIG", "IGIB",
    ],
    "intl": [
        "VTIAX", "DODFX", "VTMGX", "VEIEX", "AEPGX", "VTRIX", "VWIGX", "FSPSX", "HAINX", "OAKIX",
        "DFIEX", "TRIGX", "FDIVX", "PRITX", "VFWAX", "VEMAX", "FPADX",
        "VEA", "IEFA", "VWO", "EEM", "VXUS", "IXUS", "SCHF", "EFV", "EFG", "ACWX", "VGK", "EWJ",
        "IEMG", "SPDW", "VEU", "SCZ", "VSS",
    ],
}
# Share classes of the same portfolio. Funds that only track the same index
# (e.g., IVV and VOO) are different portfolios and stay separate.
SHARE_CLASS_GROUP = {
    "VTSAX": "vg_total_stock", "VTI": "vg_total_stock",
    "VFIAX": "vg_500", "VOO": "vg_500",
    "VIGAX": "vg_growth", "VUG": "vg_growth",
    "VVIAX": "vg_value", "VTV": "vg_value",
    "VIMAX": "vg_mid", "VO": "vg_mid",
    "VSMAX": "vg_small", "VB": "vg_small",
    "VBTLX": "vg_total_bond", "BND": "vg_total_bond",
    "VBIRX": "vg_short_bond", "BSV": "vg_short_bond",
    "VWITX": "vg_it_tax_exempt", "VWIUX": "vg_it_tax_exempt",
    "VTIAX": "vg_total_intl", "VXUS": "vg_total_intl",
    "VTMGX": "vg_developed", "VEA": "vg_developed",
    "VEIEX": "vg_emerging", "VEMAX": "vg_emerging", "VWO": "vg_emerging",
    "VFWAX": "vg_all_world_ex_us", "VEU": "vg_all_world_ex_us",
}

RECIPE_TYPES = ["stale_price", "missed_split", "missed_dividend", "expense_error"]   # M1 types
REALISTIC_TYPES = ["stale_partial", "accrual_drift"]                                   # new in M2
ERROR_TYPES = RECIPE_TYPES + REALISTIC_TYPES

SPLIT_ERR = {1: -0.0025, 2: -0.0067, 3: +0.04}     # corporate action missed on one holding (M1)
EXP_ERR = {1: -0.0005, 2: -0.0015, 3: -0.0050}     # one-day expense over-accrual (M1)
DRIFT_PER_DAY = {1: 0.0001, 2: 0.0002, 3: 0.0005}  # daily over-accrual that builds up (new)
DRIFT_DAYS = [3, 5, 10]

M1_FEATURES = ["abs_z60", "zero_x_bench", "is_ex"]                 # Week 6 choice, M1 model
NEW_FEATURES = ["abs_zbeta", "drift10_z", "ex_ratio"]
CONTEXT_FEATURES = ["abs_vs_class_med", "stale_run", "bench_abs", "abs_prev_adj"]
FULL_FEATURES = M1_FEATURES + NEW_FEATURES + CONTEXT_FEATURES


# ----------------------------------------------------------------------------- data
def fund_class():
    return {t: c for c, ts in FUNDS.items() for t in ts}


def download(start=START, end=END, cache=DATA / "raw_m2.pkl", refresh=False):
    """Download closes and distributions once and cache them as a pickle in data/."""
    if cache.exists() and not refresh:
        return pickle.loads(cache.read_bytes())
    import yfinance as yf
    tickers = list(fund_class()) + list(BENCH.values())
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, actions=True,
                      progress=False, threads=True)
    cache.parent.mkdir(exist_ok=True)
    cache.write_bytes(pickle.dumps(raw))
    return raw


def clean_prices(raw):
    """Cent-rounded closes, total distributions, and the daily-accrual check per fund."""
    close = raw["Close"].round(2)
    dist = raw["Dividends"].fillna(0)
    if "Capital Gains" in raw.columns.get_level_values(0):
        dist = dist + raw["Capital Gains"].reindex(columns=close.columns).fillna(0)
    rows = []
    for t, cls in fund_class().items():
        b = BENCH[cls]
        price_ret = close[t].pct_change()
        bench_tr = (close[b] + dist[b]) / close[b].shift(1) - 1
        ex = dist[t] > 0
        yld = dist[t] / close[t].shift(1)
        # On a distribution date the price should fall by the distribution relative to the
        # benchmark: drop_ratio near 1. Daily-accrual bond funds show a ratio near 0.
        drop_ratio = float((-(price_ret - bench_tr)[ex] / yld[ex]).median()) if ex.any() else np.nan
        is_mf = len(t) == 5 and t.endswith("X")
        rows.append(dict(ticker=t, asset_class=cls, group=SHARE_CLASS_GROUP.get(t, t),
                         vehicle="mutual fund" if is_mf else "ETF",
                         n_dist_days=int(ex.sum()), drop_ratio=round(drop_ratio, 2),
                         daily_accrual=bool(cls == "bond" and is_mf and drop_ratio < 0.5),
                         coverage=float(close[t].reindex(close[b].dropna().index).notna().mean()),
                         last_close=float(close[t].dropna().iloc[-1])))
    universe = pd.DataFrame(rows)
    return close, dist, universe


def assign_holdout(universe, seed=SEED, share=HOLDOUT_SHARE):
    """Hold out 20% of portfolio groups in each asset class, fixed before any modeling."""
    rng = np.random.default_rng(seed)
    hold = set()
    for cls, g in universe.groupby("asset_class"):
        groups = np.array(sorted(g["group"].unique()))
        k = int(round(share * len(groups)))
        hold |= set(rng.permutation(groups)[:k])
    universe = universe.copy()
    universe["holdout_fund"] = universe["group"].isin(hold)
    return universe


def build_panel(close, dist, universe):
    """Long fund-day panel of true NAVs, benchmark total returns, and metadata."""
    frames = []
    for r in universe.itertuples():
        b = BENCH[r.asset_class]
        bench_tr = (close[b] + dist[b]) / close[b].shift(1) - 1
        dn = dist[r.ticker]
        if r.daily_accrual:
            # Regular monthly income is accrued daily and never drops the NAV; only
            # unusually large (year-end, capital gain) distributions do: > 2x the median.
            dn = dn.where(dn > 2 * dn[dn > 0].median(), 0.0)
        df = pd.DataFrame({"close": close[r.ticker], "dist": dn}).dropna(subset=["close"])
        df["bench_ret"] = bench_tr.reindex(df.index)
        df["prev_close"] = df["close"].shift(1)
        df = df.iloc[1:]
        df["ticker"], df["asset_class"], df["group"] = r.ticker, r.asset_class, r.group
        df["holdout_fund"] = r.holdout_fund
        df["pos"] = np.arange(len(df))            # position within the fund's history
        frames.append(df)
    return pd.concat(frames).reset_index(names="date")


# ----------------------------------------------------------------------------- injection
def inject_errors(panel, seed=SEED, rate=INJECT_RATE, warmup=75):
    """Inject logged NAV errors. Returns (panel with obs NAV and labels, event table).

    Events are added one type at a time in rotation, so every type gets the same
    number of events, until erroneous fund-days reach `rate` of all fund-days.
    The NAV that would be released (obs) is rounded to the cent; an event that
    does not change any published NAV is discarded and counted as 'vanished'.
    """
    rng = np.random.default_rng(seed)
    p = panel.copy()
    n = len(p)
    close = p["close"].to_numpy()
    dist = p["dist"].to_numpy()
    obs = close.copy()
    lab = np.zeros(n, int)
    echo = np.zeros(n, int)
    used = np.zeros(n, bool)
    atype = np.array(["none"] * n, dtype=object)
    sev = np.zeros(n, int)
    eid = np.full(n, -1)
    idx_by_fund = {t: ix for t, ix in p.groupby("ticker").indices.items()}
    funds = sorted(idx_by_fund)
    ex_funds = [t for t in funds if (dist[idx_by_fund[t]] > 0).any()]
    events, vanished = [], {k: 0 for k in ERROR_TYPES}
    target_rows = int(rate * n)

    def try_one(k):
        fund = rng.choice(ex_funds if k == "missed_dividend" else funds)
        full = idx_by_fund[fund]
        cand = full[warmup:-12]
        if k == "missed_dividend":
            cand = cand[dist[cand] > 0]
            if len(cand) == 0:
                return None
        i = int(rng.choice(cand))
        s = 0 if k == "missed_dividend" else int(rng.integers(1, 4))
        extra = {}
        if k in ("stale_price", "stale_partial"):
            span = s
        elif k == "accrual_drift":
            span = int(rng.choice(DRIFT_DAYS))
        else:
            span = 1
        lo, hi = i - 1, i + span + 1             # one clean day before, echo day, one buffer day
        if used[lo:hi + 1].any():
            return None
        if k != "missed_dividend" and (dist[i:i + span + 1] > 0).any():
            return None                          # keep distribution days out of other error types
        new = close[i:i + span].copy()
        if k == "stale_price":
            new[:] = close[i - 1]
        elif k == "stale_partial":
            f = float(rng.uniform(0.2, 0.8))
            extra["stale_share"] = round(f, 3)
            new = close[i - 1] + (1 - f) * (close[i:i + span] - close[i - 1])
        elif k == "missed_split":
            new = close[i:i + span] * (1 + SPLIT_ERR[s])
        elif k == "missed_dividend":
            yld = dist[i] / close[i - 1]
            s = 1 if yld < 0.002 else (2 if yld < 0.01 else 3)
            new = close[i:i + span] + dist[i:i + span]
        elif k == "expense_error":
            new = close[i:i + span] * (1 + EXP_ERR[s])
        else:                                    # accrual_drift
            new = close[i:i + span] * (1 - DRIFT_PER_DAY[s] * np.arange(1, span + 1))
        new = np.round(new, 2)
        diff = new != close[i:i + span]
        if not diff.any():
            vanished[k] += 1
            return "vanished"
        e = len(events)
        rows = np.arange(i, i + span)
        obs[rows] = new
        lab[rows] = diff.astype(int)
        atype[rows[diff]] = k
        sev[rows[diff]] = s
        eid[rows[diff]] = e
        echo[i + span] = 1                       # first correct NAV after the error snaps back
        used[lo:hi + 1] = True
        err = np.abs(new - close[i:i + span])
        events.append(dict(event_id=e, atype=k, sev=s, ticker=fund, start=p.at[i, "date"],
                           span_days=span, error_days=int(diff.sum()),
                           size_bps=float((err / close[i:i + span]).max() * 1e4),
                           size_usd=float(err.max()), nav=float(close[i]), **extra))
        return "ok"

    for _ in range(200_000):
        if lab.sum() >= target_rows:
            break
        for k in ERROR_TYPES:
            for _attempt in range(200):
                if try_one(k) == "ok":
                    break

    p["obs"], p["is_anomaly"], p["echo"] = obs, lab, echo
    p["atype"], p["sev"], p["event_id"] = atype, sev, eid
    ev = pd.DataFrame(events)
    ev.attrs["vanished"] = vanished
    return p, ev


# ----------------------------------------------------------------------------- features
def add_features(p):
    """Point-in-time features. Every rolling statistic uses only earlier days."""
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = p.groupby("ticker", sort=False)
    p["prev_obs"] = g["obs"].shift(1).fillna(p["prev_close"])
    p["ret_obs"] = (p["obs"] + p["dist"]) / p["prev_obs"] - 1
    p["adj_ret"] = p["ret_obs"] - p["bench_ret"]
    g = p.groupby("ticker", sort=False)

    def past(col, fn, w=60, lag=1):
        return g[col].transform(lambda s: getattr(s.shift(lag).rolling(w), fn)())

    p["z60"] = (p["adj_ret"] - past("adj_ret", "mean")) / past("adj_ret", "std")
    p["abs_z60"] = p["z60"].abs()

    # rolling beta to the benchmark from the previous 60 days, residual z-score
    p["_xy"] = p["ret_obs"] * p["bench_ret"]
    mx, my = past("bench_ret", "mean"), past("ret_obs", "mean")
    vx, vy = past("bench_ret", "var"), past("ret_obs", "var")
    cxy = past("_xy", "mean") - mx * my
    beta = cxy / vx
    resid_sd = np.sqrt((vy - beta * cxy).clip(lower=1e-12))
    p["beta60"] = beta
    p["resid"] = p["ret_obs"] - (my - beta * mx) - beta * p["bench_ret"]
    p["zbeta"] = p["resid"] / resid_sd
    p["abs_zbeta"] = p["zbeta"].abs()
    # cumulative residual over the last 10 days, scaled by the residual SD 10 days earlier
    cum10 = p.groupby("ticker", sort=False)["resid"].transform(lambda s: s.rolling(10).sum())
    sd_lag = resid_sd.groupby(p["ticker"]).shift(9)
    p["drift10_z"] = (cum10 / (sd_lag * np.sqrt(10))).abs()

    p["zero_move"] = (p["obs"] == p["prev_obs"]).astype(int) * (p["dist"] == 0).astype(int)
    p["stale_run"] = p.groupby(["ticker", (p["zero_move"] == 0).cumsum()])["zero_move"].cumsum()
    p["bench_abs"] = p["bench_ret"].abs()
    p["zero_x_bench"] = p["zero_move"] * p["bench_abs"]   # flat NAV on a day the market moved
    p["is_ex"] = (p["dist"] > 0).astype(int)
    yld = (p["dist"] / p["prev_obs"]).where(p["is_ex"] == 1)
    p["ex_ratio"] = (p["adj_ret"] / yld).clip(-5, 5).fillna(0) * p["is_ex"]
    p["prev_adj"] = p.groupby("ticker", sort=False)["adj_ret"].shift(1)
    p["vs_class_med"] = p["ret_obs"] - p.groupby(["date", "asset_class"])["ret_obs"].transform("median")
    for c in ["ret_obs", "adj_ret", "vs_class_med", "prev_adj"]:
        p["abs_" + c] = p[c].abs()
    p["size_bps"] = (np.abs(p["obs"] - p["close"]) / p["close"] * 1e4)
    p = p.drop(columns=["_xy"])
    keep = p.dropna(subset=["z60", "zbeta", "drift10_z", "prev_adj"]).query("echo == 0")
    return keep.reset_index(drop=True)


def period(dates):
    d = pd.to_datetime(dates)
    return np.where(d < DEV_END, "past", np.where(d >= TEST_START, "future", "embargo"))


def build_dataset(refresh=False, seed=SEED):
    """Download (or load the cache), inject, featurize.

    The main injection (seed 42) is cached in data/panel_m2.pkl. Other seeds give
    independent error draws on the same clean prices (used to score the holdout).
    """
    cache = DATA / "panel_m2.pkl"
    if seed == SEED and cache.exists() and not refresh:
        return pickle.loads(cache.read_bytes())
    raw = download(refresh=refresh)
    close, dist, universe = clean_prices(raw)
    universe = assign_holdout(universe)          # always seed 42: the holdout never changes
    panel = build_panel(close, dist, universe)
    panel, events = inject_errors(panel, seed=seed)
    d = add_features(panel)
    d["period"] = period(d["date"])
    out = dict(data=d, events=events, universe=universe, vanished=events.attrs["vanished"],
               raw_rows=len(panel))
    if seed == SEED:
        cache.write_bytes(pickle.dumps(out))
    return out


# ----------------------------------------------------------------------------- metrics
def review_top(scores, budget=MAIN_BUDGET, seed=0):
    """Flag exactly `budget` of rows by score; ties broken at random."""
    s = np.asarray(scores, float)
    n = int(np.ceil(budget * len(s)))
    order = np.lexsort((np.random.default_rng(seed).random(len(s)), -s))
    flag = np.zeros(len(s), bool)
    flag[order[:n]] = True
    return flag


def event_hits(frame, flag):
    """Per event: was at least one of its erroneous days flagged?"""
    m = frame["is_anomaly"].to_numpy() == 1
    hit = pd.Series(np.asarray(flag)[m]).groupby(frame["event_id"].to_numpy()[m]).any()
    return hit


def detection_metrics(frame, flag):
    """Event recall, fund-day recall, fund-day precision for one set of flags."""
    y = frame["is_anomaly"].to_numpy() == 1
    flag = np.asarray(flag)
    hit = event_hits(frame, flag)
    return dict(event_recall=float(hit.mean()) if len(hit) else np.nan,
                day_recall=float(flag[y].mean()) if y.any() else np.nan,
                day_precision=float(y[flag].mean()) if flag.any() else np.nan,
                n_events=int(len(hit)), n_error_days=int(y.sum()), n_rows=len(y))


def pct_rank(x):
    """Percentile rank in (0, 1]; used to blend scores on a common scale."""
    return pd.Series(np.asarray(x, float)).rank(pct=True).to_numpy()


def size_bucket(bps):
    return pd.cut(bps, [0, 10, 50, np.inf], labels=["<10 bps", "10-50 bps", ">=50 bps"], right=False)
