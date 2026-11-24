"""
E0: data preparation checks and validation of the synthetic error generator.

Answers two questions before any model is trained:
  1. What did data preparation change, and how many fund-days did each fix touch?
  2. Do the injected errors look like plausible operational failures, or do some
     carry a signature that a single feature can pick out (an easy artifact)?
Outputs (results/m2/):
  e0_universe.csv        one row per fund: class, vehicle, portfolio group, holdout, daily accrual
  e0_cells.csv           rows, error days, and events in each fund x period cell
  e0_error_types.csv     per error type: events, days, vanished, size, share >= 50 bps
  e0_single_feature.csv  event recall at 5% when one label-free feature ranks the days
  e0_natural.csv         how often the error signatures occur on clean fund-days
  e0_data_fixes.csv      fund-days affected by each data-preparation fix
  e0_suspect_prints.csv  clean fund-days that spike and reverse (likely real data errors)
"""
from m2_common import *  # noqa: F401,F403

SINGLE = ["abs_ret_obs", "abs_adj_ret", "abs_z60", "abs_zbeta", "drift10_z",
          "zero_x_bench", "ex_ratio", "abs_vs_class_med"]


def main():
    out = m.build_dataset()
    d, ev, u = out["data"], out["events"], out["universe"]
    R = m.RESULTS
    u.to_csv(R / "e0_universe.csv", index=False)

    # cells: development/holdout funds x past/future periods
    cells = (d.assign(funds=np.where(d.holdout_fund, "holdout funds", "development funds"))
              .groupby(["funds", "period"])
              .agg(fund_days=("is_anomaly", "size"), error_days=("is_anomaly", "sum"),
                   events=("event_id", lambda s: s[s >= 0].nunique()), funds_n=("ticker", "nunique"))
              .reset_index())
    cells.to_csv(R / "e0_cells.csv", index=False)

    # error types: size distribution (bps of NAV) and how many vanished after cent rounding
    ev["bucket"] = m.size_bucket(ev["size_bps"])
    t = ev.groupby("atype").agg(events=("event_id", "size"), error_days=("error_days", "sum"),
                                median_bps=("size_bps", "median"),
                                p25_bps=("size_bps", lambda s: s.quantile(0.25)),
                                p75_bps=("size_bps", lambda s: s.quantile(0.75)),
                                share_ge_50bps=("size_bps", lambda s: (s >= 50).mean()),
                                share_lt_10bps=("size_bps", lambda s: (s < 10).mean()),
                                min_usd=("size_usd", "min"))
    t["vanished_sub_cent"] = pd.Series(out["vanished"])
    t = t.reindex(m.ERROR_TYPES)
    t.to_csv(R / "e0_error_types.csv")
    ev.to_csv(R / "e0_events.csv", index=False)

    # single-feature detectability on the development set
    dev = development_set(d)
    et = event_table(dev)
    rows = []
    for f in SINGLE:
        hit = m.event_hits(dev, m.review_top(dev[f].to_numpy(), m.MAIN_BUDGET, 0))
        r = hit.groupby(et["atype"].reindex(hit.index)).mean()
        rows.append(dict(feature=f, all_types=hit.mean(), **r.to_dict()))
    pd.DataFrame(rows).to_csv(R / "e0_single_feature.csv", index=False)

    # natural frequency of the signatures on clean development fund-days
    clean = dev[dev.is_anomaly == 0]
    flat = (clean.zero_move == 1) & (clean.bench_abs >= mm.FLAT_MOVE)
    nat = (clean.assign(flat_on_moving_market=flat, adj_move_gt_50bps=clean.abs_adj_ret > 0.005,
                        raw_move_gt_50bps=clean.abs_ret_obs > 0.005,
                        vehicle=clean.ticker.map(u.set_index("ticker")["vehicle"]))
           .groupby(["asset_class", "vehicle"])[["flat_on_moving_market", "adj_move_gt_50bps",
                                                  "raw_move_gt_50bps", "is_ex"]].mean())
    nat.loc[("all", "all"), :] = [flat.mean(), (clean.abs_adj_ret > 0.005).mean(),
                                  (clean.abs_ret_obs > 0.005).mean(), clean.is_ex.mean()]
    nat.to_csv(R / "e0_natural.csv")

    # data-preparation fixes: fund-days each one touched
    raw = m.download()
    close, dist, _ = m.clean_prices(raw)
    fixes = []
    for cls, b in m.BENCH.items():
        yld = (dist[b] / close[b].shift(1)).loc["2021-05-01":]
        n_funds = int((u.asset_class == cls).sum())
        fixes.append(dict(fix=f"benchmark total return: {b} ex-dividend days x {n_funds} {cls} funds",
                          fund_days_affected=int((yld > 0).sum()) * n_funds,
                          median_shift_bps=float(1e4 * yld[yld > 0].median())))
    da = u[u.daily_accrual].ticker.tolist()
    shifts = []
    for tk in da:
        dt = dist[tk]
        reg = (dt > 0) & (dt <= 2 * dt[dt > 0].median())
        shifts.append((dt / close[tk].shift(1))[reg])
    shifts = pd.concat(shifts)
    fixes.append(dict(fix=f"daily-accrual bond mutual funds ({len(da)}): monthly income left out of NAV return",
                      fund_days_affected=len(shifts), median_shift_bps=float(1e4 * shifts.median())))
    fixes.append(dict(fix=f"share classes grouped: {u.ticker.nunique()} funds -> {u.group.nunique()} portfolio groups",
                      fund_days_affected=np.nan, median_shift_bps=np.nan))
    pd.DataFrame(fixes).to_csv(R / "e0_data_fixes.csv", index=False)

    # clean fund-days that spike and reverse: likely real errors in the public data
    s = d.sort_values(["ticker", "date"]).copy()
    s["next_resid"] = s.groupby("ticker")["resid"].shift(-1)
    sus = s[(s.is_anomaly == 0) & (s.abs_zbeta > 8) & (np.sign(s.next_resid) == -np.sign(s.resid))
            & (s.next_resid.abs() > 0.8 * s.resid.abs())]
    sus[["date", "ticker", "asset_class", "prev_obs", "obs", "resid", "next_resid", "abs_zbeta",
         "holdout_fund", "period"]].to_csv(R / "e0_suspect_prints.csv", index=False)

    print(cells.to_string())
    print(t.round(3).to_string())
    print(pd.DataFrame(rows).round(2).to_string())
    print(nat.round(4).to_string())
    print(pd.DataFrame(fixes).to_string())
    print(sus[["date", "ticker", "prev_obs", "obs", "abs_zbeta"]].to_string())


if __name__ == "__main__":
    main()
