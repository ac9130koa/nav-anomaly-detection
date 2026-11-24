"""
E4 (RQ2): time-forward holdout on unseen funds, scored once.

Construction (fixed before modeling, see nav_m2.assign_holdout and nav_m2.period):
  * Holdout funds: 20% of portfolio groups in each asset class (seed 42); share classes
    of one portfolio are held out together. 26 funds in 22 groups.
  * Periods: development = before 2024-12-01; December 2024 is an embargo month (dropped,
    so no error event or rolling window straddles the cut); future = 2025-01-01 onward.
  * Training: development funds x development period only (the E1 development set).
  * Test cells: development funds x future period (time shift only), holdout funds x
    development period (fund shift only), holdout funds x future period (both; the main test).
  * Rolling features in a test cell use that fund's own earlier NAVs, which is what an
    administrator would have at NAV strike; no test labels are used anywhere.
The models are fit once on the seed-42 errors. Because one draw puts only about 67 error
events in the main cell, each cell is scored on 10 independent error draws (seeds 42-51)
on the same clean prices. Uncertainty comes from a fund-cluster bootstrap: portfolio
groups are resampled with replacement (2,000 times) and recall is pooled over draws.
Outputs: results/m2/e4_metrics.csv, e4_event_hits.csv, e4_bootstrap.csv, e4_hybrid.csv
"""
from m2_common import *  # noqa: F401,F403
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

DRAWS = list(range(42, 52))
CELLS = {"Seen funds, future period": (False, "future"),
         "Unseen funds, development period": (True, "past"),
         "Unseen funds, future period": (True, "future")}
REFS = ["Rule: benchmark-adjusted move", "Logistic, M1 features"]
B = 2000


def main():
    log = log_to(m.RESULTS / "e4_log.txt")
    d = m.build_dataset()["data"]
    dev = development_set(d)
    log(f"training on {len(dev)} development fund-days")
    fitted = {}
    for name, (est, feats) in mm.make_supervised(seed=0).items():
        est.fit(dev[feats].to_numpy(), dev["is_anomaly"].to_numpy())
        fitted[name] = (est, feats)
    iso = IsolationForest(n_estimators=200, random_state=0, n_jobs=-1).fit(dev[m.FULL_FEATURES].to_numpy())
    w, obj = mm.choose_hybrid_weight(dev, seed=0)
    pd.DataFrame([dict(weight=k, objective=v, chosen=(k == w)) for k, v in obj.items()]).to_csv(
        m.RESULTS / "e4_hybrid.csv", index=False)
    log(f"hybrid weight on logistic: {w}")

    rows, hits = [], []
    for s in DRAWS:
        ds = d if s == m.SEED else m.build_dataset(seed=s)["data"]
        for cell, (hold, per) in CELLS.items():
            test = ds[(ds.holdout_fund == hold) & (ds.period == per)].reset_index(drop=True)
            scores = dict(mm.rule_scores(test))
            for name, (est, feats) in fitted.items():
                scores[name] = est.predict_proba(test[feats].to_numpy())[:, 1]
            s_if = -iso.score_samples(test[m.FULL_FEATURES].to_numpy())
            scores["Isolation forest (no labels)"] = s_if
            scores["Hybrid (logistic + isolation forest)"] = (
                w * m.pct_rank(scores["Logistic, full features"]) + (1 - w) * m.pct_rank(s_if))
            et = event_table(test)
            y = test["is_anomaly"].to_numpy()
            for name, sc in scores.items():
                ap = average_precision_score(y, sc)
                for b in m.BUDGETS:
                    flag = m.review_top(sc, b, seed=s)
                    rows.append(dict(cell=cell, draw=s, method=name, budget=b, pr_auc=ap,
                                     **m.detection_metrics(test, flag)))
                    if b == m.MAIN_BUDGET:
                        h = m.event_hits(test, flag)
                        hits.append(pd.DataFrame(dict(
                            cell=cell, draw=s, method=name, event_id=h.index, hit=h.values,
                            group=test.groupby("event_id")["group"].first().reindex(h.index).values,
                            atype=et["atype"].reindex(h.index).values,
                            size_bps=et["size_bps"].reindex(h.index).values)))
        log(f"draw {s} scored")
    res = pd.DataFrame(rows)
    res.to_csv(m.RESULTS / "e4_metrics.csv", index=False)
    H = pd.concat(hits)
    H.to_csv(m.RESULTS / "e4_event_hits.csv", index=False)

    # fund-cluster bootstrap, recall pooled over draws
    rng = np.random.default_rng(m.SEED)
    boot = []
    for cell, hc in H.groupby("cell"):
        groups = np.array(sorted(hc["group"].unique()))
        G = len(groups)
        counts = rng.multinomial(G, np.full(G, 1 / G), size=B)          # B x G
        agg = hc.groupby(["method", "group"])["hit"].agg(["sum", "size"]).reset_index()
        rec = {}
        for meth, a in agg.groupby("method"):
            a = a.set_index("group").reindex(groups).fillna(0)
            num, den = counts @ a["sum"].to_numpy(), counts @ a["size"].to_numpy()
            rec[meth] = (a["sum"].sum() / a["size"].sum(), num / den, int(a["size"].sum() / len(DRAWS)))
        for meth, (pt, bs, n_ev) in rec.items():
            r = dict(cell=cell, method=meth, recall=pt, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                     events_per_draw=n_ev, groups=G)
            for ref in REFS:
                diff = bs - rec[ref][1]
                tag = "rule" if ref.startswith("Rule") else "lr_m1"
                r[f"diff_vs_{tag}"] = pt - rec[ref][0]
                r[f"diff_vs_{tag}_lo"] = np.percentile(diff, 2.5)
                r[f"diff_vs_{tag}_hi"] = np.percentile(diff, 97.5)
            boot.append(r)
    pd.DataFrame(boot).to_csv(m.RESULTS / "e4_bootstrap.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
