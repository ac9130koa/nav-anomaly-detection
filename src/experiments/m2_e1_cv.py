"""
E1 (RQ1): repeated fund-level cross-validation on the development set.

Development set = development funds x development period (before December 2024).
5 folds x 5 repeats over portfolio groups, stratified by asset class.
Every method is scored at review budgets of 1%, 2%, 5%, and 10% of fund-days.
Outputs (results/m2/):
  e1_fold_metrics.csv   one row per method x repeat x fold x budget
  e1_event_hits.csv     one row per method x repeat x fold x event at the 5% budget
  e1_hybrid_weights.csv the hybrid weight chosen inside each training fold
  e1_natural_rules.csv  the 50 bps rules at their own threshold (no budget)
  e1_lr_C.csv           sensitivity of logistic regression to C (repeat 0)
  e1_nb_tests.csv       Nadeau-Bengio corrected tests on event recall at 5%
"""
from m2_common import *  # noqa: F401,F403
from sklearn.metrics import average_precision_score

REPS, FOLDS = 5, 5


def main():
    log = log_to(m.RESULTS / "e1_log.txt")
    d = m.build_dataset()["data"]
    dev = development_set(d)
    log(f"development set: {len(dev)} fund-days, {dev['ticker'].nunique()} funds, "
        f"{dev.loc[dev.is_anomaly == 1, 'event_id'].nunique()} events")
    rows, hits, weights, natural, csens = [], [], [], [], []
    for rep in range(REPS):
        fold = mm.group_folds(dev, FOLDS, rep)
        for f in range(FOLDS):
            train, test = dev[fold != f], dev[fold == f].reset_index(drop=True)
            log(f"rep {rep} fold {f}: train {len(train)} rows, test {len(test)} rows")
            scores, w = score_all(train, test, seed=rep, log=log)
            weights.append(dict(rep=rep, fold=f, hybrid_w=w))
            y = test["is_anomaly"].to_numpy()
            for name, s in scores.items():
                ap = average_precision_score(y, s)
                for b in m.BUDGETS:
                    flag = m.review_top(s, b, seed=rep)
                    rows.append(dict(method=name, rep=rep, fold=f, budget=b, pr_auc=ap,
                                     **m.detection_metrics(test, flag)))
                    if b == m.MAIN_BUDGET:
                        h = m.event_hits(test, flag)
                        hits.append(pd.DataFrame(dict(method=name, rep=rep, fold=f,
                                                      event_id=h.index, hit=h.values)))
            for name, flag in mm.rule_natural_flags(test).items():
                natural.append(dict(method=name, rep=rep, fold=f, review_rate=flag.mean(),
                                    **m.detection_metrics(test, flag)))
            if rep == 0:
                for C in [0.01, 1.0, 100.0]:
                    s = mm.fit_score(mm.lr_full(C), m.FULL_FEATURES, train, test)
                    csens.append(dict(C=C, fold=f, **m.detection_metrics(test, m.review_top(s, 0.05, 0))))
        pd.DataFrame(rows).to_csv(m.RESULTS / "e1_fold_metrics.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(m.RESULTS / "e1_fold_metrics.csv", index=False)
    pd.concat(hits).to_csv(m.RESULTS / "e1_event_hits.csv", index=False)
    pd.DataFrame(weights).to_csv(m.RESULTS / "e1_hybrid_weights.csv", index=False)
    pd.DataFrame(natural).to_csv(m.RESULTS / "e1_natural_rules.csv", index=False)
    pd.DataFrame(csens).to_csv(m.RESULTS / "e1_lr_C.csv", index=False)

    # Nadeau-Bengio corrected t-tests on event recall at the 5% budget (test/train = 1/4)
    r5 = res[res.budget == m.MAIN_BUDGET].pivot_table(index=["rep", "fold"], columns="method",
                                                       values="event_recall")
    tests = []
    for ref in ["Rule: benchmark-adjusted move", "Logistic, M1 features"]:
        for meth in r5.columns:
            if meth == ref:
                continue
            tests.append(dict(method=meth, reference=ref,
                              **mm.nadeau_bengio(r5[meth] - r5[ref], 1 / (FOLDS - 1))))
    pd.DataFrame(tests).to_csv(m.RESULTS / "e1_nb_tests.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
