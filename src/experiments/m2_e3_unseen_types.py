"""
E3 (RQ3 and Dr. Itauma's question): why does recall collapse on an error type
that is missing from training?

For each held-out error type (and for the two realistic variants held out together),
the erroneous days of that type are removed from training, every method is refit,
and recall is measured on that type's events in the test funds. Repeats 0-1 of the
E1 fund-level folds are reused (10 folds), so the "seen" reference is E1 itself.

Three explanations are separated by design:
  model limitation   -> some other model family recovers recall when the type is unseen
  feature limitation -> recall is low even when the type IS in training, and no single
                        label-free feature ranks its days highly
  coverage (label)   -> the features carry the signal (seen recall high, a label-free
                        feature ranks the days highly) but every supervised model trained
                        without the type misses it
Extra diagnostics: logistic coefficients with and without the type in training, and a
feature ablation (logistic without the two flat-NAV features, all types seen).
Outputs: results/m2/e3_event_hits.csv, e3_coefs.csv, e3_hybrid_weights.csv
"""
from m2_common import *  # noqa: F401,F403

REPS, FOLDS = 2, 5
CONFIGS = {k: [k] for k in m.ERROR_TYPES}
CONFIGS["realistic pair (stale_partial + accrual_drift)"] = list(m.REALISTIC_TYPES)
SINGLE = ["abs_adj_ret", "abs_z60", "abs_zbeta", "drift10_z", "zero_x_bench", "ex_ratio",
          "abs_vs_class_med"]
FLAT_FEATURES = ["zero_x_bench", "stale_run"]


def coefs(train, feats, label):
    est = mm.lr_full().fit(train[feats].to_numpy(), train["is_anomaly"].to_numpy())
    return dict(zip(feats, est[-1].coef_[0]), model=label)


def main():
    log = log_to(m.RESULTS / "e3_log.txt")
    d = m.build_dataset()["data"]
    dev = development_set(d)
    hits, coef_rows, weights = [], [], []

    def keep(test, flag, config, method, rep, f):
        h = m.event_hits(test, flag)
        hits.append(pd.DataFrame(dict(config=config, method=method, rep=rep, fold=f,
                                      event_id=h.index, hit=h.values)))

    for rep in range(REPS):
        fold = mm.group_folds(dev, FOLDS, rep)
        for f in range(FOLDS):
            train, test = dev[fold != f], dev[fold == f].reset_index(drop=True)
            # label-free single features and the all-seen ablation (same for every config)
            for feat in SINGLE:
                keep(test, m.review_top(test[feat].to_numpy(), m.MAIN_BUDGET, rep), "none", f"feature: {feat}", rep, f)
            abl = [c for c in m.FULL_FEATURES if c not in FLAT_FEATURES]
            s = mm.fit_score(mm.lr_full(), abl, train, test)
            keep(test, m.review_top(s, m.MAIN_BUDGET, rep), "none", "Logistic, full minus flat-NAV features", rep, f)
            coef_rows.append(dict(coefs(train, m.FULL_FEATURES, "all types seen"), config="none", rep=rep, fold=f))
            for name, types in CONFIGS.items():
                log(f"rep {rep} fold {f}: holding out {name}")
                scores, w = score_all(train, test, seed=rep, exclude_types=types)
                weights.append(dict(config=name, rep=rep, fold=f, hybrid_w=w))
                for meth, s in scores.items():
                    keep(test, m.review_top(s, m.MAIN_BUDGET, rep), name, meth, rep, f)
                coef_rows.append(dict(coefs(mm.drop_types(train, types), m.FULL_FEATURES, "type unseen"),
                                      config=name, rep=rep, fold=f))
            pd.concat(hits).to_csv(m.RESULTS / "e3_event_hits.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(m.RESULTS / "e3_coefs.csv", index=False)
    pd.DataFrame(weights).to_csv(m.RESULTS / "e3_hybrid_weights.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
