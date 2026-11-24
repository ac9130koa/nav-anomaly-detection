"""
Build the Milestone 2 tables (results/m2/tables/*.csv) and figures (figures/m2_*.png)
from the experiment outputs. Run after m2_e0, m2_e1, m2_e3, and m2_e4.
"""
from m2_common import *  # noqa: F401,F403
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

R = m.RESULTS
T = R / "tables"
T.mkdir(parents=True, exist_ok=True)

# palette: fixed categorical order, neutral ink for text (dataviz reference palette)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlesize": 10, "axes.titlecolor": INK, "axes.titleweight": "bold",
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 220})

TYPE_LABEL = {"stale_price": "Stale price, full", "stale_partial": "Stale price, partial",
              "missed_split": "Missed split", "missed_dividend": "Missed dividend",
              "expense_error": "Expense error, 1 day", "accrual_drift": "Accrual drift"}
TYPE_ORDER = ["stale_price", "stale_partial", "missed_split", "missed_dividend", "expense_error",
              "accrual_drift"]
SHORT = {"Rule: raw move": "Raw 50 bps rule", "Rule: benchmark-adjusted move": "Adjusted 50 bps rule",
         "Rule: expert checks": "Expert rule", "Logistic, M1 features": "Logistic (M1, 3 features)",
         "Logistic, full features": "Logistic (10 features)", "Gaussian Naive Bayes": "Naive Bayes",
         "Decision tree": "Decision tree", "Random forest": "Random forest",
         "Gradient boosting": "Gradient boosting", "Isolation forest (no labels)": "Isolation forest",
         "Hybrid (logistic + isolation forest)": "Hybrid"}
METHOD_ORDER = list(SHORT)


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def e1_tables():
    res = pd.read_csv(R / "e1_fold_metrics.csv")
    nb = pd.read_csv(R / "e1_nb_tests.csv")
    r5 = res[res.budget == m.MAIN_BUDGET]
    main = r5.groupby("method").agg(event_recall=("event_recall", "mean"), sd=("event_recall", "std"),
                                    day_recall=("day_recall", "mean"), day_precision=("day_precision", "mean"),
                                    pr_auc=("pr_auc", "mean")).reindex(METHOD_ORDER)
    for ref, tag in [("Rule: benchmark-adjusted move", "rule"), ("Logistic, M1 features", "lr_m1")]:
        x = nb[nb.reference == ref].set_index("method")
        main[f"diff_{tag}"] = x["mean_diff"]
        main[f"se_naive_{tag}"] = x["se_naive"]
        main[f"se_nb_{tag}"] = x["se_corrected"]
        main[f"p_nb_{tag}"] = x["p"]
    main.to_csv(T / "t_main_5pct.csv")
    curve = res.pivot_table(index="method", columns="budget", values="event_recall", aggfunc="mean").reindex(METHOD_ORDER)
    curve.to_csv(T / "t_event_recall_by_budget.csv")
    dcurve = res.pivot_table(index="method", columns="budget", values="day_recall", aggfunc="mean").reindex(METHOD_ORDER)
    dcurve.to_csv(T / "t_day_recall_by_budget.csv")
    pcurve = res.pivot_table(index="method", columns="budget", values="day_precision", aggfunc="mean").reindex(METHOD_ORDER)
    pcurve.to_csv(T / "t_day_precision_by_budget.csv")
    nat = pd.read_csv(R / "e1_natural_rules.csv").groupby("method")[
        ["review_rate", "event_recall", "day_recall", "day_precision"]].mean()
    nat.to_csv(T / "t_natural_rules.csv")

    # recall by error type and by error size, pooled over all 25 folds (event level)
    hits = pd.read_csv(R / "e1_event_hits.csv")
    d = m.build_dataset()["data"]
    et = event_table(development_set(d))
    et["bucket"] = m.size_bucket(et["size_bps"]).astype(str)
    hits = hits.join(et[["atype", "bucket", "sev", "asset_class"]], on="event_id")
    by_type = hits.pivot_table(index="method", columns="atype", values="hit", aggfunc="mean").reindex(
        index=METHOD_ORDER, columns=TYPE_ORDER)
    by_size = hits.pivot_table(index="method", columns="bucket", values="hit", aggfunc="mean").reindex(
        index=METHOD_ORDER, columns=["<10 bps", "10-50 bps", ">=50 bps"])
    by_class = hits.pivot_table(index="method", columns="asset_class", values="hit", aggfunc="mean").reindex(METHOD_ORDER)
    by_type.to_csv(T / "t_recall_by_type.csv")
    by_size.to_csv(T / "t_recall_by_size.csv")
    by_class.to_csv(T / "t_recall_by_class.csv")
    counts = et.groupby("atype").size().reindex(TYPE_ORDER), et.groupby("bucket").size()
    pd.concat(counts).to_csv(T / "t_event_counts_dev.csv")
    return main, curve, dcurve, pcurve, by_type, by_size


def e3_tables():
    h = pd.read_csv(R / "e3_event_hits.csv")
    d = m.build_dataset()["data"]
    et = event_table(development_set(d))
    h = h.join(et[["atype"]], on="event_id")
    e1 = pd.read_csv(R / "e1_event_hits.csv").query("rep < 2").join(et[["atype"]], on="event_id")
    rows = []
    configs = [c for c in h.config.unique() if c != "none"]
    for cfg in configs:
        types = m.REALISTIC_TYPES if cfg.startswith("realistic") else [cfg]
        r = {"held_out": cfg}
        seen = e1[e1.atype.isin(types)].groupby("method")["hit"].mean()
        r["Seen: logistic (10 features)"] = seen["Logistic, full features"]
        r["Seen: best supervised"] = seen[[k for k in seen.index if not k.startswith(("Rule", "Isolation", "Hybrid"))]].max()
        un = h[(h.config == cfg) & h.atype.isin(types)].groupby("method")["hit"].mean()
        for k in ["Logistic, M1 features", "Logistic, full features", "Gaussian Naive Bayes", "Decision tree",
                  "Random forest", "Gradient boosting", "Hybrid (logistic + isolation forest)",
                  "Isolation forest (no labels)", "Rule: expert checks"]:
            r[f"Unseen: {SHORT[k]}"] = un.get(k, np.nan)
        lf = h[(h.config == "none") & h.atype.isin(types) & h.method.str.startswith("feature")].groupby("method")["hit"].mean()
        r["Best single feature"] = lf.max()
        r["Best single feature name"] = lf.idxmax().replace("feature: ", "")
        abl = h[(h.config == "none") & h.atype.isin(types) & (h.method == "Logistic, full minus flat-NAV features")]["hit"].mean()
        r["Seen: logistic without flat-NAV features"] = abl
        r["events"] = int(h[(h.config == cfg) & h.atype.isin(types) & (h.method == "Logistic, full features")].shape[0])
        rows.append(r)
    loto = pd.DataFrame(rows).set_index("held_out")
    loto.to_csv(T / "t_unseen_types.csv")
    c = pd.read_csv(R / "e3_coefs.csv")
    coef = c.groupby(["config", "model"])[m.FULL_FEATURES].mean()
    coef.to_csv(T / "t_logistic_coefs.csv")
    w = pd.read_csv(R / "e3_hybrid_weights.csv").groupby("config")["hybrid_w"].agg(["mean", "min", "max"])
    w.to_csv(T / "t_hybrid_weights_unseen.csv")
    return loto, coef


def e4_tables():
    b = pd.read_csv(R / "e4_bootstrap.csv")
    b.to_csv(T / "t_holdout.csv", index=False)
    met = pd.read_csv(R / "e4_metrics.csv")
    s = met.groupby(["cell", "method", "budget"])[["event_recall", "day_recall", "day_precision", "pr_auc"]].mean()
    s.to_csv(T / "t_holdout_metrics.csv")
    h = pd.read_csv(R / "e4_event_hits.csv")
    bt = h[h.cell == "Unseen funds, future period"].pivot_table(index="method", columns="atype", values="hit",
                                                                aggfunc="mean").reindex(index=METHOD_ORDER, columns=TYPE_ORDER)
    bt.to_csv(T / "t_holdout_by_type.csv")
    return b, s


# ----------------------------------------------------------------------------- figures
def fig_audit():
    ev = pd.read_csv(R / "e0_events.csv")
    sf = pd.read_csv(R / "e0_single_feature.csv").set_index("feature")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, k in enumerate(TYPE_ORDER):
        x = ev.loc[ev.atype == k, "size_bps"].to_numpy()
        y = i + rng.uniform(-0.25, 0.25, len(x))
        ax.scatter(x, y, s=9, color=SERIES[0], alpha=0.45, lw=0)
        ax.plot([np.median(x)] * 2, [i - 0.34, i + 0.34], color=INK, lw=1.6)
    ax.axvline(50, color=SERIES[1], lw=1.2, ls="--")
    ax.text(56, len(TYPE_ORDER) - 0.42, "0.5% of NAV", color=INK2, fontsize=7.5, va="bottom")
    ax.set_xscale("log")
    ax.set_yticks(range(len(TYPE_ORDER)), [TYPE_LABEL[k] for k in TYPE_ORDER])
    ax.invert_yaxis()
    ax.set_ylim(len(TYPE_ORDER) - 0.1, -0.5)
    ax.set_xlabel("Largest error in event (bps of NAV, log scale)")
    ax.set_title("A. Error size (bar = median)", loc="left")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    ax = axes[1]
    feats = ["abs_adj_ret", "abs_z60", "abs_zbeta", "drift10_z", "zero_x_bench", "ex_ratio"]
    flab = ["Adjusted\nmove", "z-score\n(60 d)", "Beta\nz-score", "Drift\n(10 d)", "Flat\nNAV", "Ex-date\ngap"]
    M = sf.loc[feats, TYPE_ORDER].T.to_numpy()
    ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(feats)), flab, fontsize=7.5)
    ax.set_yticks(range(len(TYPE_ORDER)), [TYPE_LABEL[k] for k in TYPE_ORDER], fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("B. Recall at 5%, one feature, no model", loc="left")
    fig.tight_layout()
    fig.savefig(m.FIGURES / "m2_fig1_error_audit.png", bbox_inches="tight")
    plt.close(fig)


def fig_budget(curve, dcurve, show):
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2), sharey=True)
    budgets = [b * 100 for b in m.BUDGETS]
    for ax, data, title in [(axes[0], curve, "A. Event recall"),
                            (axes[1], dcurve, "B. Fund-day recall")]:
        for i, meth in enumerate(show):
            y = data.loc[meth].to_numpy()
            ax.plot(budgets, y, color=SERIES[i], lw=2, marker="o", ms=4.5, mec=SURF, mew=1)
        ax.axvline(5, color=GRID, lw=1, zorder=0)
        ax.set_xticks(budgets, ["1%", "2%", "5%", "10%"])
        ax.set_xlim(0.4, 10.6)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Share of fund-days reviewed")
        ax.set_title(title, loc="left")
        style(ax)
    axes[0].set_ylabel("Recall (mean of 25 folds)")
    handles = [plt.Line2D([], [], color=SERIES[i], lw=2, marker="o", ms=4) for i in range(len(show))]
    fig.legend(handles, [SHORT[s] for s in show], loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(m.FIGURES / "m2_fig2_recall_by_budget.png", bbox_inches="tight")
    plt.close(fig)


def fig_unseen(loto):
    cols = ["Seen: logistic (10 features)", "Unseen: Logistic (M1, 3 features)", "Unseen: Logistic (10 features)",
            "Unseen: Decision tree", "Unseen: Random forest", "Unseen: Gradient boosting", "Unseen: Hybrid",
            "Unseen: Isolation forest", "Best single feature"]
    lab = ["Logistic\n(10 feat.)", "Logistic\n(M1, 3 feat.)", "Logistic\n(10 feat.)", "Decision\ntree", "Random\nforest",
           "Gradient\nboosting", "Hybrid", "Isolation\nforest", "Best single\nfeature*"]
    rows = TYPE_ORDER + ["realistic pair (stale_partial + accrual_drift)"]
    rlab = [TYPE_LABEL[k] for k in TYPE_ORDER] + ["Both realistic types"]
    M = loto.reindex(rows)[cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(cols)), lab, fontsize=7.5)
    ax.set_yticks(range(len(rows)), rlab)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for x in [0.5, 6.5]:
        ax.axvline(x, color="white", lw=4)
    ax.text(0, -0.8, "Type in\ntraining", ha="center", fontsize=8, color=INK2)
    ax.text(3.5, -0.8, "Type removed from training (models refit)", ha="center", fontsize=8, color=INK2)
    ax.text(7.5, -0.8, "No labels used", ha="center", fontsize=8, color=INK2)
    ax.set_ylim(len(rows) - 0.5, -1.25)
    fig.tight_layout()
    fig.savefig(m.FIGURES / "m2_fig3_unseen_types.png", bbox_inches="tight")
    plt.close(fig)


def main():
    main_t, curve, dcurve, pcurve, by_type, by_size = e1_tables()
    loto, coef = e3_tables() if (R / "e3_coefs.csv").exists() else (None, None)
    if (R / "e4_bootstrap.csv").exists():
        e4_tables()
    fig_audit()
    show = ["Rule: benchmark-adjusted move", "Rule: expert checks", "Logistic, M1 features",
            "Gradient boosting", "Isolation forest (no labels)"]
    fig_budget(curve, dcurve, show)
    if loto is not None:
        fig_unseen(loto)
    pd.set_option("display.width", 250)
    print(main_t.round(3).to_string())
    print(curve.round(3).to_string())
    print(by_type.round(2).to_string())
    print(by_size.round(2).to_string())
    if loto is not None:
        print(loto.round(2).to_string())


if __name__ == "__main__":
    main()
