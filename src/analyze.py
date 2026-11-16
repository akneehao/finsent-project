"""
Step 4: turn the results files into the tables and figures for the report.

    python src/analyze.py              # after the main runs
    python src/analyze.py --lr_check   # after the learning-rate check
    python src/analyze.py --results_dir results_main --longer_dir results_longer --clean_dir results_clean

Main mode writes:
  results/summary_table.csv       mean and spread of test scores per model and budget
  results/paired_comparison.csv   FinBERT minus BERT per budget, comparing same-seed runs
  results/labels_saved.csv        how many labels FinBERT's finance reading is worth
  results/error_analysis_n<budget>.csv   test sentences one model gets right and the other wrong
  results/all_tests.csv           every significance test, with Holm-adjusted p-values
  results/stability_check.csv     spread between seeds and collapsed runs at small budgets
  results/seed_groups.csv         FinBERT's lead in seeds 0-4 and seeds 5-9, with library versions
  results/predicted_classes.csv   which label the models answer at each agreement level
  results/error_sources.csv       how many mistakes come from sentences annotators disagreed on
  figures/learning_curve.png, figures/accuracy_by_agreement.png, figures/confusion_<model>_n<budget>.png

Confusion matrices and error lists are made for 200 labels and all labels by default
(change with --detail_budgets).

The two extra experiments are compared with the main runs when their folders are given:
  --longer_dir   <longer_dir>/steps_check.csv and updates_effect.csv
                 the same runs with more updates
  --clean_dir    <clean_dir>/clean_check.csv and figures/clean_vs_random.png
                 training only on sentences every annotator agreed on
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")                     # write image files without needing a screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import AGREEMENT_LEVELS, LABELS, load_split

NAMES = {"tfidf": "TF-IDF + logistic regression",
         "bert": "BERT (general English)",
         "finbert": "FinBERT (finance text)"}
COLORS = {"tfidf": "#8a8f98", "bert": "#2a59c9", "finbert": "#c2410c"}
AGREEMENT_TICKS = ["100%", "75-99%", "66-74%", "50-65%"]
COLLAPSED_BELOW = 0.3                     # a run has collapsed if its negative-class F1 is below this


def budget_order(budget):
    return float("inf") if str(budget) == "all" else int(budget)


def models_in(df):
    return [m for m in NAMES if m in set(df["model"])]


def summary_table(df):
    table = (df.groupby(["model", "budget"])
               .agg(n_train=("n_train", "mean"), runs=("seed", "count"),
                    macro_f1_mean=("test_macro_f1", "mean"), macro_f1_std=("test_macro_f1", "std"),
                    acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"))
               .reset_index())
    table["order"] = table["budget"].map(budget_order)
    return table.sort_values(["model", "order"]).drop(columns="order")


def print_summary(table):
    print("\nTest macro-F1, mean ± standard deviation over seeds")
    for model in models_in(table):
        for _, r in table[table["model"] == model].iterrows():
            print(f"  {NAMES[model]:<30} {r['budget']:>4} labels   "
                  f"{r['macro_f1_mean']:.3f} ± {r['macro_f1_std']:.3f}   ({r['runs']} runs)")


def plot_learning_curve(table, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for model in models_in(table):
        t = table[table["model"] == model]
        ax.errorbar(t["n_train"], t["macro_f1_mean"], yerr=t["macro_f1_std"].fillna(0),
                    marker="o", capsize=3, color=COLORS[model], label=NAMES[model])
    ticks = table.drop_duplicates("budget").sort_values("n_train")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(ticks["n_train"])
    ax.set_xticklabels([f"all\n({int(n)})" if b == "all" else b for b, n in zip(ticks["budget"], ticks["n_train"])])
    ax.set_xlabel("Labelled training sentences (log scale)")
    ax.set_ylabel("Test macro-F1")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def paired_t_p(a, b):
    """p-value of a paired t-test, or NaN if scipy is missing or there are too few pairs."""
    if len(a) < 3:
        return np.nan
    try:
        from scipy.stats import ttest_rel
    except ImportError:
        return np.nan
    return float(ttest_rel(a, b).pvalue)


def welch_t_p(a, b):
    """p-value of Welch's t-test for two independent groups of runs, or NaN as above."""
    if min(len(a), len(b)) < 3:
        return np.nan
    try:
        from scipy.stats import ttest_ind
    except ImportError:
        return np.nan
    return float(ttest_ind(a, b, equal_var=False).pvalue)


def mean_ci(values):
    """95% confidence interval for a mean, or NaNs if scipy is missing or there are too few values."""
    if len(values) < 3:
        return np.nan, np.nan
    try:
        from scipy.stats import t
    except ImportError:
        return np.nan, np.nan
    half_width = t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values))
    return float(np.mean(values) - half_width), float(np.mean(values) + half_width)


def holm_adjust(p_values):
    """Holm-adjusted p-values, so that a family of tests keeps a 5% overall chance of a false alarm.

    With many tests, a few p < 0.05 turn up by luck alone. Holm's method sorts the
    p-values, multiplies the smallest by the number of tests, the next by one fewer and
    so on, and keeps the results in order. NaNs are left out of the count.
    """
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok])]
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (len(order) - rank) * p[i])
        adjusted[i] = min(1.0, running_max)
    return adjusted


def paired_comparison(df):
    """FinBERT minus BERT for each budget. Same seed = same training sentences, so runs pair up."""
    wide = (df[df["model"].isin(["bert", "finbert"])]
            .pivot_table(index=["budget", "seed"], columns="model", values="test_macro_f1"))
    if not {"bert", "finbert"} <= set(wide.columns):
        return pd.DataFrame()
    wide = wide.dropna()
    wide["diff"] = wide["finbert"] - wide["bert"]
    rows = []
    for budget, g in wide.groupby(level="budget"):
        ci_low, ci_high = mean_ci(g["diff"])
        rows.append({"budget": budget, "pairs": len(g),
                     "mean_diff": g["diff"].mean(), "std_diff": g["diff"].std(),
                     "ci95_low": ci_low, "ci95_high": ci_high,
                     "finbert_wins": int((g["diff"] > 0).sum()),
                     "p_value_paired_t": paired_t_p(g["finbert"], g["bert"])})
    out = pd.DataFrame(rows)
    out["order"] = out["budget"].map(budget_order)
    return out.sort_values("order").drop(columns="order")


def print_paired(paired, title):
    print(f"\n{title}")
    for _, r in paired.iterrows():
        print(f"  {r['budget']:>4} labels: {r['mean_diff']:+.3f} ± {r['std_diff']:.3f}, "
              f"FinBERT better in {r['finbert_wins']} of {r['pairs']} seeds, p = {r['p_value_paired_t']:.3f}, "
              f"95% CI {r['ci95_low']:+.3f} to {r['ci95_high']:+.3f}")


def labels_saved(table):
    """For each FinBERT budget, roughly how many labels general BERT needs to match it.

    This reads BERT's learning curve backwards, interpolating between the measured
    budgets on a log scale, so treat the numbers as rough estimates.
    """
    bert = table[table["model"] == "bert"].sort_values("n_train")
    fin = table[table["model"] == "finbert"].sort_values("n_train")
    if bert.empty or fin.empty:
        return pd.DataFrame()
    x = np.log(bert["n_train"].to_numpy(float))
    y = np.maximum.accumulate(bert["macro_f1_mean"].to_numpy())   # non-decreasing, so it can be inverted
    rows = []
    for _, r in fin.iterrows():
        target = r["macro_f1_mean"]
        if target > y[-1]:
            # Past the end of BERT's curve, so there is nothing to read off. Print both scores
            # instead of calling it "better", since the gap here is usually within seed noise.
            needed, note = np.nan, (f"at {target:.3f}, level with or above BERT with all "
                                    f"{int(bert['n_train'].max())} labels ({y[-1]:.3f})")
        elif target < y[0]:
            needed, note = np.nan, f"worse than BERT with {int(bert['n_train'].min())} labels"
        else:
            needed = float(np.exp(np.interp(target, y, x)))
            note = f"about as good as BERT with ~{needed:.0f} labels"
        rows.append({"finbert_budget": r["budget"], "finbert_n_train": int(r["n_train"]),
                     "finbert_macro_f1": target, "bert_labels_needed": needed, "note": note})
    return pd.DataFrame(rows)


def plot_by_agreement(df, path, budgets=("200", "all")):
    budgets = [b for b in budgets if b in set(df["budget"])]
    if not budgets:
        return
    models = models_in(df)
    width = 0.8 / len(models)
    x = np.arange(len(AGREEMENT_LEVELS))
    fig, axes = plt.subplots(1, len(budgets), figsize=(5.4 * len(budgets), 3.9), sharey=True, squeeze=False)
    for ax, budget in zip(axes[0], budgets):
        sub = df[df["budget"] == budget]
        for i, model in enumerate(models):
            values = [sub.loc[sub["model"] == model, f"test_acc_agree{level}"].mean() for level in AGREEMENT_LEVELS]
            ax.bar(x + (i - (len(models) - 1) / 2) * width, values, width, color=COLORS[model], label=NAMES[model])
        ax.set_xticks(x)
        ax.set_xticklabels(AGREEMENT_TICKS)
        ax.set_xlabel("Annotators who agreed with the label")
        ax.set_title("All labelled sentences" if budget == "all" else f"{budget} labelled sentences")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("Test accuracy")
    axes[0][-1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def prediction_files(pred_dir, model, budget):
    return sorted(glob.glob(os.path.join(pred_dir, f"{model}_n{budget}_seed*.csv")))


def plot_confusions(test_df, pred_dir, fig_dir, budget):
    """One confusion matrix per model, adding up the test predictions of every seed."""
    for model in ["bert", "finbert"]:
        files = prediction_files(pred_dir, model, budget)
        if not files:
            continue
        cm = np.zeros((3, 3), dtype=int)
        for f in files:
            merged = pd.read_csv(f).merge(test_df[["id", "label_id"]], on="id")
            np.add.at(cm, (merged["label_id"].to_numpy(), merged["pred_id"].to_numpy()), 1)
        fig, ax = plt.subplots(figsize=(4.4, 3.9))
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(3))
        ax.set_xticklabels(LABELS)
        ax.set_yticks(range(3))
        ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"{NAMES[model]}\n{budget} labels, {len(files)} seeds added together", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"confusion_{model}_n{budget}.png"), dpi=200)
        plt.close(fig)


def error_analysis(test_df, pred_dir, budget):
    """Test sentences where one model is right in at least 80% of seeds and the other in at most 20%."""
    right_share, usual_label = {}, {}
    for model in ["bert", "finbert"]:
        files = prediction_files(pred_dir, model, budget)
        if not files:
            return pd.DataFrame()
        preds = pd.DataFrame({f: pd.read_csv(f).set_index("id")["pred_id"].reindex(test_df["id"]).to_numpy()
                              for f in files})
        right_share[model] = (preds.to_numpy() == test_df["label_id"].to_numpy()[:, None]).mean(axis=1)
        usual_label[model] = preds.mode(axis=1)[0].astype(int).map(lambda i: LABELS[i]).to_numpy()
    out = test_df[["id", "sentence", "label", "agreement"]].copy()
    out["bert_right_share"] = right_share["bert"]
    out["finbert_right_share"] = right_share["finbert"]
    out["bert_usual_label"] = usual_label["bert"]
    out["finbert_usual_label"] = usual_label["finbert"]
    out["pattern"] = np.select(
        [(out["finbert_right_share"] >= 0.8) & (out["bert_right_share"] <= 0.2),
         (out["bert_right_share"] >= 0.8) & (out["finbert_right_share"] <= 0.2),
         (out["bert_right_share"] <= 0.2) & (out["finbert_right_share"] <= 0.2)],
        ["finbert_wins", "bert_wins", "both_wrong"], default="")
    return out[out["pattern"] != ""].sort_values(["pattern", "agreement"])


def print_both_wrong(errors):
    """Which wrong label the models usually pick on the sentences both get wrong."""
    both = errors[errors["pattern"] == "both_wrong"]
    if both.empty:
        return
    same = (both["bert_usual_label"] == both["finbert_usual_label"]).sum()
    print(f"Both models usually pick the same wrong label on {same} of {len(both)} both-wrong sentences; "
          f"{(both['agreement'] < 75).sum()} of them have under 75% agreement.")
    for model in ["bert", "finbert"]:
        print(f"{NAMES[model]}, true label (rows) against its usual wrong label (columns):")
        print(pd.crosstab(both["label"], both[f"{model}_usual_label"]).to_string())


def seeds_with_both(df, budget):
    """Seeds for which both BERT and FinBERT have finished at this budget."""
    sub = df[(df["budget"] == budget) & df["model"].isin(["bert", "finbert"])]
    models_per_seed = sub.groupby("seed")["model"].nunique()
    return set(models_per_seed[models_per_seed == 2].index)


def bert_vs_finbert(df, budget, seeds):
    """Mean scores and the same-seed FinBERT-minus-BERT gap at one budget, over the given seeds."""
    sub = df[(df["budget"] == budget) & df["seed"].isin(seeds) & df["model"].isin(["bert", "finbert"])]
    wide = sub.pivot_table(index="seed", columns="model", values="test_macro_f1")
    return {"budget": budget, "steps": int(sub["steps"].median()), "seeds": len(wide),
            "bert": wide["bert"].mean(), "finbert": wide["finbert"].mean(),
            "finbert_minus_bert": (wide["finbert"] - wide["bert"]).mean(),
            "finbert_wins": int((wide["finbert"] > wide["bert"]).sum()),
            "p_value_paired_t": paired_t_p(wide["finbert"], wide["bert"]),
            "bert_worst_negative_f1": sub.loc[sub["model"] == "bert", "test_f1_negative"].min()}


def steps_check(main_df, longer_df):
    """Does FinBERT's lead at small budgets survive training for longer?

    The longer runs differ from the main runs only in their number of updates, so a
    lead that comes only from FinBERT learning faster would shrink here. Only seeds
    that finished in both folders are compared.
    """
    rows = []
    for budget in sorted(set(longer_df["budget"]), key=budget_order):
        seeds = seeds_with_both(main_df, budget) & seeds_with_both(longer_df, budget)
        if seeds:
            rows += [bert_vs_finbert(main_df, budget, seeds), bert_vs_finbert(longer_df, budget, seeds)]
    return pd.DataFrame(rows)


def updates_effect(main_df, longer_df):
    """How much extra updates change each model on its own, comparing each seed with itself."""
    rows = []
    for model in ["bert", "finbert"]:
        for budget in sorted(set(longer_df["budget"]), key=budget_order):
            scores = [d[(d["model"] == model) & (d["budget"] == budget)].set_index("seed")["test_macro_f1"]
                      for d in (main_df, longer_df)]
            seeds = scores[0].index.intersection(scores[1].index)
            if len(seeds):
                before, after = scores[0][seeds], scores[1][seeds]
                rows.append({"model": model, "budget": budget, "main_updates": before.mean(),
                             "more_updates": after.mean(), "change": (after - before).mean(),
                             "p_value_paired_t": paired_t_p(after, before)})
    return pd.DataFrame(rows)


def clean_check(main_df, clean_df):
    """Is it better to train only on sentences that every annotator agreed on?

    Compares the main runs (a random sample of the whole training pool) with runs on
    the same number of unanimously labelled sentences. The two groups see different
    sentences, so they're compared with Welch's t-test rather than seed by seed.
    'all' is left out because it means a different number of sentences in each pool.
    """
    rows = []
    for model in ["bert", "finbert"]:
        for budget in sorted(set(clean_df["budget"]) - {"all"}, key=budget_order):
            random_runs = main_df[(main_df["model"] == model) & (main_df["budget"] == budget)]
            clean_runs = clean_df[(clean_df["model"] == model) & (clean_df["budget"] == budget)]
            if random_runs.empty or clean_runs.empty:
                continue
            row = {"model": model, "budget": budget,
                   "runs_random": len(random_runs), "runs_clean": len(clean_runs),
                   "random_macro_f1": random_runs["test_macro_f1"].mean(),
                   "clean_macro_f1": clean_runs["test_macro_f1"].mean(),
                   "random_std": random_runs["test_macro_f1"].std(),
                   "clean_std": clean_runs["test_macro_f1"].std()}
            row["clean_minus_random"] = row["clean_macro_f1"] - row["random_macro_f1"]
            row["p_value_welch"] = welch_t_p(clean_runs["test_macro_f1"], random_runs["test_macro_f1"])
            for level in AGREEMENT_LEVELS:
                row[f"random_acc_agree{level}"] = random_runs[f"test_acc_agree{level}"].mean()
                row[f"clean_acc_agree{level}"] = clean_runs[f"test_acc_agree{level}"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def plot_clean_vs_random(main_df, clean_df, path, budget="200"):
    """Left: test macro-F1 against budget for both kinds of training sentences.
    Right: test accuracy by annotator agreement at one budget."""
    budgets = sorted(set(clean_df["budget"]) - {"all"}, key=budget_order)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    kinds = [(main_df, "random sentences"), (clean_df, "unanimous only")]
    for model in ["bert", "finbert"]:
        for df, kind in kinds:
            t = summary_table(df[df["budget"].isin(budgets)])
            t = t[t["model"] == model]
            left.errorbar(t["n_train"], t["macro_f1_mean"], yerr=t["macro_f1_std"].fillna(0), marker="o", capsize=3,
                          linestyle="-" if df is main_df else "--", color=COLORS[model], label=f"{NAMES[model]}, {kind}")
    left.set_xscale("log")
    left.minorticks_off()
    left.set_xticks([int(b) for b in budgets])
    left.set_xticklabels(budgets)
    left.set_xlabel("Labelled training sentences (log scale)")
    left.set_ylabel("Test macro-F1")
    left.grid(alpha=0.3)
    left.legend(loc="lower right", fontsize=8)

    bars = [(model, df, kind) for model in ["bert", "finbert"] for df, kind in kinds]
    width = 0.8 / len(bars)
    x = np.arange(len(AGREEMENT_LEVELS))
    for i, (model, df, kind) in enumerate(bars):
        sub = df[(df["model"] == model) & (df["budget"] == budget)]
        values = [sub[f"test_acc_agree{level}"].mean() for level in AGREEMENT_LEVELS]
        right.bar(x + (i - (len(bars) - 1) / 2) * width, values, width, color=COLORS[model],
                  alpha=1.0 if df is main_df else 0.45, hatch=None if df is main_df else "//",
                  label=f"{NAMES[model]}, {kind}")
    right.set_xticks(x)
    right.set_xticklabels(AGREEMENT_TICKS)
    right.set_xlabel("Annotators who agreed with the test label")
    right.set_ylabel("Test accuracy")
    right.set_title(f"{budget} labelled sentences")
    right.set_ylim(0, 1)
    right.grid(axis="y", alpha=0.3)
    right.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def read_results(results_dir):
    """A folder's results.csv as a DataFrame, or None if it doesn't exist yet."""
    path = os.path.join(results_dir, "results.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["budget"] = df["budget"].astype(str)
    return df


def report_longer(main_df, longer_dir):
    longer_df = read_results(longer_dir)
    if longer_df is None:
        print(f"\n{longer_dir}/results.csv not found yet, so the longer-training check is skipped.")
        return
    steps = steps_check(main_df, longer_df)
    steps.to_csv(os.path.join(longer_dir, "steps_check.csv"), index=False)
    print("\nTraining for longer: test macro-F1 of the same seeds with the main number of updates and with more")
    print(steps.round(3).to_string(index=False))
    effect = updates_effect(main_df, longer_df)
    effect.to_csv(os.path.join(longer_dir, "updates_effect.csv"), index=False)
    print("\nThe same seeds with more updates, each model compared with itself:")
    print(effect.round(3).to_string(index=False))
    return steps


def report_clean(main_df, clean_dir, train_pool, figures_dir):
    clean_df = read_results(clean_dir)
    if clean_df is None:
        print(f"\n{clean_dir}/results.csv not found yet, so the unanimous-labels experiment is skipped.")
        return

    def label_mix(df):
        shares = df["label"].value_counts(normalize=True).reindex(LABELS).fillna(0)
        return ", ".join(f"{label} {share:.0%}" for label, share in shares.items())

    unanimous = train_pool[train_pool["agreement"] == 100]
    print(f"\nUnanimous sentences: {len(unanimous)} of the {len(train_pool)} in the training pool "
          f"({label_mix(unanimous)}; whole pool {label_mix(train_pool)})")
    clean = clean_check(main_df, clean_df)
    clean.to_csv(os.path.join(clean_dir, "clean_check.csv"), index=False)
    print("Test macro-F1 when training only on unanimous sentences vs a random sample of the same size:")
    print(clean[["model", "budget", "runs_random", "runs_clean", "random_macro_f1", "clean_macro_f1",
                 "clean_minus_random", "p_value_welch"]].round(3).to_string(index=False))

    at_200 = clean[clean["budget"] == "200"]
    if not at_200.empty:
        print("\nTest accuracy by annotator agreement at 200 labels (100%, 75-99%, 66-74%, 50-65%):")
        for _, r in at_200.iterrows():
            for kind in ["random", "clean"]:
                values = "  ".join(f"{r[f'{kind}_acc_agree{level}']:.3f}" for level in AGREEMENT_LEVELS)
                print(f"  {NAMES[r['model']]:<24} {kind:<7} {values}")

    paired = paired_comparison(clean_df)
    if not paired.empty:
        print_paired(paired, "FinBERT minus BERT when both train on the same unanimous sentences:")
    plot_clean_vs_random(main_df, clean_df, os.path.join(figures_dir, "clean_vs_random.png"))
    return clean, paired


def all_tests(paired, steps, clean, clean_paired):
    """Every significance test in the report in one table, with Holm-adjusted p-values."""
    rows = []
    for _, r in paired.iterrows():
        rows.append({"test": "FinBERT - BERT, main runs", "model": "", "budget": r["budget"],
                     "difference": r["mean_diff"], "p": r["p_value_paired_t"]})
    if steps is not None:
        # The longer runs only; their main-run partners are already in the rows above.
        for _, r in steps[steps["steps"] > steps.groupby("budget")["steps"].transform("min")].iterrows():
            rows.append({"test": f"FinBERT - BERT, {r['steps']} updates", "model": "", "budget": r["budget"],
                         "difference": r["finbert_minus_bert"], "p": r["p_value_paired_t"]})
    if clean is not None:
        for _, r in clean.iterrows():
            rows.append({"test": "unanimous - random sentences", "model": r["model"], "budget": r["budget"],
                         "difference": r["clean_minus_random"], "p": r["p_value_welch"]})
    if clean_paired is not None:
        for _, r in clean_paired.iterrows():
            rows.append({"test": "FinBERT - BERT, unanimous sentences", "model": "", "budget": r["budget"],
                         "difference": r["mean_diff"], "p": r["p_value_paired_t"]})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p"])
    return out


def spread_tests(bert_scores, finbert_scores):
    """p-values for a difference in spread between two groups of runs, or NaNs if scipy is missing.

    The F-test is the textbook choice, but a few unusual runs can throw it off.
    Brown-Forsythe compares distances from the median, so it is safer with few runs.
    """
    try:
        from scipy.stats import f, levene
    except ImportError:
        return np.nan, np.nan
    ratio = np.var(bert_scores, ddof=1) / np.var(finbert_scores, ddof=1)
    tail = f.sf(ratio, len(bert_scores) - 1, len(finbert_scores) - 1)
    return float(2 * min(tail, 1 - tail)), float(levene(bert_scores, finbert_scores, center="median").pvalue)


def stability_check(experiments):
    """Spread between seeds and collapsed runs in every experiment, at 50 to 200 labels."""
    rows = []
    for name, df in experiments.items():
        for budget in ["50", "100", "200"]:
            bert = df[(df["model"] == "bert") & (df["budget"] == budget)]
            finbert = df[(df["model"] == "finbert") & (df["budget"] == budget)]
            if len(bert) < 3 or len(finbert) < 3:
                continue
            p_f, p_brown_forsythe = spread_tests(bert["test_macro_f1"], finbert["test_macro_f1"])
            rows.append({"experiment": name, "budget": budget, "runs_each": len(bert),
                         "bert_std": bert["test_macro_f1"].std(), "finbert_std": finbert["test_macro_f1"].std(),
                         "bert_collapsed": int((bert["test_f1_negative"] < COLLAPSED_BELOW).sum()),
                         "finbert_collapsed": int((finbert["test_f1_negative"] < COLLAPSED_BELOW).sum()),
                         "p_f_test": p_f, "p_brown_forsythe": p_brown_forsythe})
    return pd.DataFrame(rows)


def report_stability(experiments, results_dir):
    """Checks on the stability claim. These are kept apart from the mean comparisons in all_tests.csv."""
    stability = stability_check(experiments)
    if stability.empty:
        return
    stability.to_csv(os.path.join(results_dir, "stability_check.csv"), index=False)
    print("\nSpread between seeds and collapsed runs (negative-class F1 below "
          f"{COLLAPSED_BELOW}), at 50 to 200 labels:")
    print(stability.round(3).to_string(index=False))

    main = next(iter(experiments.values()))
    bert_200 = np.sort(main.loc[(main["model"] == "bert") & (main["budget"] == "200"), "test_macro_f1"].to_numpy())
    print(f"BERT's standard deviation at 200 labels: {bert_200.std(ddof=1):.3f} with all runs, "
          f"{bert_200[2:].std(ddof=1):.3f} without its two weakest ({bert_200[0]:.3f} and {bert_200[1]:.3f})")
    rows = stability[(stability["experiment"] == stability["experiment"].iloc[0])
                     & stability["budget"].isin(["100", "200"])]
    runs, bert, finbert = rows["runs_each"].sum(), rows["bert_collapsed"].sum(), rows["finbert_collapsed"].sum()
    try:
        from scipy.stats import fisher_exact
        p = fisher_exact([[bert, runs - bert], [finbert, runs - finbert]])[1]
        print(f"Collapsed runs at 100 and 200 labels: BERT {bert} of {runs}, FinBERT {finbert} of {runs}, "
              f"Fisher's exact p = {p:.3f}")
    except ImportError:
        pass


def seed_groups(experiments):
    """FinBERT's lead in seeds 0-4 and seeds 5-9, with the library versions each group ran on.

    The main runs of the two groups used different versions of Transformers. The extra
    experiments used one version throughout, so a gap between the groups that also
    shows up there can't come from the version.
    """
    rows = []
    for name, df in experiments.items():
        sub = df[df["model"].isin(["bert", "finbert"])]
        for budget in sorted(set(sub["budget"]), key=budget_order):
            wide = sub[sub["budget"] == budget].pivot_table(index="seed", columns="model", values="test_macro_f1")
            lead = wide["finbert"] - wide["bert"]
            versions = [sorted(sub.loc[in_group, "transformers_version"].dropna().unique())
                        for in_group in (sub["seed"] < 5, sub["seed"] >= 5)]
            rows.append({"experiment": name, "budget": budget,
                         "lead_seeds_0_4": lead[lead.index < 5].mean(), "lead_seeds_5_9": lead[lead.index >= 5].mean(),
                         "transformers_seeds_0_4": "/".join(versions[0]), "transformers_seeds_5_9": "/".join(versions[1])})
    return pd.DataFrame(rows)


def repeat_check(df, lr_dir):
    """Runs repeated with the same settings and library version should give the same validation score."""
    path = os.path.join(lr_dir, "lr_check.csv")
    if not os.path.exists(path):
        return
    first = pd.read_csv(path)
    first["budget"] = first["budget"].astype(str)
    both = first.merge(df, on=["model", "budget", "seed", "lr"], suffixes=("_first", "_repeat"))
    if both.empty:
        return
    same = np.isclose(both["val_macro_f1_first"], both["val_macro_f1_repeat"], rtol=0, atol=1e-9).sum()
    print(f"\nRepeat check: {same} of {len(both)} repeated runs gave an identical validation score.")


def predicted_classes(test_df, pred_dirs):
    """What the models answer at each agreement level: accuracy, share of each label predicted, and recall.

    Recall is the share of the sentences with a label that the model finds. always_neutral
    is the accuracy of answering neutral every time, a simple floor to compare with.
    """
    rows = []
    for training, pred_dir in pred_dirs.items():
        for model in ["bert", "finbert"]:
            for budget in ["50", "100", "200", "400", "800", "all"]:
                files = prediction_files(pred_dir, model, budget)
                if not files:
                    continue
                merged = pd.concat(pd.read_csv(f) for f in files).merge(test_df[["id", "label_id", "agreement"]], on="id")
                for level in AGREEMENT_LEVELS:
                    group = merged[merged["agreement"] == level]
                    row = {"training": training, "model": model, "budget": budget, "agreement": level,
                           "accuracy": (group["pred_id"] == group["label_id"]).mean(),
                           "always_neutral": (group["label_id"] == LABELS.index("neutral")).mean()}
                    for i, label in enumerate(LABELS):
                        row[f"predicts_{label}"] = (group["pred_id"] == i).mean()
                        row[f"recall_{label}"] = (group.loc[group["label_id"] == i, "pred_id"] == i).mean()
                    rows.append(row)
    return pd.DataFrame(rows)


def error_sources(test_df, pred_dir, budget="all"):
    """How many of a run's mistakes come from sentences with under 75% agreement, averaged over seeds."""
    rows = []
    for model in ["bert", "finbert"]:
        files = prediction_files(pred_dir, model, budget)
        if not files:
            continue
        merged = pd.concat(pd.read_csv(f) for f in files).merge(test_df[["id", "label_id", "agreement"]], on="id")
        wrong = merged[merged["pred_id"] != merged["label_id"]]
        rows.append({"model": model, "budget": budget, "errors_per_run": len(wrong) / len(files),
                     "share_of_errors_under_75": (wrong["agreement"] < 75).mean(),
                     "share_of_test_under_75": (test_df["agreement"] < 75).mean()})
    return pd.DataFrame(rows)


def report_predictions(test_df, pred_dirs, results_dir):
    table = predicted_classes(test_df, pred_dirs)
    if table.empty:
        return
    table.to_csv(os.path.join(results_dir, "predicted_classes.csv"), index=False)
    print("\nMost ambiguous test sentences (50-65% agreement), by training set:")
    print(table[table["agreement"] == 50].drop(columns="agreement").round(3).to_string(index=False))
    sources = error_sources(test_df, next(iter(pred_dirs.values())))
    if not sources.empty:
        sources.to_csv(os.path.join(results_dir, "error_sources.csv"), index=False)
        print("\nWhere the mistakes with all labels come from:")
        print(sources.round(3).to_string(index=False))


def lr_check(results_dir):
    path = os.path.join(results_dir, "lr_check.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run the learning-rate check first.")
    df = pd.read_csv(path)
    df["budget"] = df["budget"].astype(str)
    table = df.pivot_table(index=["model", "lr"], columns="budget", values="val_macro_f1")
    table["mean"] = table.mean(axis=1)
    print("Validation macro-F1 for each learning rate (higher is better):")
    print(table.round(3).to_string())
    best = table["mean"].groupby(level="model").idxmax()
    suggestion = {model: f"{lr:g}" for model, (_, lr) in best.items()}
    print(f"\nSuggested BEST_LR = {suggestion}")
    print("Pick one learning rate per model now and keep it for every main run.")


def main():
    parser = argparse.ArgumentParser(description="Make the report's tables and figures.")
    parser.add_argument("--lr_check", action="store_true", help="summarise the learning-rate check instead")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--figures_dir", default="figures")
    parser.add_argument("--detail_budgets", nargs="+", default=["200", "all"],
                        help="budgets that get confusion matrices and an error-analysis list")
    parser.add_argument("--longer_dir", help="results of the longer-training runs, compared with --results_dir")
    parser.add_argument("--clean_dir", help="results of the unanimous-labels runs, compared with --results_dir")
    parser.add_argument("--lr_dir", default="results", help="folder holding lr_check.csv, for the repeat check")
    args = parser.parse_args()
    if args.lr_check:
        lr_check(args.results_dir)
        return

    df = read_results(args.results_dir)
    if df is None:
        raise SystemExit(f"{args.results_dir}/results.csv not found. Run the training commands first.")
    os.makedirs(args.figures_dir, exist_ok=True)
    train_pool, _, test_df = load_split(args.data_dir)
    pred_dir = os.path.join(args.results_dir, "predictions")

    table = summary_table(df)
    table.to_csv(os.path.join(args.results_dir, "summary_table.csv"), index=False)
    print_summary(table)
    plot_learning_curve(table, os.path.join(args.figures_dir, "learning_curve.png"))

    paired = paired_comparison(df)
    if not paired.empty:
        paired.to_csv(os.path.join(args.results_dir, "paired_comparison.csv"), index=False)
        print_paired(paired, "FinBERT minus BERT (test macro-F1), same-seed pairs:")

    saved = labels_saved(table)
    if not saved.empty:
        saved.to_csv(os.path.join(args.results_dir, "labels_saved.csv"), index=False)
        print("\nWhat FinBERT's finance reading is worth (rough, interpolated):")
        for _, r in saved.iterrows():
            print(f"  FinBERT with {r['finbert_budget']} labels is {r['note']}")

    plot_by_agreement(df, os.path.join(args.figures_dir, "accuracy_by_agreement.png"))
    # Confusion matrices and error lists for a small budget (where the models differ most) and for all labels.
    for budget in args.detail_budgets:
        plot_confusions(test_df, pred_dir, args.figures_dir, budget)
        errors = error_analysis(test_df, pred_dir, budget)
        if not errors.empty:
            errors_path = os.path.join(args.results_dir, f"error_analysis_n{budget}.csv")
            errors.to_csv(errors_path, index=False)
            print(f"\nSentences for the error analysis at {budget} labels ({errors_path}):")
            print(errors["pattern"].value_counts().to_string())
            print_both_wrong(errors)

    leak_path = os.path.join(args.results_dir, "leakage_check.csv")
    if os.path.exists(leak_path):
        leak = pd.read_csv(leak_path).iloc[0]
        print(f"\nLeakage check: {leak['model']} scores test macro-F1 {leak['test_macro_f1']:.3f}. "
              "Compare it with the table above.")

    steps = report_longer(df, args.longer_dir) if args.longer_dir else None
    clean, clean_paired = None, None
    if args.clean_dir:
        # report_clean returns None when the folder doesn't exist yet
        clean, clean_paired = report_clean(df, args.clean_dir, train_pool, args.figures_dir) or (None, None)

    if not paired.empty:
        tests = all_tests(paired, steps, clean, clean_paired)
        tests.to_csv(os.path.join(args.results_dir, "all_tests.csv"), index=False)
        print(f"\nAll {len(tests)} significance tests together. p_holm corrects for running this many tests:")
        print("a result counts as significant only if p_holm < 0.05.")
        print(tests.round(3).to_string(index=False))

        experiments = {"main runs": df}
        pred_dirs = {"random": pred_dir}
        if args.longer_dir and read_results(args.longer_dir) is not None:
            experiments["more updates"] = read_results(args.longer_dir)
        if args.clean_dir and read_results(args.clean_dir) is not None:
            experiments["unanimous only"] = read_results(args.clean_dir)
            pred_dirs["unanimous"] = os.path.join(args.clean_dir, "predictions")
        report_stability(experiments, args.results_dir)
        groups = seed_groups(experiments)
        groups.to_csv(os.path.join(args.results_dir, "seed_groups.csv"), index=False)
        print("\nFinBERT's lead in seeds 0-4 and seeds 5-9, with the Transformers version of each group:")
        print(groups.round(3).to_string(index=False))
        repeat_check(df, args.lr_dir)
        report_predictions(test_df, pred_dirs, args.results_dir)
    print(f"\nFigures saved in {args.figures_dir}/")


if __name__ == "__main__":
    main()
