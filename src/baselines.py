"""
Step 3: two reference points for the report.

TF-IDF + logistic regression
    A classic, non-neural text classifier trained on exactly the same labelled
    subsets (same budgets, same seeds) as the BERT runs. Its regularisation
    strength C is chosen on the validation set for every run. It shows how much
    the pretrained models really add. Rows go into results/results.csv as model
    "tfidf".

Leakage check
    ProsusAI/finbert is a popular finance sentiment model, but it was fine-tuned
    on Financial PhraseBank itself, so it has probably seen many of our test
    sentences. We score it on our test set with no training at all. A very high
    score is evidence of leakage, which is why it isn't one of the compared
    models. The row goes into results/leakage_check.csv.

    python src/baselines.py
    python src/baselines.py --skip_leakage_check
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import (BUDGETS, LABEL2ID, SEEDS, append_row, already_done, load_split,
                    parse_budget, sample_budget, score, score_test)


# Regularisation strengths to try. Small C = strong regularisation, which on tiny
# training sets makes the classifier predict "neutral" for everything.
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]


def fit_tfidf(train_df, C):
    """Fit word and word-pair counts plus logistic regression on the training subset only."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    classifier = LogisticRegression(C=C, max_iter=2000)
    classifier.fit(vectorizer.fit_transform(train_df["sentence"]), train_df["label_id"])
    return lambda sentences: classifier.predict(vectorizer.transform(sentences))


def run_tfidf(train_pool, val_df, test_df, results_csv):
    for seed in SEEDS:
        for budget_name in BUDGETS:
            if already_done(results_csv, "tfidf", budget_name, seed):
                continue
            train_df = sample_budget(train_pool, parse_budget(budget_name), seed)
            # Pick C on the validation set (never the test set), so the baseline gets a fair chance.
            candidates = [(C, fit_tfidf(train_df, C)) for C in C_GRID]
            best_C, predict = max(candidates,
                                  key=lambda c: score(val_df, c[1](val_df["sentence"]), "val")["val_macro_f1"])

            row = {"model": "tfidf", "budget": budget_name, "n_train": len(train_df), "seed": seed, "C": best_C}
            row.update(score(val_df, predict(val_df["sentence"]), "val"))
            row.update(score_test(test_df, predict(test_df["sentence"])))
            append_row(results_csv, row)
            print(f"tfidf n={budget_name} seed={seed} C={best_C:g}: test macro-F1 {row['test_macro_f1']:.3f}")


def run_leakage_check(test_df, out_csv):
    import torch
    from transformers import BertForSequenceClassification
    try:
        from transformers import BertTokenizerFast as BertTokenizer
    except ImportError:
        from transformers import BertTokenizer

    name = "ProsusAI/finbert"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(name)
    model = BertForSequenceClassification.from_pretrained(name).to(device).eval()

    # The model orders its outputs its own way, so map each output position to our label ids.
    try:
        to_ours = np.array([LABEL2ID[model.config.id2label[i].lower()] for i in range(model.config.num_labels)])
    except KeyError as err:
        raise SystemExit(f"Can't match {name}'s labels {model.config.id2label} to ours: {err}")

    preds = []
    with torch.no_grad():
        for start in range(0, len(test_df), 64):
            batch = test_df["sentence"].iloc[start:start + 64].tolist()
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            preds.append(model(**enc).logits.argmax(dim=-1).cpu().numpy())

    row = {"model": f"{name} (no training on our data)"}
    row.update(score_test(test_df, to_ours[np.concatenate(preds)]))
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    print(f"Leakage check, {name} with no training: test accuracy {row['test_acc']:.3f}, "
          f"macro-F1 {row['test_macro_f1']:.3f}, accuracy where all annotators agreed "
          f"{row['test_acc_agree100']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="TF-IDF baseline and FinBERT leakage check.")
    parser.add_argument("--skip_leakage_check", action="store_true")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    train_pool, val_df, test_df = load_split(args.data_dir)
    run_tfidf(train_pool, val_df, test_df, os.path.join(args.results_dir, "results.csv"))
    if not args.skip_leakage_check:
        run_leakage_check(test_df, os.path.join(args.results_dir, "leakage_check.csv"))


if __name__ == "__main__":
    main()
