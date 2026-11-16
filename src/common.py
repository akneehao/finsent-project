"""
Shared helpers used by every step: label names, the fixed data split, picking
a labelled subset of a given size, scoring predictions, and saving one row per
run to a results file.
"""
import csv
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

LABELS = ["negative", "neutral", "positive"]          # a label's id is its position in this list
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

# Share of annotators who agreed with a sentence's label (see prepare_data.py).
# 100 = everyone agreed, 75 = 75-99%, 66 = 66-74%, 50 = 50-65%.
AGREEMENT_LEVELS = [100, 75, 66, 50]

# Labelled-data budgets and random seeds for the main experiment grid. Results vary a
# lot between seeds at small budgets, which leaves 5 seeds with little statistical
# power (a sign test can't reach p < 0.05 even if one model wins all 5), so we use 10.
BUDGETS = ["50", "100", "200", "400", "800", "all"]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

RESULT_COLUMNS = [
    "model", "budget", "n_train", "seed", "lr", "C", "steps",
    "val_acc", "val_macro_f1",
    "test_acc", "test_macro_f1", "test_f1_negative", "test_f1_neutral", "test_f1_positive",
    "test_acc_agree100", "test_acc_agree75", "test_acc_agree66", "test_acc_agree50",
    "train_seconds", "device", "torch_version", "transformers_version", "finished_at",
]


def load_split(data_dir="data"):
    """Return (train pool, validation set, test set) as three DataFrames."""
    path = os.path.join(data_dir, "phrasebank_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run `python src/prepare_data.py` first.")
    df = pd.read_csv(path)
    return tuple(df[df["split"] == name].reset_index(drop=True) for name in ("train", "val", "test"))


def parse_budget(text):
    """'200' -> 200 and 'all' -> None, meaning every sentence in the training pool."""
    return None if str(text).lower() == "all" else int(text)


def sample_budget(train_pool, budget, seed):
    """Pick `budget` labelled sentences with the same class mix as the whole pool.

    The same seed always picks the same sentences, so a BERT run and a FinBERT run
    with the same seed see exactly the same training data. That makes the
    comparison paired: a difference comes from the model, not from luck in which
    sentences were drawn.
    """
    if budget is None or budget >= len(train_pool):
        return train_pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    picked, _ = train_test_split(train_pool, train_size=budget,
                                 stratify=train_pool["label_id"], random_state=seed)
    return picked.reset_index(drop=True)


def score(df, pred_ids, prefix):
    """Accuracy and macro-F1 (the average of the three per-class F1 scores)."""
    y, p = df["label_id"].to_numpy(), np.asarray(pred_ids)
    return {f"{prefix}_acc": accuracy_score(y, p),
            f"{prefix}_macro_f1": f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)}


def score_test(test_df, pred_ids):
    """Test scores, plus F1 for each class and accuracy at each annotator-agreement level."""
    out = score(test_df, pred_ids, "test")
    y, p = test_df["label_id"].to_numpy(), np.asarray(pred_ids)
    per_class = f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
    for name, value in zip(LABELS, per_class):
        out[f"test_f1_{name}"] = value
    agreement = test_df["agreement"].to_numpy()
    for level in AGREEMENT_LEVELS:
        mask = agreement == level
        out[f"test_acc_agree{level}"] = accuracy_score(y[mask], p[mask]) if mask.any() else np.nan
    return out


def already_done(csv_path, model, budget, seed, lr=None):
    """True if a run with these settings is already saved, so it can be skipped."""
    if not os.path.exists(csv_path):
        return False
    done = pd.read_csv(csv_path)
    match = ((done["model"] == model) & (done["budget"].astype(str) == str(budget))
             & (done["seed"] == seed))
    if lr is not None:
        match &= np.isclose(done["lr"].astype(float), lr)
    return bool(match.any())


def append_row(csv_path, row):
    """Add one run's results to a CSV file, writing the header if the file is new."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in RESULT_COLUMNS})
