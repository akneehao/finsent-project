"""
Step 1: download Financial PhraseBank, clean it, and make the one fixed
train / validation / test split that every experiment shares.

    python src/prepare_data.py

Writes data/phrasebank_clean.csv (one row per unique sentence) and
data/split_summary.txt (counts worth quoting in the report).

Run it once at the start. Don't re-split after experiments have begun:
changing which sentences are in the test set makes earlier results
incomparable.
"""
import argparse
import os
import re
import zipfile

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import StratifiedGroupKFold

from common import AGREEMENT_LEVELS, LABEL2ID

# Each file holds the sentences where at least that share of annotators agreed on
# the label, so the files are nested: AllAgree sits inside 75Agree, and so on.
FILES = {100: "Sentences_AllAgree.txt", 75: "Sentences_75Agree.txt",
         66: "Sentences_66Agree.txt", 50: "Sentences_50Agree.txt"}


def read_file(zf, name):
    """Return the (sentence, label) pairs in one file. Each line is 'sentence@label'."""
    text = zf.read(f"FinancialPhraseBank-v1.0/{name}").decode("iso-8859-1")
    pairs = []
    for line in text.splitlines():
        if "@" in line:
            sentence, label = line.rsplit("@", 1)
            pairs.append((sentence.strip(), label.strip()))
    return pairs


def read_phrasebank():
    """One row per unique sentence with its label and agreement level, plus cleaning counts."""
    zip_path = hf_hub_download(repo_id="takala/financial_phrasebank",
                               filename="data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
    with zipfile.ZipFile(zip_path) as zf:
        files = {level: read_file(zf, name) for level, name in FILES.items()}

    members = {level: {sentence for sentence, _ in pairs} for level, pairs in files.items()}
    labels_seen = {}
    for pairs in files.values():
        for sentence, label in pairs:
            labels_seen.setdefault(sentence, set()).add(label)

    stats = {"lines_in_50agree_file": len(files[50]),
             "exact_duplicates_dropped": 0,
             "sentences_with_conflicting_labels_dropped": 0}
    rows, used = [], set()
    for sentence, label in files[50]:              # the loosest file contains every sentence
        if sentence in used:
            stats["exact_duplicates_dropped"] += 1
            continue
        used.add(sentence)
        if len(labels_seen[sentence]) > 1:         # same text given different labels: unreliable
            stats["sentences_with_conflicting_labels_dropped"] += 1
            continue
        # The agreement level is the strictest file the sentence appears in.
        agreement = next(level for level in AGREEMENT_LEVELS if sentence in members[level])
        rows.append({"sentence": sentence, "label": label, "agreement": agreement})
    return pd.DataFrame(rows), stats


def template_key(sentence):
    """Sentences that differ only in their numbers get the same key.

    PhraseBank contains templated sentences, for example 'Operating profit rose to
    EUR X mn from EUR Y mn' with different numbers. If one copy lands in training and
    a near-identical one in the test set, the test score is inflated. Grouping them
    keeps each template inside a single split.
    """
    key = re.sub(r"\d+(?:[.,]\d+)*", "#", sentence.lower())
    return re.sub(r"\s+", " ", key).strip()


def assign_split(df, seed):
    """70% train pool, 10% validation, 20% test, keeping the label mix similar in each
    and never splitting a group of near-identical sentences."""
    folds = np.empty(len(df), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    for k, (_, idx) in enumerate(splitter.split(df["sentence"], df["label_id"], groups=df["group"])):
        folds[idx] = k
    return np.select([folds < 2, folds == 2], ["test", "val"], default="train")


def summarise(df, stats):
    sizes = df.groupby("group").size()
    lines = ["Financial PhraseBank after cleaning", ""]
    lines += [f"{name.replace('_', ' ')}: {value}" for name, value in stats.items()]
    lines += [f"unique sentences kept: {len(df)}",
              f"groups of near-identical sentences (differ only in numbers): {(sizes > 1).sum()}, "
              f"covering {sizes[sizes > 1].sum()} sentences",
              "", "Sentences per split and label:",
              pd.crosstab(df["split"], df["label"], margins=True).to_string(),
              "", "Sentences per split and agreement level (% of annotators who agreed):",
              pd.crosstab(df["split"], df["agreement"], margins=True).to_string()]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Download, clean and split Financial PhraseBank.")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="decides which sentences go to train/val/test; keep the default")
    args = parser.parse_args()

    df, stats = read_phrasebank()
    unknown = set(df["label"]) - set(LABEL2ID)
    if unknown:
        raise ValueError(f"Unexpected labels in the data: {unknown}")
    df["label_id"] = df["label"].map(LABEL2ID)
    df["group"] = pd.factorize(df["sentence"].map(template_key))[0]
    df["split"] = assign_split(df, args.split_seed)
    df.insert(0, "id", np.arange(len(df)))

    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "phrasebank_clean.csv"), index=False)
    summary = summarise(df, stats)
    with open(os.path.join(args.out_dir, "split_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
