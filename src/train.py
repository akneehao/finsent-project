"""
Step 2: fine-tune one pretrained model on a labelled subset of a given size,
then score it. One call = one run = one row in the results file.

    python src/train.py --model bert --n_labels 200 --seed 0
    python src/train.py --model finbert --n_labels all --seed 3

Two settings are only changed for the extra experiments, each of which saves to
its own results folder:
    --min_steps 300       train for 300 updates instead of 100 at small budgets
    --min_agreement 100   draw the training sentences only from those every annotator agreed on

With --stage lrcheck only the validation set is scored and the row goes to
results/lr_check.csv. Use that to choose the learning rate, so the test set
stays untouched until the main runs.

Finished runs are skipped, so after a Colab disconnect just run the same
commands again.
"""
import argparse
import contextlib
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from transformers import BertModel

try:  # newer transformers versions may only ship one BERT tokenizer class
    from transformers import BertTokenizerFast as BertTokenizer
except ImportError:
    from transformers import BertTokenizer

from common import (append_row, already_done, load_split, parse_budget, sample_budget,
                    score, score_test)

# Both are BERT-base (12 layers, 768-wide, uncased). They differ in what they read
# during pretraining, and so also in their vocabularies.
MODELS = {
    "bert": "google-bert/bert-base-uncased",    # general English: Wikipedia and books
    "finbert": "yiyanghkust/finbert-pretrain",  # 10-K/10-Q filings, earnings calls, analyst reports
}
MAX_TOKENS = 128   # news sentences are short, so this rarely cuts anything off


class SentimentClassifier(nn.Module):
    """A pretrained encoder plus a small classification head.

    The head is identical for both models: average the token vectors, apply
    dropout, then one linear layer that outputs 3 scores. We skip BERT's own
    "pooler" layer because the finance checkpoint is saved as a masked-language
    model, which has none, and a trained pooler would give the general model an
    unfair head start.
    """

    def __init__(self, encoder_name, num_labels=3, dropout=0.1):
        super().__init__()
        self.encoder = BertModel.from_pretrained(encoder_name, add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        sentence_vector = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.head(self.dropout(sentence_vector))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast(device):
    """Half-precision maths on a GPU (about twice as fast); does nothing on a CPU."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def encode(tokenizer, sentences, device):
    enc = tokenizer(list(sentences), padding=True, truncation=True,
                    max_length=MAX_TOKENS, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def training_batches(train_df, batch_size, steps, rng):
    """Yield exactly `steps` batches, reshuffling the data at the start of every epoch."""
    done = 0
    while True:
        order = rng.permutation(len(train_df))
        for start in range(0, len(order), batch_size):
            yield train_df.iloc[order[start:start + batch_size]]
            done += 1
            if done == steps:
                return


def train(model, train_df, tokenizer, device, args, rng):
    """Fine-tune for a fixed number of updates and return that number.

    Small budgets get at least `min_steps` updates so the model has a chance to
    learn; larger ones get `epochs` passes over their data. The same rule applies
    to both models, and the final model is scored (no early stopping).
    """
    steps = max(args.min_steps, math.ceil(args.epochs * len(train_df) / args.batch_size))
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    warmup = max(1, int(0.1 * steps))       # ramp the learning rate up, then down to zero

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (steps - step) / max(1, steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for batch in training_batches(train_df, args.batch_size, steps, rng):
        ids, mask = encode(tokenizer, batch["sentence"], device)
        labels = torch.tensor(batch["label_id"].to_numpy(), device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device):
            logits = model(ids, mask)
        loss = loss_fn(logits.float(), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    return steps


@torch.no_grad()
def predict(model, df, tokenizer, device, batch_size=64):
    model.eval()
    preds = []
    for start in range(0, len(df), batch_size):
        ids, mask = encode(tokenizer, df["sentence"].iloc[start:start + batch_size], device)
        with autocast(device):
            preds.append(model(ids, mask).argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def save_predictions(test_df, pred_ids, args, budget_name):
    folder = os.path.join(args.results_dir, "predictions")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{args.model}_n{budget_name}_seed{args.seed}.csv")
    pd.DataFrame({"id": test_df["id"], "pred_id": pred_ids}).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune one model for one label budget and seed.")
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--n_labels", default="all", help="number of labelled training sentences, or 'all'")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--min_steps", type=int, default=100)
    parser.add_argument("--min_agreement", type=int, default=50,
                        help="only train on sentences that at least this %% of annotators agreed on (100 = unanimous)")
    parser.add_argument("--stage", choices=["main", "lrcheck"], default="main")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    budget = parse_budget(args.n_labels)
    budget_name = "all" if budget is None else str(budget)
    out_csv = os.path.join(args.results_dir, "results.csv" if args.stage == "main" else "lr_check.csv")
    run_name = f"{args.model} n={budget_name} seed={args.seed} lr={args.lr:g}"
    if args.min_agreement > 50:
        run_name += f" agreement>={args.min_agreement}%"
    if already_done(out_csv, args.model, budget_name, args.seed, args.lr):
        print(f"Skipping {run_name}: already in {out_csv}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: no GPU found, so this will be slow. In Colab: Runtime > Change runtime type > GPU.")

    train_pool, val_df, test_df = load_split(args.data_dir)
    # Only the training pool is filtered: validation and test keep every agreement level.
    train_pool = train_pool[train_pool["agreement"] >= args.min_agreement].reset_index(drop=True)
    train_df = sample_budget(train_pool, budget, args.seed)

    set_seed(args.seed)
    tokenizer = BertTokenizer.from_pretrained(MODELS[args.model])
    model = SentimentClassifier(MODELS[args.model]).to(device)

    start = time.time()
    steps = train(model, train_df, tokenizer, device, args, np.random.default_rng(args.seed))
    seconds = time.time() - start

    row = {"model": args.model, "budget": budget_name, "n_train": len(train_df),
           "seed": args.seed, "lr": args.lr, "steps": steps}
    row.update(score(val_df, predict(model, val_df, tokenizer, device), "val"))
    if args.stage == "main":
        test_pred = predict(model, test_df, tokenizer, device)
        row.update(score_test(test_df, test_pred))
        save_predictions(test_df, test_pred, args, budget_name)
    row.update({"train_seconds": round(seconds, 1),
                "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "finished_at": datetime.now().isoformat(timespec="seconds")})
    append_row(out_csv, row)

    message = f"{run_name}: val macro-F1 {row['val_macro_f1']:.3f}"
    if args.stage == "main":
        message += f", test macro-F1 {row['test_macro_f1']:.3f}"
    print(f"{message} ({seconds:.0f}s, {steps} steps)")


if __name__ == "__main__":
    main()
