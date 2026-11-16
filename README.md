# Financial sentiment with few labels

SC4001 Neural Networks and Deep Learning, group project.

**Research question:** when only a few hundred labelled sentences are available, how much does it help to start from a language model that has already read a lot of financial text? Do models make their mistakes on the same sentences that human annotators disagreed about? And for a fixed number of labels, does it help to train only on sentences that every annotator agreed on?

We fine-tune two BERT-base models that are identical in size and are given the same classification head, the same training sentences and the same training schedule. They differ only in what they read before we got them:

| Name in the code | Model | Pretrained on |
|---|---|---|
| `bert` | `google-bert/bert-base-uncased` | General English (Wikipedia and books) |
| `finbert` | `yiyanghkust/finbert-pretrain` | 4.9B tokens of 10-K/10-Q filings, earnings call transcripts and analyst reports |

Each model is trained with 50, 100, 200, 400, 800 and all available labelled sentences, ten random seeds each. A TF-IDF + logistic regression model is the non-neural reference. Two extra experiments reuse the same setup:

- **Longer training:** the 50–200 label runs again with 300 updates instead of 100, to check that FinBERT's lead isn't only faster learning.
- **Unanimous labels only:** the 50–800 label runs again, with the training sentences drawn only from those every annotator agreed on.

## Data

[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) (Malo et al., 2014): 4,846 sentences from financial news in the version we use (at least 50% annotator agreement), each labelled positive, negative or neutral by 5–8 annotators with finance training. The dataset records how many annotators agreed, which we use in the analysis. Licence: CC BY-NC-SA 3.0 (non-commercial). The data isn't included in this folder; `prepare_data.py` downloads it (about 0.7 MB).

Cleaning and splitting (`src/prepare_data.py`):

- Exact duplicate sentences are removed, and so are sentences that appear with conflicting labels.
- Sentences that differ only in their numbers are grouped and kept in the same split, so a near-copy of a test sentence can't appear in training.
- The split is 70% training pool, 10% validation and 20% test, with a similar label mix in each. It is made once (seed 42) and shared by every experiment.

## How to run (Google Colab, recommended)

1. Open `run_in_colab.ipynb` in Colab (**File → Upload notebook**) and choose **Runtime → Change runtime type → T4 GPU**.
2. Choose **Runtime → Run all**. The first cell asks you to upload `finsent-project.zip` (this folder, zipped) and the second your latest `finsent-progress.zip` (set `RESTORE = False` in that cell on the very first run).
3. Each training step ends by downloading `finsent-progress.zip` with the data split, results and figures. Keep the newest one.

Without Google Drive, Colab forgets everything when it disconnects. To carry on, run all cells again and upload the project zip and your latest progress zip: finished runs are skipped. If you'd rather keep everything in Google Drive, set `USE_DRIVE = True` in the first cell and put this folder at `MyDrive/finsent-project`.

## How to run (command line)

Needs Python 3.10+ and, realistically, a GPU.

```bash
pip install -r requirements.txt

# 1. Download, clean and split the data (once)
python src/prepare_data.py

# 2. Choose the learning rate on the validation set only
python src/train.py --stage lrcheck --model bert --lr 2e-5 --n_labels 200 --seed 0
#    ...repeat for each model, lr in {2e-5, 5e-5} and n_labels in {200, all}, then:
python src/analyze.py --lr_check

# 3. Main runs: every model x budget x seed (120 runs in total), for example
python src/train.py --model bert    --lr 5e-5 --n_labels 50 --seed 0 --results_dir results_main
python src/train.py --model finbert --lr 5e-5 --n_labels 50 --seed 0 --results_dir results_main

# 4. TF-IDF baseline and the leakage check
python src/baselines.py --results_dir results_main

# 5. Longer training (budgets 50-200) and unanimous labels only (budgets 50-800), for example
python src/train.py --model bert --lr 5e-5 --n_labels 200 --seed 0 --min_steps 300 --results_dir results_longer
python src/train.py --model bert --lr 5e-5 --n_labels 200 --seed 0 --min_agreement 100 --results_dir results_clean

# 6. Tables and figures
python src/analyze.py --results_dir results_main --figures_dir figures_main \
    --longer_dir results_longer --clean_dir results_clean
```

## What each file does

| File | Purpose |
|---|---|
| `src/prepare_data.py` | Downloads Financial PhraseBank, removes duplicates, records annotator agreement, makes the fixed split |
| `src/common.py` | Shared helpers: labels, loading the split, choosing labelled subsets, scoring, saving results |
| `src/train.py` | Fine-tunes one model for one label budget and seed, then scores it |
| `src/baselines.py` | TF-IDF + logistic regression baseline, and the ProsusAI/finbert leakage check |
| `src/analyze.py` | Summary tables, paired comparisons, figures and the error-analysis list |
| `run_in_colab.ipynb` | Runs everything in order on Google Colab |

## Outputs

| File | Contents |
|---|---|
| `data/split_summary.txt` | Dataset counts after cleaning, per split, label and agreement level |
| `results/lr_check.csv` | Validation scores from the learning-rate check |
| `results/results.csv` | One row per run: settings, validation and test scores, scores by agreement level |
| `results/predictions/` | Test predictions of every run, used for the error analysis |
| `results/summary_table.csv` | Mean and standard deviation over seeds for each model and budget |
| `results/paired_comparison.csv` | FinBERT minus BERT for each budget, compared seed by seed, with 95% confidence intervals |
| `results/labels_saved.csv` | Rough estimate of how many labels FinBERT's pretraining is worth |
| `results/all_tests.csv` | Every significance test in one table, with Holm-adjusted p-values (corrected for running 24 tests) |
| `results/stability_check.csv` | Spread between seeds and collapsed runs at 50–200 labels in each experiment, with tests of the spread |
| `results/seed_groups.csv` | FinBERT's lead in seeds 0–4 and seeds 5–9, with the Transformers version of each group |
| `results/predicted_classes.csv` | Accuracy, share of each label predicted and recall, per agreement level |
| `results/error_sources.csv` | How many mistakes come from sentences with under 75% annotator agreement |
| `results/error_analysis_n200.csv`, `..._nall.csv` | Test sentences where one model is consistently right and the other wrong, at 200 and at all labels, with the label each model usually picks |
| `results/leakage_check.csv` | ProsusAI/finbert's test score with no training on our data |
| `figures/*.png` | Learning curves, accuracy by annotator agreement, confusion matrices at 200 and at all labels |
| `results_main/`, `figures_main/` | The same outputs for the main runs, which use the learning rate chosen on validation (5e-5) and 10 seeds. `results/` then holds the first 2e-5 runs (5 seeds), kept as a robustness check |
| `results_longer/steps_check.csv` | FinBERT and BERT at 50–200 labels with 100 vs 300 updates, on the same seeds |
| `results_longer/updates_effect.csv` | How much the extra updates change each model on its own |
| `results_clean/clean_check.csv` | Training on unanimous sentences only vs a random sample of the same size, overall and by agreement level |
| `figures_main/clean_vs_random.png` | The same comparison as a figure |
| `requirements-colab.txt` | Exact package versions used in Colab |

## Design decisions to mention in the report

- **Fair comparison.** Both models are BERT-base and get the same head (the average of the token vectors, then dropout, then one linear layer), the same sentences for a given seed, and the same number of updates. BERT's own pooler layer isn't used, because the finance checkpoint is saved without one.
- **Fixed training schedule, no early stopping.** Each run makes max(100, 5 epochs) updates, and the final model is scored. In a real few-label setting there wouldn't be a large labelled validation set to pick checkpoints with. Here the validation set is used once, to choose the learning rate.
- **A fair baseline.** TF-IDF + logistic regression picks its regularisation strength C on the validation set for every run. With a fixed default it predicts "neutral" for everything at small budgets, which would make BERT look better than it really is.
- **Paired seeds.** Seed *k* draws the same labelled sentences for both models, so differences can be compared run by run (`paired_comparison.csv`).
- **Many tests.** With 24 tests, one or two p < 0.05 are expected by luck alone, so `all_tests.csv` also gives Holm-adjusted p-values. Only those below 0.05 are reported as significant.
- **10 seeds.** The experiment started with 5 seeds per setting. Scores varied a lot between seeds at small budgets, and 5 pairs give a test little power (a sign test can't reach p < 0.05 even if one model wins all 5), so every setting was extended to 10 seeds. All 10 are reported.
- **Longer-training check.** At 50–200 labels every run gets 100 updates, and general BERT sometimes barely learns the negative class in that time. The same runs with 300 updates test whether FinBERT's lead only reflects faster learning.
- **Unanimous labels only.** These runs draw their training sentences from the unanimously labelled part of the training pool (1,603 of 3,384 sentences, with a similar label mix), while validation and test keep every agreement level. The unanimous and random runs see different sentences, so they're compared with Welch's t-test rather than seed by seed. Note that unanimous sentences are also easier ones, so this tests "clear-cut examples only", not "less label noise" alone.
- **Leakage check.** The widely used `ProsusAI/finbert` was fine-tuned on Financial PhraseBank itself, so it isn't a fair comparison model. `baselines.py` scores it on our test set with no training to show how much that inflates results.
- **Stability checks.** The claim that FinBERT is steadier isn't one of the 24 tests, so `stability_check.csv` tests it separately. BERT's larger spread at 200 labels comes from two weak runs, and the Brown-Forsythe test, which outliers don't fool, gives p = 0.28. The pattern repeats across experiments but isn't proven.
- **Library versions.** Seeds 0–4 of the main runs used Transformers 5.21.0, and seeds 5–9 and both extra experiments used 5.22.0 (`requirements-colab.txt` records only the later one). The gap between the two seed groups also appears in the longer-training runs, which all used 5.22.0, so the version isn't the cause (`seed_groups.csv`). Repeating a run on the same version gives the same score.
- **Known limitation.** The two models differ in their vocabulary as well as their pretraining text, so the experiment can't separate those two effects. The checkpoint's 30,873-word vocabulary matches the version Yang et al. (2020) trained from scratch, so it doesn't appear to continue from general BERT.

## Troubleshooting

- **"Some weights ... were not used" or "newly initialized" warnings** are expected: we drop the pretraining heads and add our own classifier.
- **CUDA out of memory:** add `--batch_size 8`. Use the same batch size for every run, or the comparison stops being fair.
- **An error about loading `pytorch_model.bin`** usually means PyTorch is older than 2.6. Run `pip install -U torch`.
- **Very slow training:** check that the GPU is on (`nvidia-smi` should list one).

## Note on AI use

The code in this folder was drafted with help from Claude (Anthropic). The team is responsible for running and checking it, and must describe that in the report's AI Use Statement.
