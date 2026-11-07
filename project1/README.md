# NLP1 Project 1: text classification on MultiNLI

Compares three representations and three architectures on MultiNLI, with cross-validation
grouped by premise and a Bayesian analysis with a region of practical equivalence (ROPE).

## Files

| Path | Contents |
|---|---|
| `nli/data.py` | Loading MNLI; tuning set; grouped and random folds |
| `nli/features.py` | BPE tokeniser; representations |
| `nli/models.py` | Log-linear, MLP and Siamese MLP; training loop |
| `nli/stats.py` | Reading results; correlated t posterior and ROPE |
| `run.py` | Declares and runs the experiments |
| `analyse.py` | Prints the result numbers on the slides |
| `checks.py` | Prints corpus facts, tokeniser refit and model weights |
| `figures.py` | Draws the slide figures |
| `test_pipeline.py` | Checks on splits, features, models and the posterior |
| `results/` | One JSON line per fit, plus `analysis.txt`, `checks.txt` and `rerun_comparison.txt` |
| `slides/slides.pdf` | Slides for the oral |
| `AI_USE.md` | Record of AI assistance |

## Reproducing

Needs [uv](https://docs.astral.sh/uv/) and, in practice, a GPU.

```bash
uv sync                  # Python 3.14 and the exact versions in uv.lock
uv run python test_pipeline.py
uv run python run.py     # all experiments; skips fits already in results/
uv run python checks.py > results/checks.txt
uv run python analyse.py > results/analysis.txt
uv run python figures.py
```

To rerun from scratch, delete `results/*.jsonl` first. To run one stage, use
`uv run python run.py main`. The stages are `tokeniser`, `representations`, `search`, `main`
and `controls`.

`results/probs/` holds the probabilities behind the validation-set numbers. It is about
30 MB and is not committed, so `analyse.py` skips those numbers on a fresh clone. To
regenerate them, delete `results/main.jsonl` and `results/controls.jsonl`, then rerun both stages.

## Protocol

- 10% of premises form a fixed tuning set, used for all settings and for early stopping.
- The other 90% are used for 5-fold CV, repeated 3 times, always grouped by `promptID`.
  Each repetition has its own seed, which sets both the folds and the initialisation.
- `validation_matched` and `validation_mismatched` are used only for the final report.
- Neural settings are shared across models: hidden 512, dropout 0.3, ReLU, lr 3e-4,
  weight decay 0.3, AdamW, batch 256, at most 20 epochs, patience 3.

## What repeats exactly

Splits and seeds are fixed, but results still differ slightly between runs, because GPU
computations are not fully deterministic. A full rerun from an empty results folder changed
single neural fits by up to 1.1 pp and single log-linear fits by up to 0.2 pp. Means of
15 fits changed by at most 0.08 pp, and no verdict at the ±1 pp ROPE changed. Details are in
`results/rerun_comparison.txt`. Compare new numbers with `results/analysis.txt` allowing for that.

`promptID` and premise text do not map one-to-one in a few cases: 98 premise texts have more
than one `promptID` (636 of 392,702 training rows). We kept `promptID` as the group.
