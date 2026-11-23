# NLP1 Project 2: pretrained embeddings, embed-and-pool and recurrent encoders on MultiNLI

Slides: [`slides/slides.pdf`](slides/slides.pdf), one per question. AI assistance: [`AI_USE.md`](AI_USE.md).

The protocol is project 1's, so the two projects can be compared fold by fold:
- the same 10% tuning set, carved by premise, for every choice and for early stopping;
- the same premise-grouped 5-fold cross-validation, repeated 3 times (seeds 0, 1, 2);
- the same Bayesian correlated t-test with a region of practical equivalence of ±1 pp;
- `validation_matched` and `validation_mismatched` used only for the final report and Q6.

New in this project: every setting on the tuning set is fitted with **3 seeds**, and
differences are paired by seed, so each search result comes with its own noise.

## Files

| Path | Contents |
|---|---|
| `nli/data.py` | Loading MNLI, tuning set and grouped folds (as in project 1); subsampling premises |
| `nli/text.py` | Tokenisers (NLTK Treebank, MNLI parse tokens, whitespace, tutorial 2's bag of words); case lookup |
| `nli/embeddings.py` | Readers for every pretrained table (GloVe text, word2vec binary, ...) |
| `nli/inputs.py` | Cached token ids and embedding matrices for each tokeniser, table and case choice |
| `nli/models.py` | Embedder, embed-and-pool encoder, LSTM/GRU encoder, pair classifier |
| `nli/train.py` | AdamW training with early stopping on the tuning loss |
| `nli/stats.py` | Correlated t posterior and ROPE (project 1) |
| `nli/features.py`, `nli/bda.py` | Q6: pair features and tutorial 2's Bayesian regression |
| `run.py` | Declares all experiments as stages and runs them |
| `explore.py` | Q1 coverage and Q2 embedding analysis |
| `analyse.py` | Prints every result number for the slides |
| `errors.py` | Q6 error analysis |
| `figures.py` | Figures for the slides (`results/figures/`) |
| `test_pipeline.py` | Checks on tokenisers, readers, encoders, training, posterior and splits |
| `slides/` | Slide deck (LaTeX source and PDF), one slide per question |
| `results/` | `q1_coverage.txt`, `q2_embeddings.txt`, `lr_pilot.txt`, `search.txt`, `analysis.txt` (final systems, data size), `errors.txt` (Q6) and `errors_parallel.txt` (the same regression with parallel chains, a check); `fits/`: one JSON per fit (all 462, with per-epoch history), from which `analyse.py` and `figures.py` recompute every number; `fits_cap20/`: the 51 fits of the first run with a 20-epoch cap; `errors_posterior.npz`, `errors_items.csv.gz`: Q6 posterior samples and per-pair features |

**Where each slide's numbers come from.** Q1: `results/q1_coverage.txt`, `results/search.txt`
(tables, tokens, step 2). Q2: `results/q2_embeddings.txt`. Q3, Q4 and Q5: `results/search.txt`,
`results/lr_pilot.txt`, and `results/analysis.txt` (CV, correlated t-tests, data size). Q6:
`results/errors.txt`.

## Design and why

**Tokens and table (Q1).** Ten pretrained tables were compared: GloVe 2024 50d (the
tutorial's), GloVe 6B 50/100/200/300d, GloVe 840B (cased, Common Crawl), fastText crawl
300d (cased), word2vec GoogleNews, counter-fitted vectors (antonyms pushed apart) and
ConceptNet Numberbatch, plus 300d vectors trained from scratch. A token's vector is looked
up as written, then lowercased, then capitalised. The model's vocabulary is the MNLI types
the table covers, and unknown words get the mean vector, as in the tutorial. Coverage with
NLTK tokens: GloVe 840B 99.6%, fastText 98.9% (it has no `n't`, so it loses the negation of
37,814 "don't/didn't" tokens), word2vec 88.1% (no punctuation, no "a/of/and/to").
Tutorial 2's bag of words removes NLTK stopwords, which include *not, no, all, some*: 876
hypotheses become empty. Details: `results/q1_coverage.txt`.

**What the tables capture (Q2).** In every distributional table, antonyms are closer than
synonyms: the chance that a synonym pair is closer than an antonym pair is 0.23 to 0.35
(0.5 would be no separation). The nearest neighbour of *always* in GloVe 840B is *never*.
That matters for contradiction: 64.6% of training pairs with a WordNet antonym pair across
the sentences are contradictions. Counter-fitted vectors fix it (0.96) and, together with
Numberbatch, align hypothesis words with premise words best for entailment. Details:
`results/q2_embeddings.txt`.

**Same scale for every table.** Counter-fitted and Numberbatch vectors have norm 1, about
seven times smaller per coordinate than GloVe. Each table is multiplied by one number so
that its coordinates have SD 0.4 (cosines unchanged); otherwise frozen tables would differ
in scale, not only in content. The original scales are run as a control.

**Classifier.** Each sentence is encoded separately, and the pair is classified from
[u; v; |u−v|; u⊙v] by an MLP (512 ReLU units, dropout 0.3). These are the comparison
features of project 1's Siamese MLP, whose encoder was itself an embed-and-pool over learned
BPE embeddings. Optionally, project 1's overlap block is added.

**Encoders.** Embed-and-pool (mean, sum, max, mean+max) and LSTM/GRU (bi- or
unidirectional, 1–2 layers, last state or max/mean over states, word dropout).

**Training.** AdamW, lr 3e-4, weight decay 0.3 (not on embedding tables), batch 256,
gradient clipping 5, early stopping on the tuning loss with patience 3, at most 60 epochs.

## Stages (`run.py`)

| Stage | Question | What it compares | Fits |
|---|---|---|---|
| `lr` | Q5 | Learning rate of each encoder before any search | 18 |
| `embeddings` | Q1–Q3 | Every table in embed-and-pool, the 300d ones in the BiLSTM; original-scale control | 66 |
| `tokens` | Q1 | Tokeniser and case for GloVe 840B and fastText | 36 |
| `pool` | Q3 | One change at a time around the embed-and-pool base, plus a grid of lr × weight decay | 87 |
| `rnn` | Q4 | One change at a time around the BiLSTM base, including tutorial 2's LSTM | 66 |
| `confirm` | Q3–Q5 | Step 2 around the new bases: table, "n't" tokenisation, regularisation, combined changes | 72 |
| `main` | Q3–Q4, Q6 | 5-fold CV × 3 of the final systems and their controls (random embeddings, hypothesis only) | 90 |
| `datasize` | Q4 | 1, 3, 10, 30 and 100% of the training premises, pretrained against random | 60 |

## How the search was corrected (Q5)

- **Epoch cap.** A first run capped training at 20 epochs. Many BiLSTM fits had their
  best epoch at 18–19 and were still improving (0.5–1 pp from epoch 15 to 20), so the
  comparison would have measured speed of convergence. The 51 fits were set aside, the cap
  raised to 60, and every fit records whether it reached the cap (none has so far).
- **Learning rate.** A pilot on the tuning set (3 seeds) picked 3e-4 for both encoders:

  | lr | BiLSTM | embed-and-pool |
  |---|---|---|
  | 3e-4 | 71.6 ± 0.1 | 59.6 ± 0.2 |
  | 1e-3 | 70.4 ± 0.4 | 57.8 ± 0.1 |
  | 2e-3 / 3e-3 | 68.8 ± 0.2 | 56.3 ± 0.1 |

  Training accuracy falls with the higher rates too: with AdamW the decay per step is
  lr × weight decay, so the `pool` grid varies both.

## Search results (tuning set, 3 seeds; full tables in `results/search.txt`)

Accuracy in %, and the difference to the base setting paired by seed (SD over seeds in
brackets). The median SD of one setting over seeds is 0.17 pp.

**Tables (Q1–Q3).** Embed-and-pool, frozen: GloVe 50d about 50%; GloVe 840B 59.6;
fastText +1.2; Numberbatch +1.7; random vectors learnt from scratch +3.5 (frozen tables
underfit: train ≈ tune ≈ 60%). BiLSTM: GloVe 840B 71.6; fastText +0.9; Numberbatch
+1.4 (0.4); random −4.4. At their original scale, counter-fitted loses 1.7 pp, Numberbatch 1.6, word2vec 0.5 (frozen pool).

**Tokens (Q1).** NLTK and the parse tokens tie (within 0.4 pp); whitespace loses 1.4–2.4 pp,
tutorial 2's stopword-filtered bag of words 5–5.5 pp.

**Embed-and-pool (Q3).** Fine-tuning the table +5.1 (0.3), a 512d projection +3.9, both
+5.8 (0.1); dropout 0.1 +1.7; weight decay 0.01 +1.8; max pooling −7.4; tanh −4.2;
concatenating u and v instead of comparing them −1.2; project 1's overlap block −0.6;
tutorial 2's settings −10.6.

**BiLSTM (Q4).** Weight decay 0.1 +0.7 (0.06); 512 units +0.6; 2 layers +0.5; GRU +0.45
(0.5); last state −2.5 and mean −2.0 instead of max; fine-tuning the table −1.5 (train
78.8%: it overfits); concatenating −3.0; overlap block −1.3; tutorial 2's LSTM −6.3.

**Step 2 (`confirm`).** Effects found one at a time need not add up, and fine-tuning turns
embed-and-pool from underfitting to overfitting, so table, tokens and regularisation were
searched again around the new bases. Embed-and-pool (base 65.4): fastText with "n't"
spelt out +0.46 (0.09), whereas fastText with plain NLTK tokens −0.18: coherent tokens are
worth about 0.6 pp; every other change below 0.3 pp. BiLSTM with Numberbatch (base 73.0):
weight decay 0.1 +0.35 (73.3 ± 0.08); adding 512 units and 2 layers 73.4 ± 0.3, a tie.

**Final systems** (`run.py`, `MAIN`): embed-and-pool = fine-tuned fastText (`nltk-not`
tokens), 512d projection, mean pooling; BiLSTM = frozen Numberbatch, 256 units per
direction, max over time, weight decay 0.1. Each is run with a table learnt from scratch
and hypothesis-only as controls.

## Final results (5-fold CV × 3 on project 1's folds; `results/analysis.txt`)

| System | CV accuracy (%) | matched / mismatched |
|---|---|---|
| BiLSTM, frozen Numberbatch | 72.41 ± 0.21 | 71.9 / 72.9 |
| BiLSTM, table learnt from scratch | 66.35 ± 0.26 | 65.7 / 65.8 |
| BiLSTM, hypothesis only | 56.95 ± 0.26 | 56.2 / 56.3 |
| Embed-and-pool, fine-tuned fastText | 65.17 ± 0.21 | 65.2 / 65.7 |
| Embed-and-pool, table learnt from scratch | 62.24 ± 0.29 | 61.8 / 61.9 |
| Embed-and-pool, hypothesis only | 54.42 ± 0.19 | 54.0 / 54.1 |
| Project 1's best (Siamese MLP, BPE + overlap) | 64.85 ± 0.21 | 64.4 / 65.0 |

Validation columns: mean of the 15 fits. Correlated t-test, ROPE ±1 pp: the BiLSTM beats
embed-and-pool by 7.2 pp and project 1 by 7.6 pp (P = 1.00 each); embed-and-pool ties
project 1 (+0.3 pp, P(tie) = 1.00). Pretrained tables beat tables learnt from scratch by
2.9 pp (pool) and 6.1 pp (BiLSTM). No fit reached the epoch cap.

**Data size (Q4).** Fold 0 of each repetition, with 1–100% of the training premises: the
gain from pretraining is +3 to +4 pp for embed-and-pool at every size, and +6 to +8 pp for
the BiLSTM (largest at 1%, +8.1). All four curves still rise linearly in log(data).

**Error analysis (Q6, `results/errors.txt`).** Tutorial 2's Bayesian regression of a
correct fit on 17 pair features, Binomial over the 15 CV fits of each validation pair
(max r_hat 1.003, 0 divergences). Pretrained tables rely less than scratch tables on
overlap, the negation cue and rare words; words without a vector hurt the word-level models
but not project 1's BPE model; antonym pairs hurt every design except the BiLSTM on
Numberbatch; long premises hurt the BiLSTM as much as pooling; pairs solvable from the
hypothesis alone predict correctness most for every design; the BiLSTM's gain over
project 1 is smallest on neutral pairs (+3.8 vs +8.8 to +10.1 pp).

## Reproducing

Needs [uv](https://docs.astral.sh/uv/) and a GPU. The embedding files are downloaded into
the folder `EMB_DIR` in `nli/embeddings.py` (sources in its `SOURCES` table).

```bash
uv sync
uv run python test_pipeline.py
uv run python explore.py q1 > results/q1_coverage.txt
uv run python explore.py q2 > results/q2_embeddings.txt
uv run python run.py lr embeddings tokens pool rnn    # then confirm, then main datasize
uv run python analyse.py PATH/TO/project1/results lr embeddings tokens pool rnn confirm > results/search.txt
uv run python analyse.py PATH/TO/project1/results main datasize > results/analysis.txt
uv run python errors.py PATH/TO/project1/results/probs > results/errors.txt
uv run python figures.py PATH/TO/project1/results
```

`analyse.py` and `figures.py` only need `results/fits/`, so the numbers can be checked without
retraining. The per-pair predicted probabilities (`results/probs/`, 50 MB) are not in the
repository; `errors.py` needs them and project 1's, and `run.py main` writes them again.

Several copies of `run.py` can run at once: each claims a fit with a lock file, and
finished fits are skipped. We ran 6 on one A40 GPU (2.95 times the throughput of one).
