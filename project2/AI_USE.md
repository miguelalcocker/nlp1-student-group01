# AI assistance

The course asks us to record what we delegate to assistants and to disclose it in the blog
post. We used Claude Code (Anthropic) as the assistant for the work below.

| Delegated to the assistant | Group input | How it was checked |
|---|---|---|
| Code: tokenisers, embedding readers, encoders, training, experiment runner, analysis | Scope and rigour requirements set by a group member; reviewed by the group | `test_pipeline.py`: padding invariance, pooling order-blind and RNN order-sensitive, hypothesis-only ignores the premise, a planted cue is learnt, shuffled labels stay at chance, splits as in project 1 |
| Choosing and downloading the pretrained tables | The course's resource list, shared by a group member | Checksums of the files; coverage in `results/q1_coverage.txt` |
| Q1 coverage and Q2 embedding analysis | Reviewed by the group | Outputs in `results/` |
| Running the experiments on the GPU server, and correcting the first run (epoch cap, learning-rate pilot) | Reviewed by the group | Cap-20 fits kept for comparison; every fit records whether it reached the cap |
| This README | Reviewed by the group | Numbers copied from the files in `results/` |
| Final runs (`main`, `datasize`), `analyse.py` output, Q6 regression (`errors.py`) | Reviewed by the group; a group member asked to keep the slower sequential MCMC run as the source of the numbers, with a parallel run only as a check | `results/analysis.txt`, `results/errors.txt`; MCMC diagnostics (r_hat, divergences) printed in `errors.txt`; `test_pipeline.py` rerun after the final runs |
| Figures (`figures.py`) and the first draft of the slide deck (`slides/`) | Format of project 1's deck; reviewed by the group | Every number on the slides checked against the files in `results/` |
| A study guide for the oral (kept outside this repository) | Requested by a group member; reviewed by the group | Numbers copied from the files in `results/`; parameter counts read from the fits |
| Two revisions of the deck after critical reviews by a group member (2026-11-23): Q5 search diagram redrawn to the actual stage order, close-call table put on one base, ± SD added to every table, one name per system, texts rewritten as reasoning, captions checked (one wrong number corrected) | Review questions and the decision to revise from a group member | Each changed number re-read from `results/search.txt` and `results/analysis.txt` |

We remain responsible for all of this work, and each member has to be able to explain it.
