# AI assistance

The course asks us to record what we delegate to assistants and to disclose it in the blog
post. We used Claude Code (Anthropic) as the assistant for the work below.

| Delegated to the assistant                                                   | Group input                                                                              | How it was checked                                                                                                                                         |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Code: data splits, features, models, experiment runner, analysis and figures | Not yet reviewed line by line by the group                                                                               | `test_pipeline.py`: no premise crosses splits, sparse and dense forward passes agree, shuffled labels stay at chance, posterior statistics are as expected |
| Running the experiments, including the hyperparameter search                 | A group member questioned the limits of the first results, which led to the wider search | Tests and stage definitions checked against the stored results; full rerun from an empty results folder not finished yet                                   |
| Testing whether overlap alone is enough                                      | Proposed by a group member                                                               | Ablation in `results/analysis.txt`                                                                                                                         |
| Siamese MLP and per-sentence blocks                                          | Discussed with our TA                                                                    | Same comparisons as the other systems                                                                                                                      |
| Drafts of the slides                                                         | A group member reviewed them and requested corrections; not yet reviewed by the other members        | Numbers traced to `analysis.txt` and `checks.txt`; citations compared with the papers by the assistant                                                     |
| Simplifying the repository for the final version                             | Requested and scoped by a group member                                                   | `test_pipeline.py` passes on a clean install; every planned fit is present in `results/`                                                                   |

We remain responsible for all of this work, and each member has to be able to explain it.
