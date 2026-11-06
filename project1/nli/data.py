"""MultiNLI loading and premise-grouped splits.

Every MNLI premise was shown to annotators who wrote one hypothesis per label, so a
premise (`promptID`) appears with all three labels. A random row-level split puts
siblings of the same premise on both sides of the split; all splits here are by premise.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasets import load_dataset

LABELS = ("entailment", "neutral", "contradiction")
COLUMNS = ["promptID", "premise", "hypothesis", "genre", "label"]


def load_mnli() -> dict[str, pd.DataFrame]:
    """Return train / validation_matched / validation_mismatched as DataFrames."""
    ds = load_dataset("nyu-mll/multi_nli")
    out = {}
    for split in ("train", "validation_matched", "validation_mismatched"):
        df = ds[split].select_columns(COLUMNS).to_pandas()
        out[split] = df[df.label >= 0].reset_index(drop=True)
    return out


@dataclass
class Fold:
    train: np.ndarray
    val: np.ndarray


def carve_tuning_set(groups: np.ndarray, frac: float = 0.10, seed: int = 2026):
    """Reserve a fraction of premises for hyperparameters and early stopping.

    These premises never enter cross-validation, so model selection cannot
    leak into the fold scores we compare. Returns (pool_idx, tune_idx).
    """
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    tune_groups = rng.choice(uniq, size=int(len(uniq) * frac), replace=False)
    mask = np.isin(groups, tune_groups)
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def grouped_kfold(groups: np.ndarray, k: int, seed: int) -> list[Fold]:
    """k-fold over premises: no premise is ever on both sides of a split."""
    uniq = np.unique(groups)
    parts = np.array_split(np.random.default_rng(seed).permutation(uniq), k)
    folds = []
    for part in parts:
        mask = np.isin(groups, part)
        folds.append(Fold(np.flatnonzero(~mask), np.flatnonzero(mask)))
    return folds


def random_kfold(n: int, k: int, seed: int) -> list[Fold]:
    """Row-level k-fold (the tutorial's scheme). Kept only to measure leakage."""
    parts = np.array_split(np.random.default_rng(seed).permutation(n), k)
    return [
        Fold(np.setdiff1d(np.arange(n), part, assume_unique=True), np.sort(part))
        for part in parts
    ]
