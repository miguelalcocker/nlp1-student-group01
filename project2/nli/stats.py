"""Reading results and the Bayesian comparison shared by analyse.py and figures.py."""
import json
from pathlib import Path

import baycomp
import numpy as np
import pandas as pd
import scipy.stats as st

RESULTS = Path("results")
ROPE, RUNS = 0.01, 3     # 1 percentage point; 3 repetitions of 5-fold CV


def load(stage: str, split: str | None = None) -> pd.DataFrame:
    rows = []
    for line in (RESULTS / f"{stage}.jsonl").open():
        r = json.loads(line)
        job = r.pop("job")
        rows.append({**job, **r, "vocab": job["tok"]["vocab"], "kind": job["tok"]["kind"],
                     "lowercase": job["tok"]["lowercase"], "params": json.dumps(job["params"], sort_keys=True)})
    df = pd.DataFrame(rows)
    return df if split is None else df[df.split == split]


def scores(df: pd.DataFrame, rep: str, arch: str) -> np.ndarray:
    """Accuracy per fit, ordered by (seed, fold) so that two systems are paired."""
    return df[(df.rep == rep) & (df.arch == arch)].sort_values(["seed", "fold"]).accuracy.to_numpy()


def posterior(a: np.ndarray, b: np.ndarray):
    """Correlated t posterior over b - a (Benavoli et al., 2017)."""
    mean, var, df = baycomp.CorrelatedTTest.compute_statistics(a, b, RUNS)
    return st.t(df, loc=mean, scale=np.sqrt(var))


def regions(t, rope=ROPE):
    """P(a wins), P(tie), P(b wins)."""
    low, high = t.cdf(-rope), t.cdf(rope)
    return low, high - low, 1 - high


def verdict(t, rope=ROPE, level=0.95):
    a, tie, b = regions(t, rope)
    return "A" if a > level else "B" if b > level else "tie" if tie > level else "undecided"
