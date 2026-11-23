"""Q6: which characteristics of a pair predict that one design gets it right and another not.

Usage: python errors.py PROJECT1_PROBS > results/errors.txt   (PROJECT1_PROBS: project 1's results/probs)

Items are the pairs of validation_matched and validation_mismatched, used here after every
choice was frozen (nothing is selected on them). For each system we count, per pair, how many
of its 15 fits (5 folds x 3 repetitions) predict the gold label. Project 1's best system
(Siamese MLP on BoW + overlap) is included from its saved probabilities.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import numpyro
import pandas as pd

numpyro.set_host_device_count(4)   # lets BDA_CHAINS=parallel run the 4 chains at once; must precede jax use
import run
from nli import bda, inputs
from nli.data import carve_tuning_set, load_mnli
from nli.embeddings import SOURCES
from nli.features import pair_features, wordnet_antonyms
from nli.text import map_to_rows

CACHE = Path("cache")
SETS = ("validation_matched", "validation_mismatched")
# Alignment uses Numberbatch, the table whose alignment separated the labels best (Q2) and
# the BiLSTM's table. Uncovered tokens are counted for fastText with "n't" spelt out (the
# embed-and-pool input): its gaps are unknown words, whereas Numberbatch's are mostly
# punctuation, and with plain NLTK tokens every "n't" would count and duplicate negation.
ALIGN_TABLE, OOV_TABLE, OOV_TOK = "numberbatch", "fasttext", "nltk-not"


def correct_counts(data, keys) -> np.ndarray:
    y = np.concatenate([data[s].label.to_numpy() for s in SETS])
    k = np.zeros(len(y), dtype=int)
    for key in keys:
        z = np.load(run.RESULTS / "probs" / f"{key}.npz")
        pred = np.concatenate([z[s].astype(np.float32).argmax(1) for s in SETS])
        k += pred == y
    return k


def mean_confidence(keys) -> np.ndarray:
    """Probability of the predicted label, averaged over fits: a model-specific feature."""
    conf = [np.concatenate([np.load(run.RESULTS / "probs" / f"{key}.npz")[s].astype(np.float32).max(1) for s in SETS])
            for key in keys]
    return np.mean(conf, 0)


def project1_counts(data, folder: Path) -> np.ndarray:
    y = np.concatenate([data[s].label.to_numpy() for s in SETS])
    files = sorted(folder.glob("overlap__siamese__s*__f*.npz"))
    assert len(files) == 15, files
    k = np.zeros(len(y), dtype=int)
    for f in files:
        z = np.load(f)
        k += np.concatenate([z["matched"].argmax(1), z["mismatched"].argmax(1)]) == y
    return k


def features(data) -> pd.DataFrame:
    pool, _ = carve_tuning_set(data["train"].promptID.to_numpy())
    toks = inputs.tokens("nltk", data, CACHE)
    tr = toks["train"]
    counts = Counter()
    for i in np.unique(tr["pair_prem"][pool]):
        counts.update(t.lower() for t in tr["prem"][i])
    for i in pool:
        counts.update(t.lower() for t in tr["hyp"][i])
    log_freq = {w: float(np.log1p(c)) for w, c in counts.items() if w.isalpha()}
    tab = inputs.table(ALIGN_TABLE, data, CACHE)
    oov_tab = inputs.table(OOV_TABLE, data, CACHE)
    oov_toks = inputs.tokens(OOV_TOK, data, CACHE)
    unit = {}
    for w, r in zip(tab.words, range(len(tab.words))):
        lw = w.lower()
        if lw not in unit and lw.isalpha():
            v = tab.vectors[r]
            unit[lw] = v / (np.linalg.norm(v) + 1e-12)
    rows = []
    for s in SETS:
        st, df = toks[s], data[s]
        types = {t for x in st["prem"] + st["hyp"] for t in x}
        ot = oov_toks[s]
        covered = set(map_to_rows({t for x in ot["prem"] + ot["hyp"] for t in x}, oov_tab.index,
                                  "exact" if SOURCES[OOV_TABLE]["cased"] else "lower"))
        words = {t.lower() for t in types if t.isalpha()}
        antonyms = wordnet_antonyms(words)
        pp = st["pair_prem"]
        prem_parse = df.drop_duplicates("premise").premise_binary_parse.to_numpy()
        for i in range(len(df)):
            f = pair_features(st["prem"][pp[i]], st["hyp"][i], prem_parse[pp[i]], df.hypothesis_binary_parse.iat[i],
                              set(), unit, log_freq, antonyms)
            both = list(ot["prem"][pp[i]]) + list(ot["hyp"][i])
            f["oov"] = float(np.mean([t not in covered for t in both])) if both else 0.0
            f.update(mismatched=float(s.endswith("mismatched")), genre=df.genre.iat[i], label="ENC"[df.label.iat[i]])
            rows.append(f)
    return pd.DataFrame(rows)


def main():
    data = load_mnli()
    feats = features(data)
    systems = {name: [run.key(j) for j in run.cv(c)] for name, c in run.MAIN.items()}
    k = {name: correct_counts(data, keys) for name, keys in systems.items()}
    if len(sys.argv) > 1:
        k = {"p1-siamese": project1_counts(data, Path(sys.argv[1])), **k}
    hyp_only = k.pop("rnn-hyp") / 15            # how far the hypothesis alone gets: an annotation-artifact score
    feats["hyp_only_correct"] = hyp_only
    # how sure the hypothesis-only model is (model-specific feature of a model we do not compare)
    feats["hyp_only_confidence"] = mean_confidence(systems["rnn-hyp"])
    names = list(k)
    n_items = len(feats)
    print(f"=== Q6: {n_items} validation pairs; systems: {names}; 15 fits each")

    print("\n--- Accuracy (%) by subgroup, mean over the 15 fits")
    acc = pd.DataFrame({n: 100 * v / 15 for n, v in k.items()})
    groups = {
        "all": np.ones(n_items, bool),
        **{f"label {l}": (feats.label == l).to_numpy() for l in "ENC"},
        "negation in hypothesis, E": ((feats.neg_hypothesis == 1) & (feats.label == "E")).to_numpy(),
        "negation in hypothesis, N": ((feats.neg_hypothesis == 1) & (feats.label == "N")).to_numpy(),
        "negation in hypothesis, C": ((feats.neg_hypothesis == 1) & (feats.label == "C")).to_numpy(),
        "no negation, C": ((feats.neg_hypothesis == 0) & (feats.label == "C")).to_numpy(),
        "overlap >= 0.75, N": ((feats.overlap >= 0.75) & (feats.label == "N")).to_numpy(),
        "overlap < 0.25, E": ((feats.overlap < 0.25) & (feats.label == "E")).to_numpy(),
        "antonym pair": (feats.antonym_pair == 1).to_numpy(),
        "quantifier in hypothesis": (feats.quant_hypothesis == 1).to_numpy(),
        "premise > 30 tokens": (feats.len_premise > 30).to_numpy(),
        "premise <= 10 tokens": (feats.len_premise <= 10).to_numpy(),
        "parse depth premise >= 12": (feats.depth_premise >= 12).to_numpy(),
        "some token without vector": (feats.oov > 0).to_numpy(),
        "hypothesis-only gets it (>= 8/15)": (feats.hyp_only_correct >= 8 / 15).to_numpy(),
        "hypothesis-only fails (< 8/15)": (feats.hyp_only_correct < 8 / 15).to_numpy(),
        "matched": (feats.mismatched == 0).to_numpy(),
        "mismatched": (feats.mismatched == 1).to_numpy(),
    }
    table = pd.DataFrame({g: acc[m].mean() for g, m in groups.items()}).T
    table["n"] = [int(m.sum()) for m in groups.values()]
    print(table.round(1).to_string())
    print("Binomial SE of one cell with n pairs: about 50/sqrt(n) pp (n = 1000: 1.6 pp), before averaging fits.")

    cols = ["len_premise", "len_hypothesis", "depth_premise", "depth_hypothesis", "overlap", "neg_premise",
            "neg_hypothesis", "quant_hypothesis", "oov", "rarest_hyp_word", "antonym_pair", "alignment",
            "hyp_only_correct", "hyp_only_confidence", "mismatched", "label_N", "label_C"]
    feats["label_N"] = (feats.label == "N").astype(float)
    feats["label_C"] = (feats.label == "C").astype(float)
    X = feats[cols].to_numpy(float)
    print("\n--- Feature correlations above 0.5 in absolute value")
    corr = np.corrcoef(X.T)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr[i, j]) > 0.5:
                print(f"   {cols[i]} ~ {cols[j]}: {corr[i, j]:.2f}")

    sys_id = np.repeat(np.arange(len(names)), n_items)
    mcmc = bda.fit(sys_id, np.tile(X, (len(names), 1)), np.concatenate([k[n] for n in names]),
                   np.full(len(names) * n_items, 15), len(names))
    post = mcmc.get_samples(group_by_chain=True)
    divergences = int(mcmc.get_extra_fields()["diverging"].sum())
    from numpyro.diagnostics import summary
    s = summary(post, group_by_chain=True)
    rhat = max(float(np.max(v["r_hat"])) for v in s.values())
    print(f"\n--- Bayesian regression (tutorial 2's model, Binomial over 15 fits): max r_hat {rhat:.3f}, divergences {divergences}")
    w = np.asarray(post["w"]).reshape(-1, len(names), len(cols))
    print("Weight per standard deviation of the feature: posterior mean [95% interval]")
    rows = []
    for j, c in enumerate(cols):
        row = {"feature": c}
        for i, n in enumerate(names):
            lo, hi = np.percentile(w[:, i, j], [2.5, 97.5])
            row[n] = f"{w[:, i, j].mean():+.2f} [{lo:+.2f},{hi:+.2f}]"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n--- Differences in weights between designs: mean, 95% interval, P(difference > 0)")
    comparisons = [(a, b) for a, b in (("pool", "pool-random"), ("rnn", "rnn-random"), ("rnn", "pool"),
                                        ("pool", "p1-siamese"), ("rnn", "p1-siamese")) if a in names and b in names]
    rows = []
    for a, b in comparisons:
        d = w[:, names.index(a)] - w[:, names.index(b)]
        for j, c in enumerate(cols):
            lo, hi = np.percentile(d[:, j], [2.5, 97.5])
            if lo > 0 or hi < 0:
                rows.append({"comparison": f"{a} - {b}", "feature": c, "mean": round(float(d[:, j].mean()), 2),
                             "interval": f"[{lo:+.2f},{hi:+.2f}]", "P>0": round(float((d[:, j] > 0).mean()), 3)})
    print("(only intervals that exclude 0)")
    print(pd.DataFrame(rows).to_string(index=False))
    tag = os.environ.get("BDA_CHAINS", "")
    tag = f"_{tag}" if tag == "parallel" else ""
    out = run.RESULTS / f"errors_posterior{tag}.npz"
    np.savez_compressed(out, w=w, names=np.array(names), cols=np.array(cols))
    feats.assign(**{f"k_{n}": v for n, v in k.items()}).to_csv(run.RESULTS / f"errors_items{tag}.csv.gz", index=False)


if __name__ == "__main__":
    main()
