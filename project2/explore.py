"""Q1 and Q2: tokenisation, coverage and what the embedding tables capture.

Usage: python explore.py caches     # tokenise MultiNLI and cache the rows of every table
       python explore.py table NAME # cache one table (the tables can be built in parallel)
       python explore.py q1 > results/q1_coverage.txt
       python explore.py q2 > results/q2_embeddings.txt
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from nli import inputs
from nli.data import carve_tuning_set, load_mnli
from nli.embeddings import SOURCES
from nli.text import map_to_rows

CACHE = Path("cache")
TOKENISERS = ("nltk", "parse", "split", "tutorial-bow")
PROBE = ["n't", "'s", "'re", "do", "a", "of", "and", "to", "not", "no", ",", ".", "?", "uh", "um", "yeah",
         "gonna", "kinda", "Clinton", "U.S.", "e-mail", "1990s", "30", "%", "$", "--"]


def caches(data):
    for kind in TOKENISERS:
        inputs.tokens(kind, data, CACHE)
        print("tokens", kind, flush=True)
    for name in SOURCES:
        t = inputs.table(name, data, CACHE)
        print("table", name, len(t.words), "rows kept of", t.file_size, flush=True)


def pair_tokens(toks, split, rows=None):
    """Premise tokens (each distinct premise once) and hypothesis tokens."""
    s = toks[split]
    hyp = s["hyp"] if rows is None else [s["hyp"][i] for i in rows]
    prem_rows = np.unique(s["pair_prem"] if rows is None else s["pair_prem"][rows])
    return [s["prem"][i] for i in prem_rows] + hyp


def q1(data):
    pool, _ = carve_tuning_set(data["train"].promptID.to_numpy())
    print("=== Q1: coverage of MultiNLI by each table (CV pool, validation matched / mismatched)")
    print("type cov: share of distinct tokens with a vector; token cov: share of running tokens\n")
    rows = []
    for kind in ("nltk", "parse", "split"):
        toks = inputs.tokens(kind, data, CACHE)
        texts = {"pool": pair_tokens(toks, "train", pool), "matched": pair_tokens(toks, "validation_matched"),
                 "mismatched": pair_tokens(toks, "validation_mismatched")}
        counts = {k: Counter(t for x in v for t in x) for k, v in texts.items()}
        for name in SOURCES:
            tab = inputs.table(name, data, CACHE)
            for case in ("exact", "lower"):
                if case == "exact" and not SOURCES[name]["cased"] and kind != "nltk":
                    continue
                r = dict(tok=kind, table=name, case=case)
                for k, c in counts.items():
                    hit = map_to_rows(c, tab.index, case)
                    r[f"{k}_type"] = 100 * len(hit) / len(c)
                    r[f"{k}_token"] = 100 * sum(c[t] for t in hit) / sum(c.values())
                rows.append(r)
    df = pd.DataFrame(rows)
    print(df.round(2).to_string(index=False))

    print("\n=== Tables")
    for name, src in SOURCES.items():
        tab = inputs.table(name, data, CACHE)
        print(f"{name:13s} {tab.file_size:>9,} words, {100 * tab.share_cased:4.1f}% with capitals, "
              f"{src['dim']}d; {src['about']}")

    toks = inputs.tokens("nltk", data, CACHE)
    c = Counter(t for x in pair_tokens(toks, "train", pool) for t in x)
    print("\n=== Probe tokens (NLTK, count in pool): which tables have them, as written / lowercased")
    for t in PROBE:
        have = [f"{n}:{'y' if t in inputs.table(n, data, CACHE).index else ('l' if t.lower() in inputs.table(n, data, CACHE).index else '-')}"
                for n in ("glove840B", "fasttext", "word2vec", "glove6B-300", "glove2024-50", "counterfit", "numberbatch")]
        print(f"{t!r:10s} {c[t]:>8,}  " + " ".join(have))

    for name, case in (("glove840B", "exact"), ("fasttext", "exact"), ("word2vec", "exact"), ("glove6B-300", "lower")):
        tab = inputs.table(name, data, CACHE)
        hit = map_to_rows(c, tab.index, case)
        miss = Counter({t: n for t, n in c.items() if t not in hit})
        print(f"\nMost frequent tokens without a vector, {name} ({case}), NLTK:", miss.most_common(40))

    print("\n=== Token coverage by genre (NLTK; validation sets; glove840B exact / fasttext exact / word2vec exact / glove6B-300 lower)")
    for split in ("validation_matched", "validation_mismatched"):
        df_s = data[split]
        for g in sorted(df_s.genre.unique()):
            rows_g = np.flatnonzero(df_s.genre.to_numpy() == g)
            cg = Counter(t for x in pair_tokens(toks, split, rows_g) for t in x)
            cov = []
            for name, case in (("glove840B", "exact"), ("fasttext", "exact"), ("word2vec", "exact"), ("glove6B-300", "lower")):
                hit = map_to_rows(cg, inputs.table(name, data, CACHE).index, case)
                cov.append(100 * sum(cg[t] for t in hit) / sum(cg.values()))
            print(f"{split[11:]:10s} {g:12s} " + "  ".join(f"{x:6.2f}" for x in cov))

    print("\n=== Lengths (NLTK tokens) and truncation at", inputs.MAX_LEN)
    for split in inputs.SPLITS:
        s = toks[split]
        lp = np.array([len(x) for x in s["prem"]])
        lh = np.array([len(x) for x in s["hyp"]])
        print(f"{split:22s} premise mean {lp.mean():.1f} p99 {np.percentile(lp, 99):.0f} max {lp.max()}; "
              f"hypothesis mean {lh.mean():.1f} p99 {np.percentile(lh, 99):.0f} max {lh.max()}; "
              f"truncated premises {(lp > inputs.MAX_LEN).sum()}, hypotheses {(lh > inputs.MAX_LEN).sum()}")
    bow = inputs.tokens("tutorial-bow", data, CACHE)["train"]
    empty_h = sum(len(x) == 0 for x in bow["hyp"])
    print(f"tutorial-bow filter: {empty_h} training hypotheses become empty; "
          f"mean hypothesis length {np.mean([len(x) for x in bow['hyp']]):.1f}")


Q2_TABLES = ("glove840B", "fasttext", "word2vec", "glove6B-300", "glove2024-50", "counterfit", "numberbatch")


def unit_rows(tab, words, case):
    rows = map_to_rows(words, tab.index, case)
    vec = {w: tab.vectors[r] / (np.linalg.norm(tab.vectors[r]) + 1e-12) for w, r in rows.items()}
    return vec


def auc(pos, neg):
    """P(score of a random positive > score of a random negative)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def wordnet_pairs(words: set, rng, n_random=5000):
    """Word pairs in five relations, both words in `words` (frequent lowercase MultiNLI words)."""
    from nltk.corpus import wordnet as wn
    syn, ant, hyper, cohypo = set(), set(), set(), set()
    for w in words:
        for s in wn.synsets(w):
            lemmas = [l.name() for l in s.lemmas()]
            syn.update((w, l) for l in lemmas if l != w and l in words)
            for l in s.lemmas():
                if l.name() == w:
                    ant.update((w, a.name()) for a in l.antonyms() if a.name() in words)
            for h in s.hypernyms():
                hyper.update((w, l.name()) for l in h.lemmas() if l.name() in words and l.name() != w)
                for sib in h.hyponyms():
                    if sib != s:
                        cohypo.update((w, l.name()) for l in sib.lemmas() if l.name() in words and l.name() != w)
    norm = lambda pairs: sorted({tuple(sorted(p)) for p in pairs})
    syn, ant, cohypo = norm(syn), norm(ant), norm(cohypo)
    hyper = sorted(set(hyper))
    bad = set(syn) & set(ant)                    # a few pairs are listed as both; drop them from both
    syn, ant = [p for p in syn if p not in bad], [p for p in ant if p not in bad]
    taken = set(syn) | set(ant)
    cohypo = [p for p in cohypo if p not in taken]
    wl = sorted(words)
    rand = sorted({tuple(sorted(rng.choice(wl, 2, replace=False))) for _ in range(n_random)})
    return {"synonym": syn, "antonym": ant, "hypernym": hyper, "co-hyponym": cohypo, "random": rand}


def q2(data):
    rng = np.random.default_rng(0)
    pool, _ = carve_tuning_set(data["train"].promptID.to_numpy())
    toks = inputs.tokens("nltk", data, CACHE)
    counts = Counter(t for x in pair_tokens(toks, "train", pool) for t in x)
    tabs = {n: inputs.table(n, data, CACHE) for n in Q2_TABLES}
    common = {w for w, c in counts.items() if c >= 20 and w.isalpha() and w.islower()}
    for n, t in tabs.items():          # same words for every table, so the comparison is fair
        common &= set(map_to_rows(common, t.index, "exact" if SOURCES[n]["cased"] else "lower"))
    pairs = wordnet_pairs(common, rng)
    print(f"=== Q2: cosine similarity by WordNet relation; {len(common)} lowercase MultiNLI words seen 20+ times "
          f"and covered by all {len(tabs)} tables")
    print("pairs per relation:", {k: len(v) for k, v in pairs.items()})
    rows = []
    vecs = {}
    for n, t in tabs.items():
        vecs[n] = unit_rows(t, common, "exact" if SOURCES[n]["cased"] else "lower")
        cos = {rel: np.array([vecs[n][a] @ vecs[n][b] for a, b in ps]) for rel, ps in pairs.items()}
        r = {"table": n, **{f"{rel}": f"{c.mean():.2f} ({c.std():.2f})" for rel, c in cos.items()}}
        r["AUC syn>ant"] = round(auc(cos["synonym"], cos["antonym"]), 3)
        r["AUC syn>cohyp"] = round(auc(cos["synonym"], cos["co-hyponym"]), 3)
        r["AUC ant>random"] = round(auc(cos["antonym"], cos["random"]), 3)
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Mean (SD) cosine. AUC syn>ant: chance that a synonym pair is closer than an antonym pair (0.5 = no separation).")
    print("Examples:", {k: v[:8] for k, v in pairs.items()})

    print("\n=== Nearest neighbours among MultiNLI words (cosine), words that matter for entailment")
    probe = ["cheap", "increase", "hot", "never", "always", "all", "none", "man", "dog", "buy", "allowed", "true"]
    for n in ("glove840B", "word2vec", "counterfit"):
        words = sorted(vecs[n])
        m = np.stack([vecs[n][w] for w in words])
        print(f"-- {n}")
        for w in probe:
            if w in vecs[n]:
                sims = m @ vecs[n][w]
                top = [words[i] for i in np.argsort(-sims)[1:9]]
                print(f"   {w:9s} {top}")

    print("\n=== Task evidence: MultiNLI pairs where a premise word and a hypothesis word are WordNet antonyms")
    ant = {p for p in pairs["antonym"]}
    tr = toks["train"]
    labels = data["train"].label.to_numpy()
    found = []
    for i in pool:
        pw = {t for t in tr["prem"][tr["pair_prem"][i]] if t in common}
        hw = {t for t in tr["hyp"][i] if t in common}
        hits = [(a, b) for a in pw - hw for b in hw - pw if tuple(sorted((a, b))) in ant]
        if hits:
            found.append((i, hits[0]))
    lab = Counter(labels[i] for i, _ in found)
    print(f"{len(found)} pool pairs ({100 * len(found) / len(pool):.1f}%); labels E/N/C:",
          [round(100 * lab[k] / len(found), 1) for k in range(3)], "(pool: 33.3 each)")
    print("Cosine of that antonym pair, by table:",
          {n: round(float(np.mean([vecs[n][a] @ vecs[n][b] for _, (a, b) in found])), 2) for n in tabs})

    print("\n=== Task evidence: alignment of new hypothesis words with the premise, by label")
    print("For each pair: mean over hypothesis words not in the premise of their highest cosine with a premise word.")
    sample = rng.choice(pool, 30000, replace=False)
    out = []
    for n in tabs:
        v = vecs[n]
        score = []
        for i in sample:
            pw = [t for t in set(tr["prem"][tr["pair_prem"][i]]) if t in v]
            hw = [t for t in set(tr["hyp"][i]) - set(tr["prem"][tr["pair_prem"][i]]) if t in v]
            if not pw or not hw:
                score.append(np.nan)
                continue
            P, H = np.stack([v[t] for t in pw]), np.stack([v[t] for t in hw])
            score.append(float((H @ P.T).max(1).mean()))
        s = np.array(score)
        y = labels[sample]
        ok = ~np.isnan(s)
        r = {"table": n, **{f"{l}": round(float(np.nanmean(s[y == k])), 3) for k, l in enumerate("ENC")}}
        r["AUC E>C"] = round(auc(s[ok & (y == 0)], s[ok & (y == 2)]), 3)
        r["AUC E>N"] = round(auc(s[ok & (y == 0)], s[ok & (y == 1)]), 3)
        out.append(r)
    print(pd.DataFrame(out).to_string(index=False))
    print(f"({ok.sum()} of {len(sample)} sampled pool pairs have new hypothesis words covered by every table)")

    print("\n=== Limitation: gender direction (he - she), projection of occupation words (Bolukbasi et al., 2016)")
    jobs = ["nurse", "secretary", "teacher", "librarian", "receptionist", "doctor", "engineer", "lawyer",
            "programmer", "pilot", "boss", "scientist", "soldier", "architect", "president", "manager"]
    for n in ("glove840B", "fasttext", "word2vec", "counterfit"):
        v = vecs[n]
        if "he" not in v or "she" not in v:
            print(n, "lacks he/she")
            continue
        g = v["he"] - v["she"]
        g /= np.linalg.norm(g)
        proj = sorted(((float(v[j] @ g), j) for j in jobs if j in v))
        print(f"{n:11s} most 'she': {[j for _, j in proj[:4]]}  most 'he': {[j for _, j in proj[-4:]]}")

    print("\n=== Limitation: one vector per word type (polysemy)")
    for n in ("glove840B",):
        words = sorted(vecs[n])
        m = np.stack([vecs[n][w] for w in words])
        for w in ("bank", "right", "fine", "spring", "plant"):
            if w in vecs[n]:
                print(f"   {w:7s}", [words[i] for i in np.argsort(-(m @ vecs[n][w]))[1:11]])


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    data = load_mnli()
    if what == "caches":
        caches(data)
    elif what == "table":
        t = inputs.table(sys.argv[2], data, CACHE)
        print(sys.argv[2], len(t.words), "rows kept of", t.file_size)
    elif what == "q1":
        q1(data)
    elif what == "q2":
        q2(data)
    else:
        sys.exit(__doc__)
