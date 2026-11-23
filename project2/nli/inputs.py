"""From MultiNLI text to padded id matrices for one (tokeniser, embedding table, case) choice.

Everything is cached under cache/, keyed by that choice. The model vocabulary is the set of
MultiNLI word types that the table covers; every other token maps to <unk>, whose vector is
the mean of all vectors in the file (tutorial 2). For embeddings trained from scratch
(`random`), the vocabulary is the lowercased types seen at least twice in the CV pool.
"""
import os
import pickle
from pathlib import Path

import numpy as np

from .embeddings import SOURCES, Table, cache_table
from .text import lookup_forms, map_to_rows, pad_ids, tokenise

SPLITS = ("train", "validation_matched", "validation_mismatched")
MAX_LEN = 200          # tokens per sentence; longer ones are truncated (share reported by explore.py)
PAD, UNK = 0, 1


def _atomic_save(path: Path, write):
    tmp = path.with_name(f".{path.stem}.{os.getpid()}{path.suffix}")    # same suffix, so np.savez adds none
    write(tmp)
    os.replace(tmp, path)


def tokens(kind: str, data: dict, cache: Path) -> dict:
    """Per split: tokens of each distinct premise, the premise of each pair, tokens of each hypothesis."""
    path = cache / f"tokens_{kind}.pkl"
    if not path.exists():
        out = {}
        for s in SPLITS:
            df = data[s]
            first = df.drop_duplicates("premise")
            prem_row = {t: i for i, t in enumerate(first.premise)}
            out[s] = {"prem": tokenise(list(first.premise), list(first.premise_binary_parse), kind),
                      "pair_prem": df.premise.map(prem_row).to_numpy(),
                      "hyp": tokenise(list(df.hypothesis), list(df.hypothesis_binary_parse), kind)}
        cache.mkdir(exist_ok=True)
        _atomic_save(path, lambda p: p.write_bytes(pickle.dumps(out)))
    return pickle.loads(path.read_bytes())


def all_types(toks: dict) -> set:
    return {t for s in SPLITS for lst in (toks[s]["prem"], toks[s]["hyp"]) for x in lst for t in x}


def table(name: str, data: dict, cache: Path, tokenisers=("nltk", "parse", "split", "tutorial-bow")) -> Table:
    """The rows of an embedding file for every form of every MultiNLI token under any tokeniser."""
    path = cache / f"emb_{name}.npz"
    if not path.exists():
        wanted = set()
        for kind in tokenisers:
            for t in all_types(tokens(kind, data, cache)):
                wanted.update(lookup_forms(t, "exact"))
        _atomic_save(path, lambda p: cache_table(name, wanted, p))
    return Table(path)


def resolve_case(emb: str, case: str) -> str:
    if emb == "random":
        return "lower"
    if case == "auto":
        return "exact" if SOURCES[emb]["cased"] else "lower"
    return case


def build(tok: str, emb: str, case: str, data: dict, pool: np.ndarray, cache: Path) -> dict:
    case = resolve_case(emb, case)
    path = cache / f"inputs_{tok}_{emb}_{case}.npz"
    if path.exists():
        return dict(np.load(path, allow_pickle=True))
    toks = tokens(tok, data, cache)
    if emb == "random":
        tr = toks["train"]
        counts = {}
        pool_prem = np.unique(tr["pair_prem"][pool])
        for lst in ([tr["prem"][i] for i in pool_prem], [tr["hyp"][i] for i in pool]):
            for x in lst:
                for t in x:
                    counts[t.lower()] = counts.get(t.lower(), 0) + 1
        vocab = sorted(t for t, c in counts.items() if c >= 2)
        type_id = {t: i + 2 for i, t in enumerate(vocab)}
        to_id = lambda t: type_id.get(t.lower(), UNK)
        matrix, n_vocab = None, len(vocab) + 2
    else:
        tab = table(emb, data, cache)
        rows = map_to_rows(all_types(toks), tab.index, case)
        used = sorted(set(rows.values()))
        row_id = {r: i + 2 for i, r in enumerate(used)}
        type_id = {t: row_id[r] for t, r in rows.items()}
        to_id = lambda t: type_id.get(t, UNK)
        matrix = np.concatenate([np.zeros((1, tab.vectors.shape[1]), np.float32), tab.mean[None],
                                 tab.vectors[used]]).astype(np.float32)
        n_vocab = len(used) + 2
    out = {"n_vocab": n_vocab, "case": case}
    if matrix is not None:
        out["matrix"] = matrix
    for s in SPLITS:
        prem_ids = [[to_id(t) for t in x] for x in toks[s]["prem"]]
        hyp_ids = [[to_id(t) for t in x] for x in toks[s]["hyp"]]
        out[f"{s}/prem"], out[f"{s}/prem_len"] = pad_ids(prem_ids, MAX_LEN)
        out[f"{s}/hyp"], out[f"{s}/hyp_len"] = pad_ids(hyp_ids, MAX_LEN)
        out[f"{s}/pair_prem"] = toks[s]["pair_prem"]
        pp = toks[s]["pair_prem"]
        shared = [sorted((set(prem_ids[pp[i]]) & set(h)) - {PAD, UNK}) for i, h in enumerate(hyp_ids)]
        out[f"{s}/shared"], _ = pad_ids(shared)
    _atomic_save(path, lambda p: np.savez(p, **out))
    return dict(np.load(path, allow_pickle=True))
