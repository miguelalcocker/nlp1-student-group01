"""Pretrained embedding tables: where they come from, how to read them, and a cache of the rows we need.

The files are large (up to 2.2M words), but a model can only ever use the rows of words
that occur in MultiNLI. `cache_table` reads a file once and keeps the rows for a given set of
word forms, plus the mean of *all* vectors in the file (tutorial 2 uses it for unknown words).
"""
import gzip
import io
import zipfile
from pathlib import Path

import numpy as np

EMB_DIR = Path("/tmp/nlp2_emb")    # overridden by the NLP2_EMB environment variable in run.py

# name -> file, member inside a zip, dimension, header line, cased, source
SOURCES = {
    "glove2024-50": dict(file="glove.2024.wikigiga.50d.zip", dim=50, cased=False,
                         about="GloVe 2024, Wikipedia + Gigaword, 50d (tutorial 2)"),
    **{f"glove6B-{d}": dict(file="glove.6B.zip", member=f"glove.6B.{d}d.txt", dim=d, cased=False,
                            about=f"GloVe 6B (2014), Wikipedia + Gigaword, {d}d")
       for d in (50, 100, 200, 300)},
    "glove840B": dict(file="glove.840B.300d.zip", dim=300, cased=True,
                      about="GloVe 840B, Common Crawl, cased, 300d"),
    "fasttext": dict(file="crawl-300d-2M.vec.zip", dim=300, cased=True, header=True,
                     about="fastText, Common Crawl, cased, 300d"),
    "word2vec": dict(file="GoogleNews-vectors-negative300.bin.gz", dim=300, cased=True, binary=True,
                     about="word2vec skip-gram, Google News, cased, 300d"),
    "counterfit": dict(file="counter-fitted-vectors.txt.zip", dim=300, cased=False,
                       about="Counter-fitted (Mrksic et al. 2016): Paragram-SL999 with antonyms pushed apart"),
    "numberbatch": dict(file="numberbatch-en-19.08.txt.gz", dim=300, cased=False, header=True,
                        about="ConceptNet Numberbatch 19.08, English, 300d"),
}


def _text_lines(path: Path, member: str | None):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = member or z.namelist()[0]
            with z.open(name) as f:
                yield from io.TextIOWrapper(f, encoding="utf-8", errors="replace")
    elif path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            yield from f
    else:
        with path.open(encoding="utf-8", errors="replace") as f:
            yield from f


def _word2vec_binary(path: Path):
    """The original word2vec binary format: a header, then each word followed by float32s."""
    with gzip.open(path, "rb") as f:
        n, dim = map(int, f.readline().split())
        for _ in range(n):
            word = bytearray()
            while (c := f.read(1)) != b" ":
                if c != b"\n":
                    word += c
            yield word.decode("utf-8", errors="replace"), np.frombuffer(f.read(4 * dim), dtype=np.float32)


def iter_table(name: str, emb_dir: Path = EMB_DIR):
    """Yield (word, vector) pairs in file order. Malformed lines are skipped, as in tutorial 2."""
    src = SOURCES[name]
    path = emb_dir / src["file"]
    if src.get("binary"):
        yield from _word2vec_binary(path)
        return
    dim = src["dim"]
    for i, line in enumerate(_text_lines(path, src.get("member"))):
        if i == 0 and src.get("header"):
            continue
        parts = line.rstrip().split(" ")
        if len(parts) != dim + 1:
            continue
        yield parts[0], parts[1:]


def cache_table(name: str, wanted: set, out: Path, emb_dir: Path = EMB_DIR) -> None:
    """Save the rows of `wanted` word forms, the file's mean vector and its size."""
    dim = SOURCES[name]["dim"]
    words, vecs, seen = [], [], set()
    total, n, n_upper = np.zeros(dim, dtype=np.float64), 0, 0
    for word, vec in iter_table(name, emb_dir):
        if word in seen:            # keep the first (most frequent) entry, as tutorial 2 does
            continue
        seen.add(word)
        v = np.asarray(vec, dtype=np.float32)
        total += v
        n += 1
        n_upper += word != word.lower()
        if word in wanted:
            words.append(word)
            vecs.append(v)
    np.savez(out, words=np.array(words, dtype=object), vectors=np.stack(vecs), mean=(total / n).astype(np.float32),
             file_size=n, share_cased=n_upper / n)


class Table:
    """The cached rows of one embedding file."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=True)
        self.words = list(z["words"])
        self.vectors = z["vectors"]
        self.mean = z["mean"]
        self.file_size = int(z["file_size"])
        self.share_cased = float(z["share_cased"])
        self.index = {w: i for i, w in enumerate(self.words)}
