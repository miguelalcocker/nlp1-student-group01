"""Tokenisation and representation spaces for premise-hypothesis pairs.

A representation is a `Blocks` triple (premise, hypothesis, extra) of sparse matrices.
Flat models (log-linear, MLP) see the blocks stacked side by side; the Siamese MLP
projects the premise and hypothesis blocks through one shared embedding matrix.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Metaspace, Whitespace
from tokenizers.trainers import BpeTrainer, WordLevelTrainer


def fit_tokenizer(texts, vocab_size: int, kind: str = "bpe", lowercase: bool = False,
                  min_frequency: int = 2) -> Tokenizer:
    """Train a BPE tokeniser (tutorial setup) or a word-level one (as a comparison)."""
    if kind == "bpe":
        tok = Tokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = Metaspace()
        trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                             special_tokens=["[UNK]"])
    elif kind == "word":
        tok = Tokenizer(WordLevel(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                                   special_tokens=["[UNK]"])
    else:
        raise ValueError(kind)
    if lowercase:
        tok.normalizer = Lowercase()
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def token_counts(tok: Tokenizer, texts: list[str]) -> sp.csr_matrix:
    """Sparse [n_texts, V] token counts. Encodes each distinct text only once."""
    uniq, inverse = np.unique(np.asarray(texts, dtype=object), return_inverse=True)
    encodings = tok.encode_batch([str(t) for t in uniq], add_special_tokens=False)
    lengths = np.fromiter((len(e.ids) for e in encodings), dtype=np.int64, count=len(encodings))
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    indices = np.fromiter((i for e in encodings for i in e.ids), dtype=np.int32, count=indptr[-1])
    data = np.ones_like(indices, dtype=np.float32)
    V = tok.get_vocab_size()
    X = sp.csr_matrix((data, indices, indptr), shape=(len(uniq), V))
    X.sum_duplicates()
    return X[inverse].tocsr()


def binarise(X: sp.csr_matrix) -> sp.csr_matrix:
    X = X.copy()
    X.data = np.ones_like(X.data)
    return X


def cross_unigrams(P: sp.csr_matrix, H: sp.csr_matrix, bits: int = 18) -> sp.csr_matrix:
    """Hashed (premise token, hypothesis token) indicators, after Bowman et al. (2015)."""
    D = 1 << bits
    V = P.shape[1]
    rows, cols = [], []
    for i in range(P.shape[0]):
        p = P.indices[P.indptr[i]:P.indptr[i + 1]].astype(np.int64)
        h = H.indices[H.indptr[i]:H.indptr[i + 1]].astype(np.int64)
        pairs = np.unique(((np.outer(p, np.ones_like(h)) * V + h) * 2654435761 % (1 << 32)) >> (32 - bits))
        rows.append(np.full(pairs.size, i, dtype=np.int32))
        cols.append(pairs.astype(np.int32))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones_like(rows, dtype=np.float32), (rows, cols)), shape=(P.shape[0], D))


@dataclass
class Blocks:
    premise: sp.csr_matrix | None
    hypothesis: sp.csr_matrix | None
    extra: sp.csr_matrix | None = None

    def flat(self) -> sp.csr_matrix:
        parts = [b for b in (self.premise, self.hypothesis, self.extra) if b is not None]
        return sp.hstack(parts, format="csr") if len(parts) > 1 else parts[0]

    def take(self, idx) -> "Blocks":
        return Blocks(*[b[idx] if b is not None else None
                        for b in (self.premise, self.hypothesis, self.extra)])


def build_representation(name: str, P: sp.csr_matrix, H: sp.csr_matrix) -> Blocks:
    """Map premise and hypothesis token counts to one of the compared spaces."""
    if name == "bow-concat":     # tutorial encoder fed "premise hypothesis" as one text
        return Blocks(None, None, (P + H).tocsr())
    if name == "ind":            # one indicator block per sentence
        return Blocks(binarise(P), binarise(H))
    if name == "bow":            # one bag-of-words block per sentence
        return Blocks(P, H)
    if name == "prem-only":
        return Blocks(P, None)
    if name == "hyp-only":
        return Blocks(None, H)
    if name == "tfidf":
        return Blocks(TfidfTransformer().fit_transform(P).tocsr(),
                      TfidfTransformer().fit_transform(H).tocsr())
    if name == "overlap":        # bags + "this token occurs in both sentences"
        return Blocks(P, H, binarise(P.multiply(H).tocsr()))
    if name == "and":            # only "this token occurs in both sentences"
        return Blocks(None, None, binarise(P.multiply(H).tocsr()))
    if name == "hyp-and":        # hypothesis bag + overlap, no premise block
        return Blocks(None, H, binarise(P.multiply(H).tocsr()))
    if name == "cross":          # bags + hashed cross-sentence token pairs
        return Blocks(P, H, cross_unigrams(P, H))
    raise ValueError(name)
