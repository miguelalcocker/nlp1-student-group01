"""Tokenisers compared in Q1, and the mapping from tokens to rows of an embedding table.

- `nltk`: NLTK's Treebank tokeniser, as in tutorial 2 ("don't" -> "do", "n't").
- `parse`: the tokens of MultiNLI's own Stanford parses, i.e. the Stanford PTB tokeniser
  that GloVe was trained with.
- `split`: whitespace, as a naive reference.
- `nltk-not`: `nltk` with the clitic negation spelt out ("n't" -> "not", "ca n't" -> "can not",
  "wo n't" -> "will not"), for tables without an "n't" vector (fastText, counter-fitted),
  which would otherwise map every "n't" to the unknown vector.

The `tutorial-bow` filter reproduces tutorial 2's tokeniser for embed-and-pool: lowercase,
punctuation replaced by spaces, NLTK stopwords removed. That stopword list contains "not",
"no", "nor", "n't" forms and quantifiers such as "all", "some" and "few".
"""
import re

import nltk
import numpy as np

PUNCT = re.compile(r"[^\w\s]")


def parse_tokens(binary_parse: str) -> list[str]:
    return [t for t in binary_parse.replace("(", " ").replace(")", " ").split()]


NEGATED_STEMS = {"ca": "can", "wo": "will", "sha": "shall"}


def spell_out_negation(tokens: list[str]) -> list[str]:
    out = []
    for i, t in enumerate(tokens):
        if t.lower() == "n't":
            out.append("not")
        elif i + 1 < len(tokens) and tokens[i + 1].lower() == "n't" and t.lower() in NEGATED_STEMS:
            full = NEGATED_STEMS[t.lower()]
            out.append(full.capitalize() if t[0].isupper() else full)
        else:
            out.append(t)
    return out


def tokenise(texts, parses, kind: str) -> list[list[str]]:
    if kind == "nltk":
        return [nltk.tokenize.word_tokenize(t) for t in texts]
    if kind == "nltk-not":
        return [spell_out_negation(nltk.tokenize.word_tokenize(t)) for t in texts]
    if kind == "parse":
        return [parse_tokens(p) for p in parses]
    if kind == "split":
        return [t.split() for t in texts]
    if kind == "tutorial-bow":
        stop = set(nltk.corpus.stopwords.words("english"))
        return [[w for w in nltk.tokenize.word_tokenize(PUNCT.sub(" ", t.lower())) if w not in stop] for t in texts]
    raise ValueError(kind)


def lookup_forms(token: str, case: str) -> tuple[str, ...]:
    """Forms tried, in order, to find a token in an embedding table.

    `exact` suits cased tables (GloVe 840B, fastText, word2vec): the token as written, then
    lowercased, then capitalised. `lower` lowercases first, as tutorial 2 does.
    """
    if case == "lower":
        return (token.lower(),)
    return tuple(dict.fromkeys((token, token.lower(), token.capitalize())))


def map_to_rows(types, table_index: dict, case: str) -> dict:
    """Token type -> row in the embedding table, for the types the table covers."""
    rows = {}
    for t in types:
        for form in lookup_forms(t, case):
            if form in table_index:
                rows[t] = table_index[form]
                break
    return rows


def coverage(token_lists, covered: set) -> dict:
    """Share of token types and of running tokens that the table covers."""
    counts = {}
    for toks in token_lists:
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
    n_tok = sum(counts.values())
    hit = {t: c for t, c in counts.items() if t in covered}
    return {"types": len(counts), "type_cov": len(hit) / max(1, len(counts)),
            "token_cov": sum(hit.values()) / max(1, n_tok), "tokens": n_tok}


def pad_ids(id_lists, max_len: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Right-padded id matrix (pad id 0) and lengths. Empty texts keep length 0."""
    lengths = np.array([len(x) for x in id_lists])
    width = max(1, int(lengths.max()) if max_len is None else min(max_len, int(lengths.max())))
    out = np.zeros((len(id_lists), width), dtype=np.int32)
    for i, ids in enumerate(id_lists):
        ids = ids[:width]
        out[i, :len(ids)] = ids
    return out, np.minimum(lengths, width)
