"""Features of a premise-hypothesis pair for error analysis (Q6).

Each is an aspect of difficulty that we expected, from project 1 and the literature, to change
which architecture gets a pair right:
- length of each sentence, and depth of its binary parse (syntactic complexity);
- lexical overlap (share of hypothesis words in the premise; McCoy et al. 2019);
- negation and quantifiers in each sentence (Gururangan et al. 2018 found negation in
  hypotheses to signal contradiction);
- words the table does not cover, and the rarest hypothesis word in training;
- a WordNet antonym pair across the sentences (embeddings place antonyms close, Q2);
- how well each new hypothesis word aligns with some premise word in embedding space.
"""
import numpy as np

NEGATION = {"not", "n't", "no", "never", "nobody", "nothing", "none", "nor", "neither", "nowhere", "cannot", "without"}
QUANTIFIERS = {"all", "every", "each", "some", "any", "most", "many", "few", "several", "both", "always", "only",
               "none", "nobody", "everyone", "everybody", "someone", "somebody", "anyone", "anybody"}


def parse_depth(binary_parse: str) -> int:
    depth = best = 0
    for ch in binary_parse:
        if ch == "(":
            depth += 1
            best = max(best, depth)
        elif ch == ")":
            depth -= 1
    return best


def wordnet_antonyms(words: set) -> set:
    from nltk.corpus import wordnet as wn
    pairs = set()
    for w in words:
        for s in wn.synsets(w):
            for lemma in s.lemmas():
                if lemma.name() == w:
                    pairs.update(tuple(sorted((w, a.name()))) for a in lemma.antonyms() if a.name() in words)
    return pairs


def pair_features(prem_toks, hyp_toks, prem_parse, hyp_parse, covered: set, unit: dict, log_freq: dict,
                  antonyms: set) -> dict:
    """prem_toks / hyp_toks: NLTK tokens; covered: token types with a vector; unit: lowercase word -> unit vector."""
    p_low, h_low = [t.lower() for t in prem_toks], [t.lower() for t in hyp_toks]
    p_set, h_set = set(p_low), set(h_low)
    h_words = [t for t in h_low if t.isalpha()]
    new = [t for t in set(h_words) - p_set if t in unit]
    p_vec = [unit[t] for t in p_set if t in unit]
    if new and p_vec:
        align = float((np.stack([unit[t] for t in new]) @ np.stack(p_vec).T).max(1).mean())
    else:
        align = 1.0                      # nothing new in the hypothesis
    content = [t for t in h_words if t in log_freq]
    return {
        "len_premise": len(prem_toks),
        "len_hypothesis": len(hyp_toks),
        "depth_premise": parse_depth(prem_parse),
        "depth_hypothesis": parse_depth(hyp_parse),
        "overlap": float(np.mean([t in p_set for t in h_words])) if h_words else 0.0,
        "neg_premise": float(bool(NEGATION & p_set)),
        "neg_hypothesis": float(bool(NEGATION & h_set)),
        "quant_hypothesis": float(bool(QUANTIFIERS & h_set)),
        "oov": float(np.mean([t not in covered for t in list(prem_toks) + list(hyp_toks)])),
        "rarest_hyp_word": min((log_freq[t] for t in content), default=0.0),
        "antonym_pair": float(any(tuple(sorted((a, b))) in antonyms for a in p_set - h_set for b in h_set - p_set)),
        "alignment": align,
    }
