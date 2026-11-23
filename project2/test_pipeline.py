"""Checks on splits, readers, encoders, training and the posterior. Usage: python test_pipeline.py"""
import gzip
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

from nli.data import carve_tuning_set, grouped_kfold, load_mnli, subsample_premises
from nli.embeddings import Table, cache_table, iter_table, SOURCES
from nli.models import Embedder, PairClassifier, PoolEncoder, RNNEncoder
from nli.stats import posterior
from nli.text import lookup_forms, map_to_rows, pad_ids, parse_tokens, tokenise
from nli.train import PairTensors, fit, predict_log_probs


def test_splits(train):
    groups = train.promptID.to_numpy()
    pool, tune = carve_tuning_set(groups)
    assert not set(groups[pool]) & set(groups[tune])
    for seed in (0, 1, 2):
        for fold in grouped_kfold(groups[pool], 5, seed):
            assert not set(groups[pool][fold.train]) & set(groups[pool][fold.val])
    sub = subsample_premises(groups, pool, 0.1, 0)
    assert abs(len(np.unique(groups[sub])) / len(np.unique(groups[pool])) - 0.1) < 0.001
    assert set(groups[sub]) <= set(groups[pool]) and len(np.unique(groups[sub])) * 3 <= len(sub) + 100
    print("splits: no premise crosses tuning set, folds or subsamples")


def test_text():
    assert parse_tokens("( ( The cat ) ( sat . ) )") == ["The", "cat", "sat", "."]
    assert tokenise(["I don't know."], [""], "nltk")[0] == ["I", "do", "n't", "know", "."]
    assert tokenise(["Can't stop, won't stop, don't."], [""], "nltk-not")[0] == \
        ["Can", "not", "stop", ",", "will", "not", "stop", ",", "do", "not", "."]
    assert tokenise(["It is not all good."], [""], "tutorial-bow")[0] == ["good"]    # negation and "all" removed
    assert lookup_forms("Paris", "exact") == ("Paris", "paris") and lookup_forms("paris", "exact") == ("paris", "Paris")
    assert map_to_rows(["Paris", "THE", "xyz"], {"paris": 0, "The": 1}, "exact") == {"Paris": 0, "THE": 1}
    assert map_to_rows(["Paris", "THE"], {"paris": 0, "the": 1}, "lower") == {"Paris": 0, "THE": 1}
    ids, lens = pad_ids([[3, 4], [], [5]])
    assert ids.tolist() == [[3, 4], [0, 0], [5, 0]] and lens.tolist() == [2, 0, 1]
    print("text: tokenisers, case fallback and padding as intended")


def test_readers():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        with zipfile.ZipFile(d / "glove.zip", "w") as z:     # one malformed line, one duplicate
            z.writestr("glove.txt", "the 1 2\ncat 3 4\nbad 1\ncat 9 9\ndog 5 6\n")
        SOURCES["_test_text"] = dict(file="glove.zip", dim=2, cased=False)
        assert [w for w, _ in iter_table("_test_text", d)] == ["the", "cat", "cat", "dog"]
        cache_table("_test_text", {"cat", "dog"}, d / "t.npz", d)
        t = Table(d / "t.npz")
        assert t.words == ["cat", "dog"] and t.vectors.tolist() == [[3, 4], [5, 6]] and t.file_size == 3
        assert np.allclose(t.mean, [3, 4])                    # mean of all three kept vectors
        vecs = {"Hello": np.array([1, 2, 3], np.float32), "world": np.array([4, 5, 6], np.float32)}
        with gzip.open(d / "w2v.bin.gz", "wb") as f:
            f.write(b"2 3\n")
            for w, v in vecs.items():
                f.write(w.encode() + b" " + v.tobytes() + b"\n")
        SOURCES["_test_bin"] = dict(file="w2v.bin.gz", dim=3, cased=True, binary=True)
        got = dict(iter_table("_test_bin", d))
        assert got.keys() == vecs.keys() and all(np.array_equal(got[w], vecs[w]) for w in vecs)
        del SOURCES["_test_text"], SOURCES["_test_bin"]
    print("readers: text and word2vec binary formats, malformed lines and duplicates handled")


def toy_matrix(v=8, d=6, seed=0):
    m = torch.randn(v, d, generator=torch.Generator().manual_seed(seed))
    m[0] = 0
    return m


def encoders():
    yield "pool-mean", PoolEncoder(Embedder(toy_matrix(), False), "mean")
    yield "pool-max", PoolEncoder(Embedder(toy_matrix(), False, proj=5), "max")
    for out in ("last", "max", "mean"):
        torch.manual_seed(0)
        yield f"bilstm-{out}", RNNEncoder(Embedder(toy_matrix(), True), "lstm", 4, True, 1, out)
    torch.manual_seed(0)
    yield "gru-2layers", RNNEncoder(Embedder(toy_matrix(), False), "gru", 4, False, 2, "last")


def test_encoders():
    seqs = [[2, 3, 4], [5, 2], [7]]
    ids, lens = pad_ids(seqs)
    ids, lens = torch.as_tensor(ids), torch.as_tensor(lens)
    for name, enc in encoders():
        enc.eval()
        batched = enc(ids, lens)
        single = torch.cat([enc(torch.tensor([s]), torch.tensor([len(s)])) for s in seqs])
        assert torch.allclose(batched, single, atol=1e-6), name           # padding changes nothing
        wider = enc(torch.cat([ids, torch.zeros(3, 4, dtype=ids.dtype)], 1), lens)
        assert torch.allclose(batched, wider, atol=1e-6), name
        a, b = enc(torch.tensor([[2, 3, 4]]), torch.tensor([3])), enc(torch.tensor([[4, 3, 2]]), torch.tensor([3]))
        assert torch.allclose(a, b) == name.startswith("pool"), name        # only pooling ignores order
        empty = enc(torch.zeros(1, 2, dtype=ids.dtype), torch.tensor([0]))
        assert torch.isfinite(empty).all(), name
    print("encoders: padding-invariant, pooling order-blind, recurrent order-sensitive, empty texts finite")


def test_classifier():
    ids, lens = map(torch.as_tensor, pad_ids([[2, 3], [4]]))
    other, olens = map(torch.as_tensor, pad_ids([[5, 6, 7], [7, 2]]))
    shared = torch.zeros(2, 1, dtype=torch.long)
    hyp_only = PairClassifier(PoolEncoder(Embedder(toy_matrix(), False), "mean"), "hyp", 8, 1, "relu", 0.0).eval()
    assert torch.allclose(hyp_only(ids, lens, other, olens, shared), hyp_only(other, olens, other, olens, shared))
    full = PairClassifier(PoolEncoder(Embedder(toy_matrix(), False), "mean"), "full", 8, 1, "relu", 0.0, overlap_vocab=8).eval()
    with torch.no_grad():
        full.overlap.weight.normal_()
    base = full(ids, lens, other, olens, shared)
    assert not torch.allclose(base, full(ids, lens, other, olens, torch.tensor([[3], [0]])))
    assert torch.allclose(base[1], full(ids, lens, other, olens, torch.tensor([[3], [0]]))[1])   # pad id adds nothing
    print("classifier: hypothesis-only ignores the premise; overlap ids enter, padding does not")


def toy_data(n, rng, shuffle=False):
    """Label = which of three marker tokens the hypothesis contains; premises are noise."""
    prem = rng.integers(5, 40, (n, 6))
    hyp = rng.integers(5, 40, (n, 5))
    y = rng.integers(0, 3, n)
    hyp[:, 0] = y + 2
    y = rng.permutation(y) if shuffle else y
    t = lambda a: torch.as_tensor(a)
    return PairTensors(t(prem), t(np.full(n, 6)), t(np.arange(n)), t(hyp), t(np.full(n, 5)), torch.zeros(n, 1, dtype=torch.long), t(y))


def test_training():
    rng = np.random.default_rng(0)
    for shuffle, check in ((False, lambda a: a > 0.95), (True, lambda a: abs(a - 1 / 3) < 0.05)):
        torch.manual_seed(0)
        train, tune = toy_data(3000, rng, shuffle), toy_data(3000, rng, shuffle)
        model = PairClassifier(RNNEncoder(Embedder(toy_matrix(40, 8), True), "lstm", 8, True, 1, "max"), "full", 16, 1, "relu", 0.1)
        fit(model, train, np.arange(3000), tune, None, None, 0, 3e-3, 0.0, 64, 15, 3)
        acc = (predict_log_probs(model, tune, np.arange(3000)).argmax(1) == tune.y.numpy()).mean()
        assert check(acc), (shuffle, acc)
    print("training: learns a planted cue; shuffled labels stay at chance")


def test_posterior():
    """The correlated t with 3 runs of 5 folds: variance factor 1/15 + 1/4, so rho = 1/5."""
    rng = np.random.default_rng(0)
    a, b = rng.normal(.6, .01, 15), rng.normal(.61, .01, 15)
    t = posterior(a, b)
    d = b - a
    assert np.isclose(t.mean(), d.mean()) and t.args[0] == 14
    assert np.isclose(t.std() ** 2, d.var(ddof=1) * (1 / 15 + 1 / 4) * 14 / 12)
    print("posterior: correlated t statistics as expected")


if __name__ == "__main__":
    test_text()
    test_readers()
    test_encoders()
    test_classifier()
    test_training()
    test_posterior()
    test_splits(load_mnli()["train"])
    print("ALL PASSED")
