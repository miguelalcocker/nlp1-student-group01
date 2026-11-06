"""Sanity checks on real MNLI data. Usage: python test_pipeline.py"""
import numpy as np
import scipy.sparse as sp
import torch

from nli.data import carve_tuning_set, grouped_kfold, load_mnli
from nli.features import Blocks, build_representation, fit_tokenizer, token_counts
from nli.models import fit_loglinear, fit_network, make_network, nll_acc, predict_log_probs, to_torch
from nli.stats import posterior


def test_splits(train):
    groups = train.promptID.to_numpy()
    pool, tune = carve_tuning_set(groups)
    assert len(np.intersect1d(pool, tune)) == 0 and len(pool) + len(tune) == len(train)
    assert len(np.intersect1d(groups[pool], groups[tune])) == 0
    for seed in (0, 1, 2):
        folds = grouped_kfold(groups[pool], 5, seed)
        assert sorted(np.concatenate([f.val for f in folds]).tolist()) == list(range(len(pool)))
        for f in folds:
            assert len(np.intersect1d(groups[pool][f.train], groups[pool][f.val])) == 0
    print("splits: partition holds, no premise crosses train/validation/tuning")


def test_labels(data):
    for name, df in data.items():
        print(name, len(df), df.label.value_counts().sort_index().to_dict())
    train = data["train"]
    assert train.groupby("promptID").label.nunique().mean() > 2.9


def test_overlap():
    P = sp.csr_matrix(np.array([[2., 0, 1, 0]]))
    H = sp.csr_matrix(np.array([[1., 1, 0, 0]]))
    assert (build_representation("overlap", P, H).extra.toarray() == [[1, 0, 0, 0]]).all()
    assert (build_representation("and", P, H).flat().toarray() == [[1, 0, 0, 0]]).all()
    assert build_representation("hyp-and", P, H).flat().shape == (1, 8)
    print("overlap: block is the elementwise AND of the indicators")


def test_sparse_forward(P, H):
    """The sparse forward pass must match a dense computation of the same maths."""
    idx = np.random.default_rng(0).permutation(P.shape[0])[:64]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for rep in ("bow", "overlap"):
        b = build_representation(rep, P, H).take(idx)
        parts = [m for m in (b.premise, b.hypothesis, b.extra) if m is not None]
        sparse = [to_torch(m, device) for m in parts]
        for m, t in zip(parts, sparse):
            assert m.has_sorted_indices
            assert np.allclose(t.to_dense().cpu().numpy(), m.toarray())
        for arch in ("mlp", "siamese"):
            net = make_network(arch, b, 32, 0.0).to(device).eval()
            with torch.no_grad():
                out = net(sparse)
                dense = [t.to_dense() for t in sparse]
                if arch == "mlp":
                    h = torch.cat(dense, 1) @ net.weight + net.bias
                    want = net.head(h)
                else:
                    u, v = dense[0] @ net.embed, dense[1] @ net.embed
                    z = [u, v, (u - v).abs(), u * v]
                    if net.extra is not None:
                        z.append(dense[2] @ net.extra)
                    want = net.head(torch.cat(z, 1))
            assert torch.allclose(out, want, atol=1e-4)
    print("forward: sparse and dense computations agree")


def test_shuffled_labels(P, H, train):
    """With shuffled labels nothing can be learned: accuracy must sit at chance."""
    groups = train.promptID.to_numpy()
    pool, tune = carve_tuning_set(groups)
    fold = grouped_kfold(groups[pool], 5, 0)[0]
    tr, val = pool[fold.train][:50000], pool[fold.val]
    rep = build_representation("overlap", P, H)
    y = np.random.default_rng(0).permutation(train.label.to_numpy())
    model = fit_loglinear(rep.take(tr), y[tr], max_iter=100)
    acc = (model.predict(rep.take(val).flat()) == y[val]).mean()
    assert abs(acc - 1 / 3) < 0.015, acc
    print(f"shuffled labels: log-linear {acc:.4f}, chance 0.3333")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ = fit_network("siamese", rep.take(tr), y[tr], rep.take(tune), y[tune], None, None, 0, device, max_epochs=3)
    _, acc = nll_acc(predict_log_probs(net, rep.take(val), device), y[val])
    assert abs(acc - 1 / 3) < 0.015, acc
    print(f"shuffled labels: Siamese {acc:.4f}, chance 0.3333")


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
    data = load_mnli()
    train = data["train"]
    test_splits(train)
    test_labels(data)
    test_overlap()
    test_posterior()
    pool, _ = carve_tuning_set(train.promptID.to_numpy())
    tok = fit_tokenizer(list(train.iloc[pool].premise.unique()) + list(train.iloc[pool].hypothesis), 2000)
    P, H = token_counts(tok, list(train.premise)), token_counts(tok, list(train.hypothesis))
    test_sparse_forward(P, H)
    test_shuffled_labels(P, H, train)
    print("ALL PASSED")
