"""The three architectures compared: log-linear, one-hidden-layer MLP, Siamese MLP."""
import copy

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from .features import Blocks


def fit_loglinear(train: Blocks, y, C: float = 1.0, max_iter: int = 2000) -> LogisticRegression:
    """Softmax regression with L2 penalty."""
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter)
    return model.fit(train.flat(), y)


def to_torch(X: sp.csr_matrix, device) -> torch.Tensor:
    coo = X.tocoo()
    idx = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(coo.data), coo.shape,
                                   is_coalesced=True, device=device)


ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


def sparse_linear_init(rows: int, cols: int) -> nn.Parameter:
    return nn.Parameter(nn.init.xavier_uniform_(torch.empty(rows, cols)))


class FlatMLP(nn.Module):
    """Tutorial MLP: sparse input -> `depth` hidden layers with dropout -> 3 logits."""

    def __init__(self, dims: tuple[int, ...], hidden: int, dropout: float, depth: int = 1,
                 activation: str = "relu"):
        super().__init__()
        self.weight = sparse_linear_init(sum(dims), hidden)
        self.bias = nn.Parameter(torch.zeros(hidden))
        act = ACTIVATIONS[activation]
        layers = [act(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act(), nn.Dropout(dropout)]
        self.head = nn.Sequential(*layers, nn.Linear(hidden, 3))

    def forward(self, blocks: list[torch.Tensor]) -> torch.Tensor:
        h = sum_blocks(blocks, self.weight) + self.bias
        return self.head(h)


def sum_blocks(blocks: list[torch.Tensor], weight: torch.Tensor) -> torch.Tensor:
    """Multiply side-by-side sparse blocks by the matching slices of one weight matrix."""
    out, start = 0, 0
    for b in blocks:
        out = out + torch.sparse.mm(b, weight[start:start + b.shape[1]])
        start += b.shape[1]
    return out


class SiameseMLP(nn.Module):
    """Same embedding matrix for both sentences, then [u; v; |u-v|; u*v] -> MLP -> 3 logits.

    Comparison layer from Conneau et al. (2017). The flat MLP has to discover from data
    that column j of the premise block and column j of the hypothesis block are the same
    token; this architecture builds that knowledge in. Requires premise and hypothesis blocks.
    """

    def __init__(self, vocab: int, extra_dim: int, hidden: int, dropout: float, activation: str = "relu"):
        super().__init__()
        self.embed = sparse_linear_init(vocab, hidden)
        self.extra = sparse_linear_init(extra_dim, hidden) if extra_dim else None
        in_dim = hidden * (4 + bool(extra_dim))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), ACTIVATIONS[activation](),
                                  nn.Dropout(dropout), nn.Linear(hidden, 3))

    def forward(self, blocks: list[torch.Tensor]) -> torch.Tensor:
        u = torch.sparse.mm(blocks[0], self.embed)
        v = torch.sparse.mm(blocks[1], self.embed)
        parts = [u, v, (u - v).abs(), u * v]
        if self.extra is not None:
            parts.append(torch.sparse.mm(blocks[2], self.extra))
        return self.head(torch.cat(parts, dim=1))


def make_network(arch: str, sample: Blocks, hidden: int = 512, dropout: float = 0.3, activation: str = "relu") -> nn.Module:
    parts = [b for b in (sample.premise, sample.hypothesis, sample.extra) if b is not None]
    dims = tuple(b.shape[1] for b in parts)
    if arch == "mlp":
        return FlatMLP(dims, hidden, dropout, 1, activation)
    if arch == "mlp2":
        return FlatMLP(dims, hidden, dropout, 2, activation)
    if arch == "siamese":
        extra_dim = sample.extra.shape[1] if sample.extra is not None else 0
        return SiameseMLP(sample.premise.shape[1], extra_dim, hidden, dropout, activation)
    raise ValueError(arch)


def _blocks(b: Blocks) -> list[sp.csr_matrix]:
    return [m for m in (b.premise, b.hypothesis, b.extra) if m is not None]


@torch.no_grad()
def predict_log_probs(model: nn.Module, data: Blocks, device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    parts = _blocks(data)
    n = parts[0].shape[0]
    out = []
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(start + batch_size, n))
        out.append(F.log_softmax(model([to_torch(m[idx], device) for m in parts]), -1).cpu().numpy())
    return np.concatenate(out)


def nll_acc(log_probs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    return float(-log_probs[np.arange(len(y)), y].mean()), float((log_probs.argmax(1) == y).mean())


def fit_network(arch, train: Blocks, y_train, tune: Blocks, y_tune, val: Blocks | None, y_val,
                seed: int, device, hidden=512, dropout=0.3, activation="relu", lr=3e-4, weight_decay=0.3,
                batch_size=256, max_epochs=20, patience=3):
    """Train with AdamW; keep the epoch with the lowest loss on the tuning set.

    If `val` is given, its scores are also recorded per epoch, to measure how optimistic
    choosing the epoch on the validation fold itself would have been.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_network(arch, train, hidden, dropout, activation).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    parts, n = _blocks(train), len(y_train)
    train_probe = train.take(np.arange(min(20000, n)))   # to plot the train curve; uses no RNG
    y_t = torch.as_tensor(y_train, device=device)
    best, best_state, best_epoch, history = np.inf, None, 0, []
    for epoch in range(max_epochs):
        model.train()
        for idx in np.array_split(rng.permutation(n), max(1, n // batch_size)):
            optimiser.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model([to_torch(m[idx], device) for m in parts]), y_t[idx])
            loss.backward()
            optimiser.step()
        tune_nll, tune_acc = nll_acc(predict_log_probs(model, tune, device), y_tune)
        train_nll, train_acc = nll_acc(predict_log_probs(model, train_probe, device), y_train[:20000])
        record = {"epoch": epoch, "train_nll": train_nll, "train_acc": train_acc,
                  "tune_nll": tune_nll, "tune_acc": tune_acc}
        if val is not None:
            record["val_nll"], record["val_acc"] = nll_acc(predict_log_probs(model, val, device), y_val)
        history.append(record)
        if tune_nll < best:
            best, best_epoch = tune_nll, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "history": history}
