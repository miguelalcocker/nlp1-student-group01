"""Batching on the GPU, the training loop with early stopping, and prediction."""
import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PairTensors:
    """One split on the device. Premises are stored once; pairs point to them."""
    prem: torch.Tensor        # [n_premises, Lp] token ids, 0 = pad
    prem_len: torch.Tensor
    pair_prem: torch.Tensor   # [n_pairs] row in `prem`
    hyp: torch.Tensor         # [n_pairs, Lh]
    hyp_len: torch.Tensor
    shared: torch.Tensor      # [n_pairs, Lo] ids of word types in both sentences, 0 = pad
    y: torch.Tensor

    def batch(self, idx: torch.Tensor):
        rows = self.pair_prem[idx]
        pl, hl = self.prem_len[rows], self.hyp_len[idx]
        p = self.prem[rows, :max(1, int(pl.max()))]
        h = self.hyp[idx, :max(1, int(hl.max()))]
        return p, pl, h, hl, self.shared[idx], self.y[idx]

    def lengths(self) -> np.ndarray:
        return (self.prem_len[self.pair_prem] + self.hyp_len).cpu().numpy()


def length_batches(lengths: np.ndarray, idx: np.ndarray, batch_size: int, rng) -> list[np.ndarray]:
    """Shuffle, then sort by length within chunks of 50 batches so that batches pad little."""
    idx = rng.permutation(idx)
    out = []
    for start in range(0, len(idx), 50 * batch_size):
        chunk = idx[start:start + 50 * batch_size]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        out += [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
    return [out[i] for i in rng.permutation(len(out))]


@torch.no_grad()
def predict_log_probs(model, data: PairTensors, idx: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    order = idx[np.argsort(data.lengths()[idx], kind="stable")]
    out = np.empty((len(idx), 3), dtype=np.float32)
    pos = {int(i): k for k, i in enumerate(idx)} if len(idx) < len(data.y) else None
    device = data.y.device
    for start in range(0, len(order), batch_size):
        b = order[start:start + batch_size]
        p, pl, h, hl, s, _ = data.batch(torch.as_tensor(b, device=device))
        lp = F.log_softmax(model(p, pl, h, hl, s).float(), -1).cpu().numpy()
        rows = b if pos is None else [pos[int(i)] for i in b]
        out[rows] = lp
    return out


def nll_acc(log_probs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    return float(-log_probs[np.arange(len(y)), y].mean()), float((log_probs.argmax(1) == y).mean())


def fit(model, train: PairTensors, train_idx, tune: PairTensors, val: PairTensors | None, val_idx,
        seed: int, lr: float, weight_decay: float, batch_size: int, max_epochs: int, patience: int):
    """AdamW; keep the epoch with the lowest loss on the tuning set (as in project 1).

    Weight decay is not applied to embedding tables: on a fine-tuned pretrained table it
    would pull the vectors towards zero rather than regularise the task layers.
    """
    rng = np.random.default_rng(seed)
    table_params = [p for n, p in model.named_parameters() if n.endswith("table.weight") and p.requires_grad]
    others = [p for n, p in model.named_parameters() if not n.endswith("table.weight") and p.requires_grad]
    optimiser = torch.optim.AdamW([{"params": others, "weight_decay": weight_decay},
                                   {"params": table_params, "weight_decay": 0.0}], lr=lr)
    lengths = train.lengths()
    probe = train_idx[:20000]
    y_train, y_tune = train.y.cpu().numpy(), tune.y.cpu().numpy()
    tune_idx = np.arange(len(y_tune))
    y_val = val.y.cpu().numpy() if val is not None else None
    best, best_state, best_epoch, history = np.inf, None, 0, []
    for epoch in range(max_epochs):
        start = time.time()
        model.train()
        for b in length_batches(lengths, train_idx, batch_size, rng):
            p, pl, h, hl, s, y = train.batch(torch.as_tensor(b, device=train.y.device))
            optimiser.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(p, pl, h, hl, s), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
        record = {"epoch": epoch, "seconds": round(time.time() - start, 1)}
        record["train_nll"], record["train_acc"] = nll_acc(predict_log_probs(model, train, probe), y_train[probe])
        record["tune_nll"], record["tune_acc"] = nll_acc(predict_log_probs(model, tune, tune_idx), y_tune)
        if val is not None:
            record["val_nll"], record["val_acc"] = nll_acc(predict_log_probs(model, val, val_idx), y_val[val_idx])
        history.append(record)
        if record["tune_nll"] < best:
            best, best_epoch = record["tune_nll"], epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "epochs_run": len(history), "history": history}
