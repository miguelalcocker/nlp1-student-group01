"""Run the experiments. Usage: python run.py stage [stage ...]

Each fit writes results/fits/<key>.json, where the key is a hash of the job; a stage is a list
of jobs, and a job shared by two stages runs once. Finished fits are skipped, so a run can be
interrupted and resumed. Several copies of this command can run at once: each claims a fit
with a lock file before starting it.
"""
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from nli import inputs
from nli.data import carve_tuning_set, grouped_kfold, load_mnli, subsample_premises
from nli.models import Embedder, PairClassifier, PoolEncoder, RNNEncoder
from nli.train import PairTensors, fit, nll_acc, predict_log_probs

RESULTS, CACHE = Path("results"), Path("cache")
SEEDS, K = (0, 1, 2), 5

# Settings shared by every model unless a job changes them. The optimiser settings start from
# project 1's (AdamW, lr 3e-4, weight decay 0.3, batch 256, early stopping on the tuning NLL with
# patience 3). max_epochs is a safety cap only: a first run with a cap of 20 stopped many LSTM fits
# while they were still improving (kept in results/fits_cap20), so the cap is now 60 and every
# result records whether it was reached. The learning rate of each encoder is chosen by stage `lr`.
# emb_std: every table is multiplied by one number so that its coordinates have this standard
# deviation (GloVe 840B's is 0.405; counter-fitted and Numberbatch vectors have norm 1, about
# 0.058). Cosines are unchanged; without it, frozen tables would differ in scale, not only in
# content. Embeddings trained from scratch start from N(0, emb_std).
COMMON = dict(tok="nltk", case="auto", emb="glove840B", emb_std=0.4, trainable=False, proj=0, features="full",
              overlap=False, head_hidden=512, head_depth=1, activation="relu", dropout=0.3,
              lr=3e-4, weight_decay=0.3, batch_size=256, max_epochs=60, patience=3)
POOL = dict(COMMON, enc="pool", pooling="mean")
RNN = dict(COMMON, enc="rnn", cell="lstm", hidden=256, bidirectional=True, layers=1, output="max", word_dropout=0.1)
RNN_KEYS = ("cell", "hidden", "bidirectional", "layers", "output", "word_dropout")

PRETRAINED = ("glove2024-50", "glove6B-50", "glove6B-100", "glove6B-200", "glove6B-300",
              "glove840B", "fasttext", "word2vec", "counterfit", "numberbatch")
PRETRAINED_300 = ("glove6B-300", "glove840B", "fasttext", "word2vec", "counterfit", "numberbatch")


def cfg(base: dict, **changes) -> dict:
    out = dict(base, **changes)
    unknown = set(changes) - set(base)
    if unknown:
        raise ValueError(f"unknown settings {unknown}")
    return out


def tune(c: dict) -> list[dict]:
    """Train on the CV pool, early-stop and evaluate on the tuning set; one fit per seed."""
    return [dict(split="tune", seed=s, fold=0, frac=1.0, cfg=c) for s in SEEDS]


def cv(c: dict, seeds=SEEDS) -> list[dict]:
    return [dict(split="grouped", seed=s, fold=f, frac=1.0, cfg=c) for s in seeds for f in range(K)]


def stages() -> dict[str, list[dict]]:
    # Learning rate of each encoder, on its base setting, before any other search.
    lr = [j for x in (3e-4, 1e-3, 2e-3) for j in tune(cfg(RNN, lr=x))]
    lr += [j for x in (3e-4, 1e-3, 3e-3) for j in tune(cfg(POOL, lr=x))]

    # Q1-Q3: which table? Embed-and-pool with every table; the recurrent model with the 300d ones.
    embeddings = [j for e in PRETRAINED + ("random",) for j in tune(cfg(POOL, emb=e, trainable=e == "random"))]
    embeddings += [j for e in PRETRAINED_300 + ("random",) for j in tune(cfg(RNN, emb=e, trainable=e == "random"))]

    # Control: the same tables at their original scale.
    embeddings += [j for e in ("glove840B", "word2vec", "counterfit", "numberbatch") for j in tune(cfg(POOL, emb=e, emb_std=None))]

    # Q1: tokenisation and case, for the two cased Common Crawl tables.
    tokens = []
    for e in ("glove840B", "fasttext"):
        for tok, case in (("nltk", "exact"), ("nltk", "lower"), ("parse", "exact"), ("parse", "lower"),
                          ("split", "exact"), ("tutorial-bow", "lower")):
            tokens += tune(cfg(POOL, emb=e, tok=tok, case=case))

    # Q3: embed-and-pool design. One change at a time around POOL, then a grid of lr and decay.
    pool_changes = [{}, dict(pooling="sum"), dict(pooling="max"), dict(pooling="meanmax"), dict(features="concat"),
                    dict(trainable=True), dict(proj=512), dict(proj=512, trainable=True), dict(overlap=True),
                    dict(head_hidden=128), dict(head_hidden=1024), dict(head_depth=2), dict(dropout=0.1), dict(dropout=0.5),
                    dict(activation="tanh"), dict(activation="gelu"), dict(batch_size=64),
                    # tutorial 2's embed-and-pool settings, with our pair features
                    dict(tok="tutorial-bow", case="lower", head_hidden=50, head_depth=2, activation="tanh", dropout=0.5,
                         lr=1e-3, weight_decay=1e-4, batch_size=64)]
    pool_changes += [dict(lr=x, weight_decay=wd) for x in (POOL["lr"] / 3, POOL["lr"], POOL["lr"] * 3)
                     for wd in (0.01, 0.1, 0.3, 1.0) if (x, wd) != (POOL["lr"], POOL["weight_decay"])]
    pool = [j for ch in pool_changes for j in tune(cfg(POOL, **ch))]

    # Q4: recurrent design, one change at a time around RNN.
    rnn_changes = [{}, dict(cell="gru"), dict(bidirectional=False), dict(hidden=128), dict(hidden=512),
                   dict(output="last"), dict(output="mean"), dict(layers=2), dict(word_dropout=0.0), dict(word_dropout=0.25),
                   dict(trainable=True), dict(emb="random", trainable=True), dict(features="concat"), dict(overlap=True),
                   dict(dropout=0.1), dict(dropout=0.5), dict(lr=RNN["lr"] / 3), dict(lr=RNN["lr"] * 3),
                   dict(weight_decay=0.01), dict(weight_decay=0.1), dict(weight_decay=1.0),
                   # tutorial 2's LSTM settings (unidirectional, 20 units, last state), with our pair features
                   dict(bidirectional=False, hidden=20, output="last", word_dropout=0.0, head_hidden=50, head_depth=2,
                        activation="tanh", dropout=0.1, lr=1e-3, weight_decay=1e-4, batch_size=64)]
    rnn = [j for ch in rnn_changes for j in tune(cfg(RNN, **ch))]

    # Step 2: the changes that won one at a time need not add up, and fine-tuning the table
    # turns embed-and-pool from underfitting into overfitting, so regularisation is searched
    # again around the new bases. `nltk-not` spells out "n't": fastText has no vector for it;
    # Numberbatch has one, so there it tests whether the vector of "not" serves better.
    pool_step2 = [{}, dict(emb="numberbatch"), dict(emb="numberbatch", tok="nltk-not"), dict(emb="fasttext"),
                  dict(emb="fasttext", tok="nltk-not"), dict(emb="random"), dict(dropout=0.1), dict(dropout=0.5),
                  dict(weight_decay=0.01), dict(weight_decay=0.1), dict(weight_decay=1.0),
                  dict(lr=POOL2["lr"] / 3), dict(lr=POOL2["lr"] * 3), dict(head_depth=2), dict(overlap=True)]
    rnn_step2 = [dict(weight_decay=0.1, hidden=512, layers=2), {}, dict(tok="nltk-not"), dict(weight_decay=0.1),
                 dict(hidden=512), dict(layers=2), dict(cell="gru"), dict(word_dropout=0.0),
                 dict(emb="fasttext", tok="nltk-not")]
    confirm = [j for ch in rnn_step2 for j in tune(cfg(RNN2, **ch))]
    confirm += [j for ch in pool_step2 for j in tune(cfg(POOL2, **ch))]

    # Final comparisons: 5-fold CV repeated 3 times, on project 1's folds (same seeds, same code).
    main = [j for c in MAIN.values() for j in cv(c)]

    # Q4: data size. Fold 0 of each repetition, with a fraction of the training premises;
    # the full-data point is the matching fit of `main`.
    datasize = [dict(split="grouped", seed=s, fold=0, frac=f, cfg=MAIN[name])
                for name in ("pool", "pool-random", "rnn", "rnn-random") for f in FRACTIONS + (1.0,) for s in SEEDS]
    return dict(lr=lr, embeddings=embeddings, tokens=tokens, pool=pool, rnn=rnn, confirm=confirm, main=main,
                datasize=datasize)


# Bases of step 2, from the one-change searches: fine-tuning plus a 512d projection for
# embed-and-pool (+5.8 pp), Numberbatch for the BiLSTM (+1.4 pp).
POOL2 = cfg(POOL, proj=512, trainable=True)
RNN2 = cfg(RNN, emb="numberbatch")

# Chosen settings, from step 2 (tuning set, 3 seeds). Embed-and-pool: fastText with "n't"
# spelt out, 65.84 +- 0.10 (+0.46 over the step-2 base, SD of the paired difference 0.09);
# the other changes gained less than 0.3 pp each and were not combined untested. BiLSTM:
# Numberbatch with weight decay 0.1, 73.31 +- 0.08; adding 512 units and 2 layers gave
# 73.42 +- 0.32, a tie within seed noise, so the simpler model is kept.
BEST_POOL = cfg(POOL2, emb="fasttext", tok="nltk-not")
BEST_RNN = cfg(RNN2, weight_decay=0.1)
MAIN = {    # recurrent systems first: they take longest, so workers end together
    "rnn": BEST_RNN,
    "rnn-random": cfg(BEST_RNN, emb="random", trainable=True),
    "rnn-hyp": cfg(BEST_RNN, features="hyp"),
    "pool": BEST_POOL,
    "pool-random": cfg(BEST_POOL, emb="random", trainable=True),
    "pool-hyp": cfg(BEST_POOL, features="hyp"),
}
FRACTIONS = (0.01, 0.03, 0.1, 0.3)


def canonical(c: dict) -> dict:
    c = dict(c)
    if c["enc"] == "pool":
        for k in RNN_KEYS:
            c.pop(k, None)
    return c


def key(job: dict) -> str:
    job = dict(job, cfg=canonical(job["cfg"]))
    return hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]


class Inputs:
    """Keeps the arrays of one input choice on the device at a time."""

    def __init__(self, data, pool, tune_idx, device):
        self.data, self.pool, self.tune_idx, self.device = data, pool, tune_idx, device
        self.current, self.splits, self.meta = None, None, None

    def get(self, c: dict):
        choice = (c["tok"], c["emb"], inputs.resolve_case(c["emb"], c["case"]))
        if choice != self.current:
            self.splits = self.meta = None
            torch.cuda.empty_cache()
            arrays = inputs.build(c["tok"], c["emb"], c["case"], self.data, self.pool, CACHE)
            ids = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int32), device=self.device)
            t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=self.device)
            self.splits = {}
            for s in inputs.SPLITS:
                y = self.data[s].label.to_numpy()
                self.splits[s] = PairTensors(ids(arrays[f"{s}/prem"]), t(arrays[f"{s}/prem_len"]), t(arrays[f"{s}/pair_prem"]),
                                             ids(arrays[f"{s}/hyp"]), t(arrays[f"{s}/hyp_len"]), ids(arrays[f"{s}/shared"]), t(y))
            self.meta = {"n_vocab": int(arrays["n_vocab"]),
                         "matrix": torch.as_tensor(arrays["matrix"]) if "matrix" in arrays else None}
            self.current = choice
        return self.splits, self.meta


def subset(pt: PairTensors, idx: np.ndarray) -> PairTensors:
    i = torch.as_tensor(idx, device=pt.y.device)
    return PairTensors(pt.prem, pt.prem_len, pt.pair_prem[i], pt.hyp[i], pt.hyp_len[i], pt.shared[i], pt.y[i])


def build_model(c: dict, meta: dict, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    if c["emb"] == "random":
        matrix = torch.randn(meta["n_vocab"], 300) * c["emb_std"]
        matrix[0] = 0
    else:
        matrix = meta["matrix"].clone()
        if c["emb_std"]:
            matrix *= c["emb_std"] / matrix[1:].std()
    embedder = Embedder(matrix, c["trainable"], c["proj"], c.get("word_dropout", 0.0))
    if c["enc"] == "pool":
        encoder = PoolEncoder(embedder, c["pooling"])
    else:
        encoder = RNNEncoder(embedder, c["cell"], c["hidden"], c["bidirectional"], c["layers"], c["output"])
    return PairClassifier(encoder, c["features"], c["head_hidden"], c["head_depth"], c["activation"], c["dropout"],
                          meta["n_vocab"] if c["overlap"] else 0)


def run_job(job: dict, ins: Inputs, save_probs: bool) -> dict:
    c = job["cfg"]
    splits, meta = ins.get(c)
    train = splits["train"]
    groups = ins.data["train"].promptID.to_numpy()
    if job["split"] == "tune":
        tr_idx, val_idx = ins.pool, ins.tune_idx
    else:
        fold = grouped_kfold(groups[ins.pool], K, job["seed"])[job["fold"]]
        tr_idx, val_idx = ins.pool[fold.train], ins.pool[fold.val]
    tr_idx = subsample_premises(groups, tr_idx, job["frac"], job["seed"])
    tune_pt = subset(train, ins.tune_idx)
    model = build_model(c, meta, job["seed"]).to(ins.device)
    start = time.time()
    on_fold = job["split"] != "tune"     # also track the fold, to measure early-stopping optimism
    info = fit(model, train, tr_idx, tune_pt, train if on_fold else None, val_idx if on_fold else None,
               job["seed"], c["lr"], c["weight_decay"], c["batch_size"], c["max_epochs"], c["patience"])
    y = train.y.cpu().numpy()
    lp = predict_log_probs(model, train, val_idx)
    out = dict(job=dict(job, cfg=canonical(c)), **info, seconds=round(time.time() - start, 1), n_train=len(tr_idx),
               n_params=sum(p.numel() for p in model.parameters() if p.requires_grad))
    out["capped"] = info["epochs_run"] >= c["max_epochs"]
    out["nll"], out["accuracy"] = nll_acc(lp, y[val_idx])
    out["macro_f1"] = float(f1_score(y[val_idx], lp.argmax(1), average="macro"))
    out["train_accuracy"] = nll_acc(predict_log_probs(model, train, tr_idx[:20000]), y[tr_idx[:20000]])[1]
    if save_probs:
        # Probabilities for error analysis (fold) and the final report (validation sets); no selection uses them.
        probs = {"fold_idx": val_idx.astype(np.int32), "fold": np.exp(lp).astype(np.float16)}
        for s in ("validation_matched", "validation_mismatched"):
            pt = splits[s]
            probs[s] = np.exp(predict_log_probs(model, pt, np.arange(len(pt.y)))).astype(np.float16)
        d = RESULTS / "probs"
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"{key(job)}.npz", **probs)
    return out


def claim(lock: Path) -> bool:
    """Create the lock file, or remove it if the worker that made it on this machine has died."""
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{socket.gethostname()} {os.getpid()}".encode())
            os.close(fd)
            return True
        except FileExistsError:
            host, pid = (lock.read_text().split() + ["", "0"])[:2]
            if host != socket.gethostname():
                return False
            try:
                os.kill(int(pid), 0)
                return False
            except (ProcessLookupError, ValueError):
                lock.unlink(missing_ok=True)
    return False


SAVE_PROBS = {"main"}


def main():
    names = sys.argv[1:]
    all_stages = stages()
    if not names or any(n not in all_stages for n in names):
        sys.exit(f"usage: python run.py stage [stage ...]; stages: {', '.join(all_stages)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    data = load_mnli()
    pool, tune_idx = carve_tuning_set(data["train"].promptID.to_numpy())
    ins = Inputs(data, pool, tune_idx, device)
    folder = RESULTS / "fits"
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        jobs = all_stages[name]
        done = sum((folder / f"{key(j)}.json").exists() for j in jobs)
        print(f"{name}: {len(jobs) - done} fits to run, {done} already done", flush=True)
        for job in jobs:
            out_path, lock = folder / f"{key(job)}.json", folder / f"{key(job)}.lock"
            if out_path.exists() or not claim(lock):
                continue
            try:
                result = None
                for attempt in range(20):
                    # The GPU is shared with other users: if it runs out of memory, wait and refit
                    # from scratch (same seed, so the same fit) instead of losing the worker.
                    try:
                        result = run_job(job, ins, name in SAVE_PROBS)
                        break
                    except torch.OutOfMemoryError:
                        print(f"{time.strftime('%d %H:%M')} out of GPU memory on {key(job)}, retry {attempt + 1} in 3 min", flush=True)
                        torch.cuda.empty_cache()
                        time.sleep(180)
                if result is None:
                    continue
                tmp = out_path.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_text(json.dumps(result))
                os.replace(tmp, out_path)
                c = job["cfg"]
                changed = {k: v for k, v in canonical(c).items() if (POOL if c["enc"] == "pool" else RNN).get(k) != v}
                print(f"{time.strftime('%d %H:%M')} {name} {job['split']} s{job['seed']} f{job['fold']} frac {job['frac']} "
                      f"{c['enc']} {c['emb']} {changed} acc {result['accuracy']:.4f} "
                      f"epochs {result['epochs_run']} {result['seconds'] / 60:.1f} min", flush=True)
            finally:
                lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
