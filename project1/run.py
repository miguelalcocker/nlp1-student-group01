"""Run the experiments. Usage: python run.py [stage]    (no stage: run all of them)

Each fit appends one JSON line to results/<stage>.jsonl. Finished fits are skipped,
so a run can be interrupted and resumed.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from nli.data import carve_tuning_set, grouped_kfold, load_mnli, random_kfold
from nli.features import build_representation, fit_tokenizer, token_counts
from nli.models import fit_loglinear, fit_network, nll_acc, predict_log_probs

RESULTS, CACHE = Path("results"), Path("cache")
BASE_TOK = dict(vocab=2000, kind="bpe", lowercase=False)
FIRST_GUESS = dict(lr=1e-3, weight_decay=1e-4)
SEEDS, K = (0, 1, 2), 5
REPS = ("ind", "bow", "overlap")
ARCHS = ("loglinear", "mlp", "siamese")
NEURAL = ("mlp", "siamese")
HELD_OUT = (("overlap", "siamese"), ("hyp-only", "mlp"))    # systems reported on the validation sets


def tune_job(rep="bow", arch="loglinear", tok=BASE_TOK, **params):
    """Train on the CV pool, evaluate on the tuning set."""
    return dict(split="tune", rep=rep, arch=arch, tok=tok, params=params, seed=0, fold=0)


def cv_jobs(reps, archs, split="grouped", seeds=SEEDS):
    return [dict(split=split, rep=r, arch=a, tok=BASE_TOK, params={}, seed=s, fold=f)
            for r in reps for a in archs for s in seeds for f in range(K)]


def bpe(vocab, lowercase=False):
    return dict(vocab=vocab, kind="bpe", lowercase=lowercase)


def stages():
    tokeniser = [tune_job(tok=bpe(v)) for v in (250, 1000, 5000, 10000, 30000)]
    tokeniser += [tune_job(tok=bpe(v, lowercase=True)) for v in (1000, 10000)]
    tokeniser += [tune_job(tok=dict(vocab=v, kind="word", lowercase=False)) for v in (1000, 10000)]
    tokeniser += [tune_job(arch="mlp", tok=bpe(v)) for v in (1000, 2000, 5000, 10000, 30000)]
    tokeniser += [tune_job(arch="siamese", tok=bpe(v)) for v in (1000, 2000, 5000)]

    representations = [tune_job(rep=r, arch=a) for a in ("loglinear", "mlp")
                       for r in ("bow-concat", "ind", "bow", "tfidf", "overlap", "cross")]

    # Step 1: one group of settings at a time, from a first guess.
    search = [tune_job(rep=r, C=c) for r in REPS for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)]
    for r in REPS:
        for a in NEURAL:
            search += [tune_job(r, a, lr=lr, weight_decay=wd)
                       for lr in (1e-3, 3e-3) for wd in (1e-4, 1e-2, 1e-1)]
            search += [tune_job(r, a, lr=3e-4, weight_decay=wd) for wd in (1e-4, 1e-2)]
            # Step 2: step 1 picked the edge of its grid (lowest lr, highest decay), so extend it.
            search += [tune_job(r, a, lr=lr, weight_decay=wd)
                       for lr in (1e-4, 3e-4) for wd in (0.1, 0.3, 1.0)]
            # Step 3: is the chosen setting a local optimum? Change one thing at a time.
            search += [tune_job(r, a, **p) for p in (
                {}, dict(hidden=128), dict(dropout=0.1), dict(dropout=0.5),
                dict(activation="tanh"), dict(activation="gelu"))]
        search.append(tune_job(r, "mlp2"))

    controls = cv_jobs(("prem-only", "hyp-only", "bow-concat", "and", "hyp-and"), ("loglinear", "mlp"))
    controls += cv_jobs(("prem-only", "hyp-only", "bow"), ("loglinear", "mlp"), split="random", seeds=(0,))

    return dict(tokeniser=tokeniser, representations=representations, search=search,
                main=cv_jobs(REPS, ARCHS), controls=controls)


def encoded(tok_cfg, data, pool, _last={}):
    """Token counts of all splits under one tokeniser, fitted on the CV pool only."""
    path = CACHE / f"{tok_cfg['kind']}_{tok_cfg['vocab']}_{'lc' if tok_cfg['lowercase'] else 'cs'}.pkl"
    if path not in _last:
        _last.clear()    # keep one tokeniser in memory at a time
        if not path.exists():
            train = data["train"].iloc[pool]
            corpus = list(train.premise.unique()) + list(train.hypothesis)
            tok = fit_tokenizer(corpus, tok_cfg["vocab"], tok_cfg["kind"], tok_cfg["lowercase"])
            enc = {s: (token_counts(tok, list(df.premise)), token_counts(tok, list(df.hypothesis)))
                   for s, df in data.items()}
            enc["vocab"] = tok.get_vocab()
            CACHE.mkdir(exist_ok=True)
            path.write_bytes(pickle.dumps(enc))
        _last[path] = pickle.loads(path.read_bytes())
    return _last[path]


def split_indices(job, pool, tune, groups):
    if job["split"] == "tune":
        return pool, tune
    folds = grouped_kfold(groups[pool], K, job["seed"]) if job["split"] == "grouped" \
        else random_kfold(len(pool), K, job["seed"])
    fold = folds[job["fold"]]
    return pool[fold.train], pool[fold.val]


def run_job(job, data, pool, tune):
    enc = encoded(job["tok"], data, pool)
    y = data["train"].label.to_numpy()
    rep = build_representation(job["rep"], *enc["train"])
    tr_idx, val_idx = split_indices(job, pool, tune, data["train"].promptID.to_numpy())
    train, val = rep.take(tr_idx), rep.take(val_idx)
    out = dict(job=job)

    if job["arch"] == "loglinear":
        model = fit_loglinear(train, y[tr_idx], **job["params"])
        predict = lambda b: model.predict_log_proba(b.flat())
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        on_fold = job["split"] != "tune"    # also track the fold, to measure early-stopping optimism
        net, info = fit_network(job["arch"], train, y[tr_idx], rep.take(tune), y[tune],
                                val if on_fold else None, y[val_idx] if on_fold else None,
                                job["seed"], device, **job["params"])
        out.update(info)
        predict = lambda b: predict_log_probs(net, b, device)

    log_probs = predict(val)
    out["nll"], out["accuracy"] = nll_acc(log_probs, y[val_idx])
    out["macro_f1"] = float(f1_score(y[val_idx], log_probs.argmax(1), average="macro"))
    probe = np.arange(min(20000, len(tr_idx)))
    out["train_accuracy"] = nll_acc(predict(train.take(probe)), y[tr_idx][probe])[1]

    # Validation-set probabilities are stored for the final report; no selection uses them.
    if job["split"] == "grouped" and (job["rep"], job["arch"]) in HELD_OUT:
        probs = {name: np.exp(predict(build_representation(job["rep"], *enc[f"validation_{name}"]))).astype(np.float32)
                 for name in ("matched", "mismatched")}
        (RESULTS / "probs").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(RESULTS / "probs" / f"{job['rep']}__{job['arch']}__s{job['seed']}__f{job['fold']}.npz", **probs)
    return out


def key(job):
    return json.dumps(job, sort_keys=True)


def run_stage(name, jobs, data, pool, tune):
    path = RESULTS / f"{name}.jsonl"
    done = {key(json.loads(line)["job"]) for line in path.open()} if path.exists() else set()
    todo = [j for j in jobs if key(j) not in done]
    print(f"{name}: {len(todo)} fits to run, {len(jobs) - len(todo)} already done", flush=True)
    start = time.time()
    with path.open("a") as log:
        for i, job in enumerate(todo, 1):
            result = run_job(job, data, pool, tune)
            log.write(json.dumps(result) + "\n")
            log.flush()
            print(f"[{i}/{len(todo)}, {(time.time() - start) / 60:.0f} min] {job['split']} {job['rep']} {job['arch']} "
                  f"V={job['tok']['vocab']} {job['params']} s{job['seed']} f{job['fold']}: "
                  f"accuracy {result['accuracy']:.4f}", flush=True)


def main():
    all_stages = stages()
    names = sys.argv[1:] or list(all_stages)
    RESULTS.mkdir(exist_ok=True)
    data = load_mnli()
    pool, tune = carve_tuning_set(data["train"].promptID.to_numpy())
    for name in names:
        run_stage(name, all_stages[name], data, pool, tune)


if __name__ == "__main__":
    main()
