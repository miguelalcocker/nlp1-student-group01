"""Print every result number used on the slides. Usage: python analyse.py > results/analysis.txt

Tuning-set results: every setting was fitted with 3 seeds (initialisation and batch order;
the data are the same). We report the mean and SD over seeds, and differences to the base
setting paired by seed, with their SD over the 3 seeds.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import run
from nli.stats import ROPE, posterior, regions, verdict

FITS = run.RESULTS / "fits"
P1_RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../project1/results")


def load(stage: str) -> pd.DataFrame:
    rows = []
    for job in run.stages()[stage]:
        path = FITS / f"{run.key(job)}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        c = run.canonical(job["cfg"])
        rows.append({**{k: v for k, v in job.items() if k != "cfg"}, **c, "cfg": json.dumps(c, sort_keys=True),
                     "accuracy": r["accuracy"], "nll": r["nll"], "macro_f1": r["macro_f1"],
                     "train_accuracy": r["train_accuracy"], "best_epoch": r["best_epoch"],
                     "epochs_run": r["epochs_run"], "capped": r.get("capped", r["epochs_run"] >= c["max_epochs"]),
                     "minutes": r["seconds"] / 60, "n_params": r["n_params"],
                     "history": r["history"]})
    df = pd.DataFrame(rows)
    capped = int(df.capped.sum()) if len(df) else 0
    print(f"[{stage}: {len(rows)} of {len(run.stages()[stage])} fits; {capped} reached the epoch cap]")
    return df


def change_label(c: dict, base: dict) -> str:
    diff = {k: v for k, v in c.items() if base.get(k) != v}
    return ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in diff.items()) or "base"


def by_setting(df: pd.DataFrame, base: dict) -> pd.DataFrame:
    """Mean and SD over seeds, and the seed-paired difference to the base setting (pp)."""
    df = df.copy()
    df["setting"] = [change_label(run.canonical(json.loads(c)), run.canonical(base)) for c in df.cfg]
    base_acc = df[df.setting == "base"].set_index("seed").accuracy
    rows = []
    for s, g in df.groupby("setting", sort=False):
        g = g.set_index("seed")
        d = (g.accuracy - base_acc).dropna() * 100
        rows.append({"setting": s, "n": len(g), "acc": 100 * g.accuracy.mean(), "sd": 100 * g.accuracy.std(),
                     "delta": d.mean() if len(d) else np.nan, "delta_sd": d.std() if len(d) > 1 else np.nan,
                     "train": 100 * g.train_accuracy.mean(), "epochs": g.epochs_run.mean(), "best_epoch": g.best_epoch.mean(),
                     "minutes": g.minutes.mean()})
    return pd.DataFrame(rows)


def show(df, title):
    print(f"\n=== {title}")
    print(df.round(2).to_string(index=False))


def lr():
    df = load("lr")
    for enc in ("rnn", "pool"):
        g = df[df.enc == enc]
        t = g.groupby("lr").agg(acc=("accuracy", "mean"), sd=("accuracy", "std"), nll=("nll", "mean"),
                                train=("train_accuracy", "mean"), best_epoch=("best_epoch", "mean"),
                                epochs=("epochs_run", "mean"), minutes=("minutes", "mean"))
        t[["acc", "sd", "train"]] *= 100
        show(t.reset_index(), f"Learning rate, {enc} base setting, tuning set (3 seeds)")
        for x, h in g.groupby("lr"):
            curve = np.mean([[e["tune_acc"] for e in r][:min(len(r) for r in h.history)] for r in h.history], 0)
            print(f"  {enc} lr {x:g}: mean tuning accuracy by epoch", " ".join(f"{100 * a:.1f}" for a in curve))


def embeddings():
    df = load("embeddings")
    for enc, base in (("pool", run.POOL), ("rnn", run.RNN)):
        t = by_setting(df[df.enc == enc], base)
        show(t, f"Q1-Q3: tables, {enc}, tuning accuracy % (3 seeds); delta to glove840B, paired by seed")
    seeds_sd = df.groupby("cfg").accuracy.std() * 100
    print(f"\nSD over seeds of one setting, median over settings: {seeds_sd.median():.2f} pp, max {seeds_sd.max():.2f} pp")


def tokens():
    df = load("tokens")
    df["setting"] = df.emb + " " + df.tok + " " + df.case
    t = df.groupby("setting", sort=False).agg(acc=("accuracy", "mean"), sd=("accuracy", "std"), train=("train_accuracy", "mean"))
    show((t * 100).reset_index(), "Q1: tokeniser and case, embed-and-pool, tuning accuracy % (3 seeds)")


def search(stage, base, df=None):
    df = load(stage) if df is None else df
    t = by_setting(df, base)
    show(t, f"{stage}: one change at a time, tuning accuracy % (3 seeds); delta paired by seed")
    return df


def project1() -> dict[str, np.ndarray]:
    """Project 1's 15 fold accuracies per system, ordered by (seed, fold): the same folds as ours."""
    rows = [json.loads(line) for line in (P1_RESULTS / "main.jsonl").open()]
    out = {}
    for rep, arch in sorted({(r["job"]["rep"], r["job"]["arch"]) for r in rows}):
        r = sorted((x for x in rows if (x["job"]["rep"], x["job"]["arch"]) == (rep, arch)),
                   key=lambda x: (x["job"]["seed"], x["job"]["fold"]))
        out[f"p1 {rep} {arch}"] = np.array([x["accuracy"] for x in r])
    return out


def cv_scores(name: str) -> pd.DataFrame:
    df = load_jobs(run.cv(run.MAIN[name]))
    return df.sort_values(["seed", "fold"]) if len(df) else df


def load_jobs(jobs) -> pd.DataFrame:
    rows = []
    for job in jobs:
        path = FITS / f"{run.key(job)}.json"
        if path.exists():
            r = json.loads(path.read_text())
            h = r["history"]
            rows.append({"seed": job["seed"], "fold": job["fold"], "frac": job["frac"], "accuracy": r["accuracy"],
                         "nll": r["nll"], "macro_f1": r["macro_f1"], "train_accuracy": r["train_accuracy"],
                         "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"], "capped": r.get("capped", False),
                         "n_train": r["n_train"], "n_params": r["n_params"], "minutes": r["seconds"] / 60,
                         # fold accuracy at the epoch early stopping kept, and the best the fold itself would pick
                         "val_at_best": h[r["best_epoch"]].get("val_acc", np.nan),
                         "val_oracle": max((e.get("val_acc", np.nan) for e in h), default=np.nan),
                         "key": run.key(job)})
    return pd.DataFrame(rows)


def compare(name_a, a, name_b, b):
    t = posterior(a, b)
    pa, tie, pb = regions(t)
    print(f"  {name_b:>18} - {name_a:<18} {100 * (b - a).mean():+6.2f} pp  P({name_a} better) {pa:.3f}  "
          f"P(tie) {tie:.3f}  P({name_b} better) {pb:.3f}  -> {verdict(t)}")


def heldout(keys: list[str], labels: dict) -> dict:
    """Mean accuracy of the single fits, and of the average of their probabilities, on each validation set."""
    out = {}
    for s, y in labels.items():
        probs = [np.load(run.RESULTS / "probs" / f"{k}.npz")[s].astype(np.float32) for k in keys]
        single = [(p.argmax(1) == y).mean() for p in probs]
        out[s] = (100 * np.mean(single), 100 * np.std(single, ddof=1), 100 * (np.mean(probs, 0).argmax(1) == y).mean())
    return out


def main_results():
    from nli.data import load_mnli
    data = load_mnli()
    labels = {s: data[s].label.to_numpy() for s in ("validation_matched", "validation_mismatched")}
    systems = {name: cv_scores(name) for name in run.MAIN}
    rows = []
    for name, df in systems.items():
        if len(df) == 0:
            continue
        row = {"system": name, "fits": len(df), "acc": 100 * df.accuracy.mean(), "sd": 100 * df.accuracy.std(),
               "f1": 100 * df.macro_f1.mean(), "train": 100 * df.train_accuracy.mean(), "epochs": df.epochs_run.mean(),
               "capped": int(df.capped.sum()), "es_optimism": 100 * (df.val_oracle - df.val_at_best).mean(),
               "params": int(df.n_params.iloc[0]), "minutes": df.minutes.mean()}
        if len(df) == 15 and all((run.RESULTS / "probs" / f"{k}.npz").exists() for k in df.key):
            for s, (m, sd, vote) in heldout(list(df.key), labels).items():
                tag = "m" if s.endswith("_matched") else "mm"
                row[f"{tag}_single"], row[f"{tag}_sd"], row[f"{tag}_vote"] = m, sd, vote
        rows.append(row)
    show(pd.DataFrame(rows), "Final systems: grouped 5-fold CV x 3 (accuracy %), and validation sets "
                             "(single = mean over the 15 fits, vote = average of their probabilities)")
    print("es_optimism: fold accuracy of the best epoch for the fold minus that of the epoch kept by the tuning set (pp).")
    p1 = project1()
    print("\n=== Project 1 systems on the same folds (accuracy %, 15 fits)")
    for n, a in p1.items():
        print(f"  {n:<24} {100 * a.mean():.2f} +- {100 * a.std(ddof=1):.2f}")
    acc = {n: df.accuracy.to_numpy() for n, df in systems.items() if len(df) == 15}
    best_p1 = "p1 overlap siamese"
    acc[best_p1] = p1[best_p1]
    print(f"\n=== Correlated Bayesian t-test (runs=3), ROPE +-{100 * ROPE:.0f} pp; difference = second - first")
    pairs = [(best_p1, "pool"), (best_p1, "rnn"), ("pool", "rnn"), ("pool-random", "pool"), ("rnn-random", "rnn"),
             (best_p1, "pool-random"), (best_p1, "rnn-random"), ("pool-random", "rnn-random"),
             ("pool-hyp", "pool"), ("rnn-hyp", "rnn"), ("pool-hyp", "rnn-hyp")]
    pairs += [(a, b) for a, b in combinations([n for n in run.MAIN if n not in ("pool", "pool-random", "rnn",
                                                                             "rnn-random", "pool-hyp", "rnn-hyp")], 2)]
    for a, b in pairs:
        if a in acc and b in acc:
            compare(a, acc[a], b, acc[b])


def datasize():
    jobs = run.stages()["datasize"]
    rows = []
    for name in ("pool", "pool-random", "rnn", "rnn-random"):
        df = load_jobs([j for j in jobs if j["cfg"] == run.MAIN[name]])
        if len(df) == 0:
            continue
        for f, g in df.groupby("frac"):
            rows.append({"system": name, "frac": f, "n": len(g), "n_train": int(g.n_train.mean()),
                         "acc": 100 * g.accuracy.mean(), "sd": 100 * g.accuracy.std(), "train": 100 * g.train_accuracy.mean(),
                         "epochs": g.epochs_run.mean(), "capped": int(g.capped.sum()),
                         "_by_seed": g.set_index("seed").accuracy})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("[datasize: no fits yet]")
        return
    show(df.drop(columns="_by_seed"), "Q4: data size, fold 0 of each repetition (accuracy %, 3 fits per point)")
    print("\n=== Gain from pretraining at each data size: pretrained - random, paired by repetition (pp)")
    for enc in ("pool", "rnn"):
        for f in sorted(df.frac.unique()):
            a = df[(df.system == enc) & (df.frac == f)]
            b = df[(df.system == f"{enc}-random") & (df.frac == f)]
            if len(a) and len(b):
                d = 100 * (a._by_seed.iloc[0] - b._by_seed.iloc[0]).dropna()
                print(f"  {enc:>4} frac {f:<5} {d.mean():+6.2f} +- {d.std():.2f}  (per repetition: {', '.join(f'{x:+.2f}' for x in d)})")


def confirm():
    df = load("confirm")
    search("confirm (embed-and-pool, around fine-tuned + 512d projection)", run.POOL2, df[df.enc == "pool"])
    search("confirm (BiLSTM, around Numberbatch)", run.RNN2, df[df.enc == "rnn"])


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for name in sys.argv[2:] or ["lr", "embeddings", "tokens", "pool", "rnn", "confirm", "main", "datasize"]:
        {"lr": lr, "embeddings": embeddings, "main": main_results, "confirm": confirm, "datasize": datasize, "tokens": tokens,
         "pool": lambda: search("pool", run.POOL), "rnn": lambda: search("rnn", run.RNN)}[name]()
