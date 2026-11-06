"""Print every result number on the slides. Usage: python analyse.py > results/analysis.txt"""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from nli.data import LABELS, load_mnli
from nli.stats import RESULTS, load, posterior, regions, scores, verdict

REPS = ("ind", "bow", "overlap")
ARCHS = ("loglinear", "mlp", "siamese")
NEURAL_SYSTEMS = [(r, a) for r in REPS for a in ("mlp", "siamese")]


def pct(x):
    return round(float(x) * 100, 2)


def tune_acc(df, rep, arch, **params):
    match = df[(df.rep == rep) & (df.arch == arch) & (df.params == json.dumps(params, sort_keys=True))]
    return match.accuracy.iloc[-1]


def q2_tokeniser():
    print("\n=== Q2: tokeniser, tuning-set accuracy, one fit per setting")
    df = load("tokeniser")
    print(df[["arch", "kind", "lowercase", "vocab", "accuracy", "train_accuracy"]]
          .sort_values(["arch", "kind", "lowercase", "vocab"]).round(4).to_string(index=False))


def q3_features(main, controls):
    print("\n=== Q3: representations, CV accuracy (%), mean and SD of 15 fits")
    both = pd.concat([main, controls])
    table = both[both.rep.isin(["and", "bow-concat", "hyp-only", "bow", "hyp-and", "overlap"]) & (both.arch != "siamese")]
    print((table.groupby(["rep", "arch"]).accuracy.agg(["mean", "std"]) * 100).round(2).to_string())
    print("\nAlternatives, tuning-set accuracy and train accuracy, one fit")
    rep = load("representations")
    print(rep.pivot_table(index="rep", columns="arch", values=["accuracy", "train_accuracy"]).round(4).to_string())


def q4_architectures():
    search = load("search")
    print("\n=== Q4: search on the tuning set, mean change in pp over 6 neural systems")
    around = {"hidden 128": dict(hidden=128), "dropout 0.1": dict(dropout=0.1), "dropout 0.5": dict(dropout=0.5),
              "GELU": dict(activation="gelu"), "tanh": dict(activation="tanh")}
    for name, change in around.items():
        d = [tune_acc(search, r, a, **change) - tune_acc(search, r, a) for r, a in NEURAL_SYSTEMS]
        print(f"{name:12s} {np.mean(d) * 100:+.2f}   per system: {np.round(np.array(d) * 100, 2).tolist()}")
    base = dict(lr=3e-4, weight_decay=0.3)
    for name, change in (("lr 1e-4", dict(lr=1e-4, weight_decay=0.3)), ("wd 0.1", dict(lr=3e-4, weight_decay=0.1))):
        d = [tune_acc(search, r, a, **change) - tune_acc(search, r, a, **base) for r, a in NEURAL_SYSTEMS]
        print(f"{name:12s} {np.mean(d) * 100:+.2f}")
    d = [tune_acc(search, r, "mlp2") - tune_acc(search, r, "mlp") for r in REPS]
    print(f"{'2 layers':12s} {np.mean(d) * 100:+.2f}   3 MLPs")
    print("\nLog-linear C:")
    print(search[search.arch == "loglinear"].pivot_table(index="params", columns="rep", values="accuracy").round(4).to_string())
    print("\nFirst guess (lr 1e-3, wd 1e-4) against chosen (lr 3e-4, wd 0.3):")
    for r, a in (("bow", "mlp"), ("bow", "siamese"), ("overlap", "siamese")):
        print(f"{r:8s} {a:8s} {pct(tune_acc(search, r, a, lr=1e-3, weight_decay=1e-4))} {pct(tune_acc(search, r, a, **base))}")


def q5_evaluation(main, controls, data):
    print("\n=== Q5: CV accuracy (%), random against premise split, seed 0, 5 folds")
    both = pd.concat([main, controls[controls.split == "grouped"]])
    random = controls[controls.split == "random"].groupby(["rep", "arch"]).accuracy
    grouped = both[both.seed == 0].groupby(["rep", "arch"]).accuracy.mean()
    print((pd.DataFrame({"random": random.mean(), "random_sd": random.std(), "premise": grouped}).dropna() * 100).round(2).to_string())
    print("\nMax |macro-F1 - accuracy| in the grid (pp): any single fit",
          round(float((main.macro_f1 - main.accuracy).abs().max() * 100), 2), ", means of 15 fits",
          round(float((main.groupby(["rep", "arch"]).macro_f1.mean() - main.groupby(["rep", "arch"]).accuracy.mean()).abs().max() * 100), 2))
    nets = main[main.arch != "loglinear"]
    on_fold = nets.history.map(lambda h: min(h, key=lambda e: e["val_nll"])["val_acc"])
    on_tuning = [h[int(b)]["val_acc"] for h, b in zip(nets.history, nets.best_epoch)]
    print("Early stopping on the fold, not the tuning set, mean change (pp):", round(float((on_fold - on_tuning).mean() * 100), 2))
    hyp, best = scores(both, "hyp-only", "mlp").mean(), scores(main, "overlap", "siamese").mean()
    print(f"Hypothesis-only share of the best gain over chance: {(hyp - 1 / 3) / (best - 1 / 3) * 100:.1f}%")
    for rep, arch in (("overlap", "siamese"), ("hyp-only", "mlp")):
        held_out(rep, arch, data)


def held_out(rep, arch, data):
    files = sorted((RESULTS / "probs").glob(f"{rep}__{arch}__s*__f*.npz"))
    if not files:
        print(f"\nNo validation-set probabilities for {rep}/{arch}: they are written by run.py, not committed.")
        return
    print(f"\nValidation sets, {rep}/{arch}, {len(files)} models")
    rng = np.random.default_rng(0)
    for name in ("matched", "mismatched"):
        df = data[f"validation_{name}"]
        y, genre = df.label.to_numpy(), df.genre.to_numpy()
        probs = np.stack([np.load(f)[name] for f in files])
        singles = (probs.argmax(2) == y).mean(1) * 100
        votes = np.stack([(probs.argmax(2) == k).sum(0) for k in range(3)], 1) + rng.random((probs.shape[1], 3)) * .5
        vote = votes.argmax(1)
        print(f"{name}: single {singles.mean():.2f} (SD {singles.std(ddof=1):.2f}), vote {pct((vote == y).mean())},"
              f" macro-F1 {pct(f1_score(y, vote, average='macro'))}, n {len(y)}")
        print("  recall", {LABELS[k][0].upper(): pct((vote[y == k] == k).mean()) for k in range(3)})
        print("  genres", {str(g): pct((vote[genre == g] == y[genre == g]).mean()) for g in np.unique(genre)})


def q6_decision(main):
    print("\n=== Q6: CV accuracy (%) and NLL, mean of 15 fits")
    grid = main.groupby(["rep", "arch"])
    for name, values, scale in (("accuracy", "accuracy", 100), ("NLL", "nll", 1)):
        print(f"\n{name}, mean\n", (grid[values].mean() * scale).unstack().loc[list(REPS), list(ARCHS)].round(3).to_string())
        print(f"{name}, SD over fits\n", (grid[values].std() * scale).unstack().loc[list(REPS), list(ARCHS)].round(3).to_string())
    print("\nSD between the 3 CV repetitions (pp)\n",
          (main.groupby(["rep", "arch", "seed"]).accuracy.mean().groupby(["rep", "arch"]).std() * 100).unstack().round(3).to_string())

    print("\nAll pairs: gain of b over a (pp), 95% interval, P(a), P(tie), P(b), verdict at ROPE 0.5, 1, 2 pp")
    systems = [(r, a) for r in REPS for a in ARCHS]
    verdicts = {0.005: [], 0.01: [], 0.02: []}
    for s1, s2 in combinations(systems, 2):
        t = posterior(scores(main, *s1), scores(main, *s2))
        for rope in verdicts:
            verdicts[rope].append(verdict(t, rope))
        p = regions(t)
        print(f"{'/'.join(s1):20s} {'/'.join(s2):20s} {t.mean() * 100:+6.2f} [{t.ppf(.025) * 100:+6.2f}, {t.ppf(.975) * 100:+6.2f}] "
              f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}  {verdicts[0.005][-1]} {verdicts[0.01][-1]} {verdicts[0.02][-1]}")
    at_1 = np.array(verdicts[0.01])
    print("\nAt 1 pp:", {v: int((at_1 == v).sum()) for v in ("A", "B", "tie", "undecided")})
    for rope in (0.005, 0.02):
        print(f"At {rope * 100} pp, {int((np.array(verdicts[rope]) != at_1).sum())} verdicts change")


if __name__ == "__main__":
    main_df, controls_df = load("main"), load("controls")
    mnli = load_mnli()
    q2_tokeniser()
    q3_features(main_df, controls_df[controls_df.split == "grouped"])
    q4_architectures()
    q5_evaluation(main_df, controls_df, mnli)
    q6_decision(main_df)
