"""Figures for the slides. Usage: python figures.py PATH/TO/project1/results

Reads the same fits as analyse.py and writes PDFs to results/figures/. Every number shown
is also printed by analyse.py or errors.py.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyse
import run

OUT = run.RESULTS / "figures"
BLUE, ORANGE, AQUA, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 6.5, "lines.linewidth": 1.2, "lines.markersize": 3, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "pdf.fonttype": 42})


# One name per system across slides and figures ("pool" is embed-and-pool, "scratch" a table trained from scratch).
NAME = {"rnn": "BiLSTM", "rnn-random": "BiLSTM, scratch", "rnn-hyp": "BiLSTM, hyp. only",
        "pool": "pool", "pool-random": "pool, scratch", "pool-hyp": "pool, hyp. only"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


SHORT = [("tok=tutorial-bow", "tutorial 2 settings"), ("hidden=20", "tutorial 2 LSTM"),
         ("emb=random, trainable=True", "scratch table"), ("proj=512, trainable=True", "fine-tune + 512d proj."),
         ("trainable=True", "fine-tune table"), ("proj=512", "512d projection"), ("tok=nltk-not", "n't → not"),
         ("weight_decay=", "decay "), ("head_hidden=", "MLP width "), ("head_depth=2", "2 MLP layers"),
         ("features=concat", "[u; v] only"), ("overlap=True", "+ P1 overlap block"), ("word_dropout=", "word dropout "),
         ("bidirectional=False", "unidirectional"), ("batch_size=", "batch "), ("activation=", ""), ("pooling=", ""),
         ("output=", "output "), ("cell=", ""), ("emb=random", "scratch table"), ("emb=", ""), ("dropout=", "dropout "), ("hidden=", "hidden "),
         ("layers=", "layers "), ("lr=", "lr ")]


def short(setting: str) -> str:
    for old, new in SHORT[:2]:
        if old in setting:
            return new
    for old, new in SHORT[2:]:
        setting = setting.replace(old, new)
    return setting


def dots(ax, t, color, label=None):
    """Horizontal dot plot of the seed-paired difference to the base, with its SD over seeds."""
    t = t[t.setting != "base"].sort_values("delta")
    t = t.assign(setting=[short(x) for x in t.setting])
    y = np.arange(len(t))
    ax.errorbar(t.delta, y, xerr=t.delta_sd, fmt="o", ms=2.5, color=color, ecolor=color, elinewidth=1, capsize=0, label=label)
    ax.set_yticks(y, t.setting)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvspan(-1, 1, color=GREY, alpha=0.12, lw=0)
    ax.set_xlabel("Δ to base (pp)")


def tables():
    df = analyse.load("embeddings")
    df = df[df.emb_std.notna()]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    order = [e for e in run.PRETRAINED + ("random",)]
    for enc, color, off in (("pool", BLUE, -0.12), ("rnn", ORANGE, 0.12)):
        g = df[df.enc == enc].groupby("emb").accuracy.agg(["mean", "std"]) * 100
        es = [e for e in order if e in g.index]
        ax.errorbar([order.index(e) + off for e in es], g.loc[es, "mean"], yerr=g.loc[es, "std"], fmt="o", ms=4,
                    color=color, label=NAME[enc])
    ax.set_xticks(range(len(order)), ["scratch (trained)" if e == "random" else e for e in order], rotation=35, ha="right")
    ax.set_ylabel("tuning accuracy (%), tables frozen")
    ax.legend(loc="lower right")
    save(fig, "q1_tables")


def search_fig(name, parts, size):
    fig, axes = plt.subplots(1, len(parts), figsize=size)
    for ax, (title, df, base, color) in zip(np.atleast_1d(axes), parts):
        dots(ax, analyse.by_setting(df, base), color)
        ax.set_title(title, fontsize=6.5, loc="left")
    save(fig, name)


def searches():
    pool, rnn, conf = analyse.load("pool"), analyse.load("rnn"), analyse.load("confirm")
    # the lr x decay grid off the base learning rate is in results/search.txt; the plot keeps the base lr
    pool = pool[pool.cfg.apply(lambda c: json.loads(c)["lr"] == run.POOL["lr"] or json.loads(c)["tok"] == "tutorial-bow")]
    search_fig("q3_pool_search", [("Step 1, base 59.6%", pool, run.POOL, BLUE),
                                  ("Step 2, base 65.4%", conf[conf.enc == "pool"],
                                   run.POOL2, BLUE)], (3.9, 2.55))
    search_fig("q4_rnn_search", [("Step 1, base 71.6%", rnn, run.RNN, ORANGE),
                                 ("Step 2, base 73.0%", conf[conf.enc == "rnn"], run.RNN2, ORANGE)],
               (3.9, 2.55))


def datasize():
    jobs = run.stages()["datasize"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(3.4, 1.75))
    gains = {}
    for name, color, ls in (("rnn", ORANGE, "-"), ("rnn-random", ORANGE, "--"), ("pool", BLUE, "-"), ("pool-random", BLUE, "--")):
        df = analyse.load_jobs([j for j in jobs if j["cfg"] == run.MAIN[name]])
        g = df.groupby("frac").agg(n=("n_train", "mean"), acc=("accuracy", "mean"), sd=("accuracy", "std"))
        ax.errorbar(g.n, 100 * g.acc, yerr=100 * g.sd, color=color, ls=ls, marker="o", ms=2.5, lw=1.2, label=NAME[name])
        gains[name] = df.set_index(["frac", "seed"]).accuracy
    for enc, color in (("rnn", ORANGE), ("pool", BLUE)):
        d = 100 * (gains[enc] - gains[f"{enc}-random"]).groupby("frac").agg(["mean", "std"])
        n = analyse.load_jobs([j for j in jobs if j["cfg"] == run.MAIN[enc]]).groupby("frac").n_train.mean()
        ax2.errorbar(n.loc[d.index], d["mean"], yerr=d["std"], color=color, marker="o", ms=2.5, lw=1.2, label=NAME[enc])
    for a in (ax, ax2):
        a.set_xscale("log")
        a.set_xlabel("training pairs")
    ax.set_ylabel("fold accuracy (%)")
    ax2.set_ylabel("pretrained − scratch (pp)")
    ax2.set_ylim(bottom=0)
    ax.legend(fontsize=5.5)
    ax2.legend(fontsize=5.5)
    save(fig, "q4_datasize")


def learning_rate():
    df = analyse.load("lr")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, enc, title in zip(axes, ("rnn", "pool"), ("BiLSTM", "pool")):
        for (x, h), color in zip(df[df.enc == enc].groupby("lr"), (BLUE, AQUA, VIOLET)):
            n = min(len(r) for r in h.history)
            for key, ls in (("tune_acc", "-"), ("train_acc", ":")):
                curve = 100 * np.mean([[e[key] for e in r][:n] for r in h.history], 0)
                ax.plot(np.arange(1, n + 1), curve, color=color, ls=ls, lw=2 if ls == "-" else 1.2,
                        label=f"lr {x:g}" if key == "tune_acc" else None)
        ax.set_title(f"{title} (solid: tuning set, dotted: training)", fontsize=9, loc="left")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("accuracy (%), mean of 3 seeds")
    save(fig, "q5_learning_rate")


def final():
    p1 = analyse.project1()
    rows = [(NAME[n], 100 * analyse.cv_scores(n).accuracy.to_numpy()) for n in run.MAIN]
    rows += [(n.replace("p1 ", "P1 "), 100 * p1[n]) for n in ("p1 overlap siamese", "p1 bow siamese", "p1 overlap loglinear")]
    rows.sort(key=lambda r: r[1].mean())
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, (n, a) in enumerate(rows):
        color = ORANGE if n.startswith("BiLSTM") else BLUE if n.startswith("pool") else GREY
        ax.barh(i, a.mean(), color=color, height=0.6, alpha=0.35 if ("scratch" in n or "hyp" in n) else 1)
        ax.plot(a, np.full(len(a), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(a)), "o", ms=2, color=INK, alpha=0.5)
        ax.text(a.mean() + 0.6, i, f"{a.mean():.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(50, 76)
    ax.set_xlabel("accuracy (%), grouped 5-fold CV × 3 (dots: the 15 folds)")
    save(fig, "final_systems")


def q6():
    path = run.RESULTS / "errors_posterior.npz"
    if not path.exists():
        print("[q6: run errors.py first]")
        return
    z = np.load(path)
    w, names, cols = z["w"], list(z["names"]), list(z["cols"])
    feats = [("overlap", "word overlap"), ("alignment", "alignment (Numberbatch)"), ("neg_hypothesis", "negation in hyp."),
             ("rarest_hyp_word", "log freq. rarest hyp. word"), ("oov", "share without vector"),
             ("antonym_pair", "antonym pair"), ("len_premise", "premise length"), ("depth_premise", "premise parse depth")]
    show = [("p1-siamese", GREY, "P1 Siamese"), ("pool", BLUE, "pool"), ("pool-random", "#9cc0ec", "pool, scratch"),
            ("rnn", ORANGE, "BiLSTM"), ("rnn-random", "#f4ae90", "BiLSTM, scratch")]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    y = np.arange(len(feats))
    for k, (n, color, label) in enumerate(show):
        s = w[:, names.index(n)][:, [cols.index(f) for f, _ in feats]]
        lo, hi = np.percentile(s, [2.5, 97.5], axis=0)
        off = (k - (len(show) - 1) / 2) * 0.15
        ax.errorbar(s.mean(0), y + off, xerr=[s.mean(0) - lo, hi - s.mean(0)], fmt="o", ms=2.2, color=color,
                    elinewidth=0.8, capsize=0, label=label)
    ax.set_yticks(y, [l for _, l in feats])
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("weight per SD (log-odds correct), 95% interval")
    ax.legend(fontsize=5.5, loc="lower right")
    save(fig, "q6_weights")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (tables, searches, datasize, learning_rate, final, q6):
        f()
