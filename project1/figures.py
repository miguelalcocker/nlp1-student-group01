"""Figures used on the slides. Usage: python figures.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nli.stats import RESULTS, ROPE, load, posterior, regions, scores, verdict

FIG = RESULTS / "figures"
INK, MUTED, BLUE, RED, GREY, ROPE_FILL = "#1F2933", "#616E7C", "#1F4E79", "#A31F34", "#8592A0", "#DDE2E7"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
                     "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def tokeniser():
    """Tuning-set accuracy against vocabulary size."""
    bpe = load("tokeniser")
    bpe = bpe[(bpe.kind == "bpe") & ~bpe.lowercase][["arch", "vocab", "accuracy"]]
    rep = load("representations")
    base = rep[(rep.rep == "bow") & (rep.arch == "loglinear")].assign(vocab=2000)[["arch", "vocab", "accuracy"]]
    df = pd.concat([bpe, base]).sort_values("vocab")
    fig, ax = plt.subplots(figsize=(3.05, 2.1))
    ax.axvline(2000, color=MUTED, lw=.8, ls=(0, (2, 2)))
    for arch, name, color, marker in (("siamese", "Siamese", RED, "s"), ("mlp", "MLP", BLUE, "o"),
                                      ("loglinear", "Log-linear", GREY, "^")):
        s = df[df.arch == arch]
        ax.plot(s.vocab, s.accuracy * 100, color=color, lw=1.8, marker=marker, ms=4)
        ax.text(s.vocab.iloc[-1] * 1.15, s.accuracy.iloc[-1] * 100, name, color=INK, va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([250, 1000, 2000, 5000, 10000, 30000], ["250", "1k", "2k", "5k", "10k", "30k"])
    ax.minorticks_off()
    ax.set_xlim(220, 130000)
    ax.set_xlabel("BPE vocabulary size V")
    ax.set_ylabel("Tuning-set accuracy (%)")
    save(fig, "tokeniser")


def dumbbell(rows, left_name, right_name, name, xlim, figsize, chance=True):
    """Two values per row, drawn on the axis and listed in two columns so that labels cannot collide."""
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(rows))[::-1]
    columns = ax.get_yaxis_transform()
    if chance:
        ax.axvline(100 / 3, color=MUTED, lw=.8, ls=(0, (2, 2)))
        ax.text(100 / 3 + .7, -.4, "chance", ha="left", va="center", fontsize=7, color=MUTED)
    for yi, (_, a, b) in zip(y, rows):
        ax.plot([a, b], [yi, yi], color=ROPE_FILL, lw=3, zorder=1, solid_capstyle="round")
        ax.scatter([a], [yi], color=GREY, marker="^", s=24, zorder=2)
        ax.scatter([b], [yi], color=BLUE, marker="o", s=20, zorder=3)
        ax.text(1.17, yi, f"{a:.1f}", ha="right", va="center", transform=columns, fontsize=7.5)
        ax.text(1.46, yi, f"{b:.1f}", ha="right", va="center", transform=columns, fontsize=7.5)
    ax.text(1.17, len(rows) - .3, {"Random": "Rand.", "Log-lin.": "LL"}[left_name], ha="right", va="center", transform=columns, fontsize=7, color=MUTED)
    ax.text(1.46, len(rows) - .3, {"Premise": "Prem.", "MLP": "MLP"}[right_name], ha="right", va="center", transform=columns, fontsize=7, color=MUTED)
    ax.scatter([], [], color=GREY, marker="^", s=18, label=left_name)
    ax.scatter([], [], color=BLUE, marker="o", s=16, label=right_name)
    ax.legend(loc="lower left", bbox_to_anchor=(-.02, 1), ncol=2, frameon=False, fontsize=7, handletextpad=.1,
              columnspacing=.8, borderaxespad=0, borderpad=0)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(-.6, len(rows))
    ax.set_xlabel("CV accuracy (%)")
    save(fig, name)


def ablation(df):
    mean = lambda r, a: scores(df, r, a).mean() * 100
    rows = [(label, mean(r, "loglinear"), mean(r, "mlp")) for label, r in (
        ("Overlap only", "and"), ("One bag, p + h", "bow-concat"), ("h only", "hyp-only"),
        ("BoW, p; h", "bow"), ("h; overlap", "hyp-and"), ("BoW + overlap", "overlap"))]
    dumbbell(rows, "Log-lin.", "MLP", "ablation", (30, 68), (1.45, 1.5))


def leak(df, leak_df):
    def row(label, r, a):
        grouped = df[(df.rep == r) & (df.arch == a) & (df.seed == 0)].accuracy.mean() * 100
        random = leak_df[(leak_df.rep == r) & (leak_df.arch == a)].accuracy.mean() * 100
        return label, random, grouped
    rows = [row("Premise only, log-lin.", "prem-only", "loglinear"), row("Premise only, MLP", "prem-only", "mlp"),
            row("Hypothesis only, MLP", "hyp-only", "mlp"), row("Pair BoW, log-lin.", "bow", "loglinear"),
            row("Pair BoW, MLP", "bow", "mlp")]
    dumbbell(rows, "Random", "Premise", "leak", (22, 66), (1.75, 1.45), chance=False)


def curves(main):
    """Train and tuning-set NLL per epoch, averaged over the CV fits that reached that epoch."""
    fits = main[main.rep == "overlap"]
    runs = {arch: fits[fits.arch == arch].to_dict("records") for arch in ("mlp", "siamese")}
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.1), sharey=True)
    for ax, (arch, rs) in zip(axes, runs.items()):
        n = max(len(r["history"]) for r in rs)
        for key, color, label in (("train_nll", GREY, "train"), ("tune_nll", BLUE, "tuning")):
            y = [np.mean([r["history"][e][key] for r in rs if len(r["history"]) > e]) for e in range(n)]
            ax.plot(np.arange(1, n + 1), y, color=color, lw=1.8)
            ax.text(n + .25, y[-1], label, color=INK, fontsize=7.5, va="center")
        ax.axvline(np.mean([r["best_epoch"] for r in rs]) + 1, color=INK, lw=.8, ls=(0, (2, 2)))
        ax.set_title("MLP" if arch == "mlp" else "Siamese MLP", fontsize=8, loc="left")
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(2, n + 1, 2))
        ax.set_xlim(.5, n + 4.2)
    axes[0].set_ylabel("NLL")
    fig.subplots_adjust(wspace=.08)
    save(fig, "curves")


def forest(df):
    """Posterior mean and 95% interval for the comparisons behind the decision."""
    L, M, S = "loglinear", "mlp", "siamese"
    pairs = [("BoW: log-linear → MLP", ("bow", L), ("bow", M), "MLP"),
             ("BoW: MLP → Siamese", ("bow", M), ("bow", S), "Siamese"),
             ("Overlap: log-linear → MLP", ("overlap", L), ("overlap", M), "MLP"),
             ("Overlap: MLP → Siamese", ("overlap", M), ("overlap", S), "Siamese"),
             ("Log-linear: BoW → overlap", ("bow", L), ("overlap", L), "overlap"),
             ("MLP: BoW → overlap", ("bow", M), ("overlap", M), "overlap"),
             ("Siamese: BoW → overlap", ("bow", S), ("overlap", S), "overlap"),
             ("Siamese: indicator → BoW", ("ind", S), ("bow", S), "BoW")]
    fig, ax = plt.subplots(figsize=(1.75, 1.75))
    ax.axvspan(-ROPE * 100, ROPE * 100, color=ROPE_FILL, lw=0)
    ax.axvline(0, color=MUTED, lw=.6)
    y = np.arange(len(pairs))[::-1]
    for yi, (_, a, b, winner) in zip(y, pairs):
        t = posterior(scores(df, *a), scores(df, *b))
        lo, hi = t.ppf(.025) * 100, t.ppf(.975) * 100
        ax.plot([lo, hi], [yi, yi], color=BLUE, lw=1.6)
        ax.scatter([t.mean() * 100], [yi], color=BLUE, s=16, zorder=3)
        v = verdict(t)
        p = regions(t)
        word = {"B": f"{winner} wins", "A": "first wins", "tie": "tie"}.get(v, f"undecided, tie {p[1]:.2f}")
        ax.text(10.4, yi, f"{t.mean() * 100:+.1f}", ha="left", va="center", fontsize=7.5)
        ax.text(13.1, yi, word, ha="left", va="center", fontsize=7.5)
    ax.set_yticks(y, [p[0] for p in pairs])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-1.6, 10)
    ax.set_ylim(-.6, len(pairs) - .4)
    ax.set_xlabel("Gain, second − first (pp); grey band: ROPE")
    save(fig, "forest")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    main_df, controls = load("main"), load("controls")
    both = pd.concat([main_df, controls[controls.split == "grouped"]])
    tokeniser()
    ablation(both)
    leak(both, controls[controls.split == "random"])
    curves(main_df)
    forest(main_df)
