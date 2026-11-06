"""Numbers on the slides that need extra fits or the corpus. Usage: python checks.py > results/checks.txt"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from nli.data import LABELS, carve_tuning_set, grouped_kfold, load_mnli
from nli.features import build_representation, fit_tokenizer, token_counts
from nli.models import make_network
from run import BASE_TOK, encoded

data = load_mnli()
train = data["train"]
groups, y = train.promptID.to_numpy(), train.label.to_numpy()
pool, tune = carve_tuning_set(groups)
folds = grouped_kfold(groups[pool], 5, 0)
enc = encoded(BASE_TOK, data, pool)
P, H = enc["train"]
V = P.shape[1]


def fit(X, idx):
    return LogisticRegression(max_iter=2000).fit(X[idx], y[idx])


words = lambda text: text.str.split().str.len()
print("=== Corpus")
print({name: len(df) for name, df in data.items()}, "pool", len(pool), "tuning", len(tune))
print("premises", train.promptID.nunique(), "pairs per premise", round(len(train) / train.promptID.nunique(), 2))
print("labels", train.label.value_counts().sort_index().to_dict())
print("mean words premise, hypothesis", round(words(train.premise).mean(), 1), round(words(train.hypothesis).mean(), 1))
print("genres", {name: sorted(df.genre.unique()) for name, df in data.items()})

print("\n=== Tokeniser data")
sub = train.iloc[pool]
premises = sub.premise.drop_duplicates()
print("unique premises", len(premises), "hypotheses", len(sub))
print("running words", int(words(premises).sum() + words(sub.hypothesis).sum()),
      "word types", len(set(" ".join(premises).split()) | set(" ".join(sub.hypothesis).split())))
unk = enc["vocab"]["[UNK]"]
for name in ("validation_matched", "validation_mismatched"):
    a, b = enc[name]
    print(name, "[UNK] share of tokens", (a[:, unk].sum() + b[:, unk].sum()) / (a.sum() + b.sum()))

print("\n=== Tokeniser refit inside each fold: log-linear BoW accuracy, global against per-fold")
X = build_representation("bow", P, H).flat()
gaps = []
for fold in folds:
    tr, va = pool[fold.train], pool[fold.val]
    part = train.iloc[tr]
    tok = fit_tokenizer(list(part.premise.unique()) + list(part.hypothesis), BASE_TOK["vocab"])
    X_fold = build_representation("bow", token_counts(tok, list(train.premise)), token_counts(tok, list(train.hypothesis))).flat()
    a, b = (fit(X, tr).predict(X[va]) == y[va]).mean(), (fit(X_fold, tr).predict(X_fold[va]) == y[va]).mean()
    gaps.append(b - a)
    print(round(float(a), 4), round(float(b), 4))
print("mean change (pp)", round(float(np.mean(gaps) * 100), 3))

print("\n=== Weights for contradiction: log-linear, BoW + overlap, 5 folds")
rep = build_representation("overlap", P, H)
X = rep.flat()
tokens = ("▁never", "▁no", "▁not")
ids = [enc["vocab"][t] for t in tokens]
contradiction = LABELS.index("contradiction")
weights = []
for fold in folds:
    coef = fit(X, pool[fold.train]).coef_
    coef = coef - coef.mean(0, keepdims=True)
    weights.append([[coef[contradiction, V + i], coef[contradiction, 2 * V + i]] for i in ids])
weights = np.array(weights)
for j, token in enumerate(tokens):
    print(token, "hypothesis, overlap, mean", weights[:, j].mean(0).round(2), "SD", weights[:, j].std(0, ddof=1).round(2),
          "sum", weights[:, j].sum(1).mean().round(2))

print("\n=== Shared tokens per pair, CV pool")
shared = np.asarray(rep.extra[pool].sum(1)).ravel()
for k, label in enumerate(LABELS):
    print(label, "mean", round(float(shared[y[pool] == k].mean()), 2), "SD", round(float(shared[y[pool] == k].std()), 2))

print("\n=== Parameters")
for name in ("bow", "overlap"):
    r = build_representation(name, P, H)
    print(name, "log-linear", 3 * r.flat().shape[1] + 3,
          {a: sum(p.numel() for p in make_network(a, r).parameters()) for a in ("mlp", "siamese")})
