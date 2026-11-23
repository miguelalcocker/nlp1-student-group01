"""Sentence encoders (embed-and-pool, recurrent) and the pair classifier on top of them.

Premise and hypothesis go through the *same* encoder, and the classifier sees
[u; v; |u - v|; u * v] (Conneau et al., 2017), as the Siamese MLP of project 1 did. That
Siamese MLP was itself an embed-and-pool model: sum pooling of learned 512-d embeddings of
BPE tokens. Here the embeddings are word vectors, pretrained or not.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class Embedder(nn.Module):
    """Embedding table (row 0 is padding), optional word dropout and per-token projection."""

    def __init__(self, matrix: torch.Tensor, trainable: bool, proj: int = 0, word_dropout: float = 0.0):
        super().__init__()
        self.table = nn.Embedding.from_pretrained(matrix, freeze=not trainable, padding_idx=0)
        self.proj = nn.Linear(matrix.shape[1], proj) if proj else None
        self.word_dropout = word_dropout
        self.out_dim = proj or matrix.shape[1]

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = ids.ne(0)
        e = self.table(ids)
        if self.training and self.word_dropout > 0:
            # Word dropout zeroes whole tokens (Gal & Ghahramani, 2016), as in tutorial 2.
            drop = (torch.rand(ids.shape, device=ids.device) < self.word_dropout) & valid
            e = e.masked_fill(drop.unsqueeze(-1), 0.0)
        if self.proj is not None:
            e = self.proj(e)
        return e, valid


class PoolEncoder(nn.Module):
    def __init__(self, embedder: Embedder, pooling: str):
        super().__init__()
        self.embedder, self.pooling = embedder, pooling
        self.out_dim = embedder.out_dim * (2 if pooling == "meanmax" else 1)

    def forward(self, ids, lengths):
        e, valid = self.embedder(ids)
        m = valid.unsqueeze(-1).to(e.dtype)
        total = (e * m).sum(1)
        mean = total / lengths.clamp_min(1).unsqueeze(-1).to(e.dtype)
        if self.pooling == "sum":
            return total
        if self.pooling == "mean":
            return mean
        mx = e.masked_fill(~valid.unsqueeze(-1), float("-inf")).amax(1)
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))     # empty text -> 0
        return mx if self.pooling == "max" else torch.cat([mean, mx], -1)


class RNNEncoder(nn.Module):
    """LSTM or GRU over the tokens; the text vector is the final state or a pooling of all states."""

    def __init__(self, embedder: Embedder, cell: str, hidden: int, bidirectional: bool, layers: int, output: str):
        super().__init__()
        self.embedder, self.output = embedder, output
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(embedder.out_dim, hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional)
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, ids, lengths):
        e, valid = self.embedder(ids)
        packed = pack_padded_sequence(e, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
        states, last = self.rnn(packed)
        if self.output == "last":
            h = last[0] if isinstance(last, tuple) else last          # LSTM returns (h, c)
            return torch.cat([h[-2], h[-1]], -1) if self.rnn.bidirectional else h[-1]
        states, _ = pad_packed_sequence(states, batch_first=True, total_length=ids.shape[1])
        if self.output == "max":
            return states.masked_fill(~valid.unsqueeze(-1), float("-inf")).amax(1).nan_to_num(0.0, neginf=0.0)
        m = valid.unsqueeze(-1).to(states.dtype)
        return (states * m).sum(1) / lengths.clamp_min(1).unsqueeze(-1).to(states.dtype)


def mlp_head(in_dim: int, hidden: int, depth: int, activation: str, dropout: float) -> nn.Sequential:
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Dropout(dropout), nn.Linear(d, hidden), ACTIVATIONS[activation]()]
        d = hidden
    return nn.Sequential(*layers, nn.Dropout(dropout), nn.Linear(d, 3))


class PairClassifier(nn.Module):
    """features: 'full' = [u; v; |u-v|; u*v], 'concat' = [u; v], 'hyp' = [v] (hypothesis only).

    overlap > 0 adds project 1's overlap block: a learned vector for each word type that occurs
    in both sentences, summed (an EmbeddingBag over the shared types).
    """

    def __init__(self, encoder: nn.Module, features: str, head_hidden: int, head_depth: int,
                 activation: str, dropout: float, overlap_vocab: int = 0):
        super().__init__()
        self.encoder, self.features = encoder, features
        d = encoder.out_dim
        in_dim = {"full": 4 * d, "concat": 2 * d, "hyp": d}[features]
        self.overlap = nn.EmbeddingBag(overlap_vocab, head_hidden, mode="sum", padding_idx=0) if overlap_vocab else None
        in_dim += head_hidden if overlap_vocab else 0
        self.head = mlp_head(in_dim, head_hidden, head_depth, activation, dropout)

    def forward(self, p, pl, h, hl, shared=None):
        v = self.encoder(h, hl)
        if self.features == "hyp":
            parts = [v]
        else:
            u = self.encoder(p, pl)
            parts = [u, v] if self.features == "concat" else [u, v, (u - v).abs(), u * v]
        if self.overlap is not None:
            parts.append(self.overlap(shared))
        return self.head(torch.cat(parts, -1))
