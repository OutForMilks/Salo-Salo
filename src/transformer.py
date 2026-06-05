"""A Transformer encoder-decoder implemented from scratch in PyTorch.

This is the standard architecture from Vaswani et al. (2017): sinusoidal
positional encodings, multi-head scaled dot-product attention, position-wise
feed-forward blocks, residual connections and post-layer-norm.  No use of
``torch.nn.Transformer`` -- every sub-layer is written out so the mechanics are
visible and modifiable.

Masking convention used throughout: a mask value of ``True`` means "block this
position" (it will receive -inf attention weight before the softmax).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)

        def split(x, lin):
            # (B, L, d_model) -> (B, n_heads, L, d_k)
            return lin(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        q, k, v = split(q, self.w_q), split(k, self.w_k), split(v, self.w_v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask broadcasts to (B, n_heads, Lq, Lk); True -> blocked
            scores = scores.masked_fill(mask, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)                      # (B, h, Lq, d_k)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, mem_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, mem_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8,
                 d_ff=2048, n_layers=6, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ---- masks -----------------------------------------------------------
    @staticmethod
    def _pad_mask(pad):
        # pad: (B, L) bool -> (B, 1, 1, L) so it blocks key positions
        return pad.unsqueeze(1).unsqueeze(2)

    @staticmethod
    def _causal_mask(T, device):
        # (1, 1, T, T) upper-triangular True above the diagonal -> block future
        m = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return m.unsqueeze(0).unsqueeze(1)

    # ---- forward pieces --------------------------------------------------
    def encode(self, src, src_pad):
        x = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        src_mask = self._pad_mask(src_pad)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt_in, memory, src_pad, tgt_pad):
        T = tgt_in.size(1)
        x = self.pos(self.tgt_emb(tgt_in) * math.sqrt(self.d_model))
        tgt_mask = self._causal_mask(T, tgt_in.device) | self._pad_mask(tgt_pad)
        mem_mask = self._pad_mask(src_pad)
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, mem_mask)
        return x

    def forward(self, src, tgt_in, src_pad, tgt_pad):
        memory = self.encode(src, src_pad)
        dec = self.decode(tgt_in, memory, src_pad, tgt_pad)
        return self.generator(dec)            # (B, T, V) logits