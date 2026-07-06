"""Train the multilingual G2P Transformer on a TPU via PyTorch/XLA.

XLA compiles one graph per unique tensor shape and recompiles whenever a shape changes,
so the per-batch padding used in ``train.py`` would trigger constant recompilation on a
TPU.  Here every batch is padded to fixed ``max_src``/``max_tgt`` lengths (computed
once from the training data) and ``drop_last=True`` keeps the batch dimension
constant, so XLA compiles a single graph and reuses it every step.

Designed for free Colab/Kaggle TPU (single core by default -- simplest and
robust).  Decoding still belongs on CPU/GPU: run ``decode.py`` separately; the
checkpoints saved here load there unchanged.

Run several seeds, then ensemble the checkpoints with decode.py
"""


import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.vocab import build_vocabs, save_vocabs, PAD_ID
from src.dataset import G2PDataset
from src.transformer import Transformer


# --------------------------------------------------------------------------
# Static-shape padding: this is the key change for TPU.
# --------------------------------------------------------------------------
def make_fixed_collate(max_src, max_tgt):
    """Return a collate fn that pads every batch to fixed lengths."""
    def collate(batch):
        B = len(batch)
        src = torch.full((B, max_src), PAD_ID, dtype=torch.long)
        tgt_in = torch.full((B, max_tgt), PAD_ID, dtype=torch.long)
        tgt_out = torch.full((B, max_tgt), PAD_ID, dtype=torch.long)
        for i, (s, t) in enumerate(batch):
            s = s[:max_src]
            ti, to = t[:-1][:max_tgt], t[1:][:max_tgt]
            src[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            tgt_in[i, : len(ti)] = torch.tensor(ti, dtype=torch.long)
            tgt_out[i, : len(to)] = torch.tensor(to, dtype=torch.long)
        src_pad = src.eq(PAD_ID)
        tgt_pad = tgt_in.eq(PAD_ID)
        return src, tgt_in, tgt_out, src_pad, tgt_pad
    return collate


def make_dynamic_collate():
    """Return a collate fn that pads each batch only to its own max lengths.

    GPU-only alternative to ``make_fixed_collate``: eager CUDA kernels don't
    care about varying shapes, and most words are far shorter than the global
    max (mean src length ~8 vs. a fixed 48), so per-batch padding skips most
    of the padding compute.  Keep the fixed version on XLA, where every new
    shape would trigger a recompile.
    """
    def collate(batch):
        max_src = max(len(s) for s, _ in batch)
        max_tgt = max(len(t) - 1 for _, t in batch)  # decoder input length
        return make_fixed_collate(max_src, max_tgt)(batch)
    return collate


def compute_max_lengths(dataset, pad_to_multiple=8):
    """Largest source / decoder-input lengths in the data.

    Rounded up to a multiple (TPUs like dimensions that are multiples of 8/128;
    8 is a safe, cheap choice here) so the single compiled graph is MXU-friendly.
    """
    max_src = max(len(s) for s, _ in dataset.examples)
    max_tgt = max(len(t) - 1 for _, t in dataset.examples)  # decoder input length

    def roundup(n):
        return ((n + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple

    return roundup(max_src), roundup(max_tgt)


class NoamLR:
    """Vaswani LR schedule. Uses xm.optimizer_step on XLA so the graph executes
    and gradients are consolidated; plain optimizer.step() otherwise."""

    def __init__(self, optimizer, d_model, warmup, xla=False):
        self.opt = optimizer
        self.d_model = d_model
        self.warmup = warmup
        self.xla = xla
        self.step_num = 0
        if xla:
            import torch_xla.core.xla_model as xm
            self._xm = xm

    def step(self):
        self.step_num += 1
        s = self.step_num
        lr = (self.d_model ** -0.5) * min(s ** -0.5, s * self.warmup ** -1.5)
        for g in self.opt.param_groups:
            g["lr"] = lr
        if self.xla:
            self._xm.optimizer_step(self.opt)
        else:
            self.opt.step()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, pad_id, smoothing=0.1):
        super().__init__()
        self.pad_id, self.smoothing, self.vocab_size = pad_id, smoothing, vocab_size

    def forward(self, logits, target):
        logits = logits.view(-1, self.vocab_size)
        target = target.reshape(-1)
        logprobs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(logprobs, self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            true_dist[:, self.pad_id] = 0
            mask = target == self.pad_id
            true_dist[mask] = 0
        loss = torch.sum(-true_dist * logprobs, dim=1)
        return loss.sum() / (~mask).sum().clamp(min=1)


def infinite_loader(loader):
    while True:
        for b in loader:
            yield b


