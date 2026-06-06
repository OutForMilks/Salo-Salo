"""Train one multilingual G2P Transformer.

Run this several times with different ``--seed`` values and keep the
intermediate checkpoints; ``decode.py`` then ensembles them.  Defaults follow
the paper (Vaswani-style hyper-parameters, 200k steps, checkpoints every 50k).

Example:
    python train.py --data prepared --out runs/seed1 --seed 1 \
        --steps 200000 --save-every 50000
"""

import argparse
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.vocab import build_vocabs, save_vocabs, PAD_ID
from src.dataset import G2PDataset, collate
from src.transformer import Transformer


class NoamLR:
    """Vaswani et al. learning-rate schedule: warmup then inverse-sqrt decay."""

    def __init__(self, optimizer, d_model, warmup):
        self.opt = optimizer
        self.d_model = d_model
        self.warmup = warmup
        self.step_num = 0

    def step(self):
        self.step_num += 1
        s = self.step_num
        lr = (self.d_model ** -0.5) * min(s ** -0.5, s * self.warmup ** -1.5)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.step()


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing, ignoring padding positions."""

    def __init__(self, vocab_size, pad_id, smoothing=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, logits, target):
        # logits: (B, T, V); target: (B, T)
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
        n_tokens = (~mask).sum().clamp(min=1)
        return loss.sum() / n_tokens


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


