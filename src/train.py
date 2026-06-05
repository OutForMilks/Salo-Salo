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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with train.src/.tgt, dev.src/.tgt")
    ap.add_argument("--out", required=True, help="output dir for checkpoints")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--save-every", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # Vocabularies are built once from the training data and saved alongside the
    # checkpoints so decoding uses the exact same mapping.
    src_vocab, tgt_vocab = build_vocabs(
        os.path.join(args.data, "train.src"), os.path.join(args.data, "train.tgt"))
    save_vocabs(src_vocab, tgt_vocab, os.path.join(args.out, "vocab.json"))
    print(f"src vocab={len(src_vocab)}  tgt vocab={len(tgt_vocab)}")

    train_ds = G2PDataset(os.path.join(args.data, "train.src"),
                          os.path.join(args.data, "train.tgt"),
                          src_vocab, tgt_vocab)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, drop_last=True)
    print(f"training examples: {len(train_ds)}")

    model = Transformer(len(src_vocab), len(tgt_vocab), d_model=args.d_model,
                        n_heads=args.n_heads, d_ff=args.d_ff,
                        n_layers=args.n_layers, dropout=args.dropout,
                        pad_id=PAD_ID).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    sched = NoamLR(opt, args.d_model, args.warmup)
    criterion = LabelSmoothingLoss(len(tgt_vocab), PAD_ID, args.smoothing)

    model.train()
    data_iter = infinite_loader(loader)
    running, count = 0.0, 0
    for step in range(1, args.steps + 1):
        src, tgt_in, tgt_out, src_pad, tgt_pad = next(data_iter)
        src, tgt_in, tgt_out = src.to(args.device), tgt_in.to(args.device), tgt_out.to(args.device)
        src_pad, tgt_pad = src_pad.to(args.device), tgt_pad.to(args.device)

        logits = model(src, tgt_in, src_pad, tgt_pad)
        loss = criterion(logits, tgt_out)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sched.step()

        running += loss.item()
        count += 1
        if step % args.log_every == 0:
            print(f"step {step:>7}/{args.steps}  loss {running / count:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}")
            running, count = 0.0, 0

        if step % args.save_every == 0 or step == args.steps:
            ckpt = os.path.join(args.out, f"ckpt_{step}.pt")
            torch.save({"model": model.state_dict(),
                        "config": vars(args),
                        "src_vocab": src_vocab.to_dict(),
                        "tgt_vocab": tgt_vocab.to_dict()}, ckpt)
            print(f"  saved {ckpt}")
