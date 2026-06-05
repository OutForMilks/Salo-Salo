"""Dataset and batching for G2P.

Each example is a (source ids, target ids) pair.  The target is wrapped with
BOS/EOS so the decoder can learn where sequences start and stop.  Batches are
padded to the longest sequence and carry boolean masks marking padding.
"""

import torch
from torch.utils.data import Dataset

from src.vocab import (tokenize_source, tokenize_target, PAD_ID)


class G2PDataset(Dataset):
    def __init__(self, src_path, tgt_path, src_vocab, tgt_vocab):
        self.examples = []
        with open(src_path, encoding="utf-8") as fs, \
             open(tgt_path, encoding="utf-8") as ft:
            for s, t in zip(fs, ft):
                src_ids = src_vocab.encode(tokenize_source(s), add_bos_eos=False)
                tgt_ids = tgt_vocab.encode(tokenize_target(t), add_bos_eos=True)
                if src_ids and len(tgt_ids) > 2:  # skip empty lines
                    self.examples.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch):
    """Pad a list of (src_ids, tgt_ids) into tensors.

    Returns
        src        : (B, S)  source token ids
        tgt_in     : (B, T)  decoder input  (target without final EOS)
        tgt_out    : (B, T)  decoder target (target without initial BOS)
        src_pad    : (B, S)  True where src is padding
        tgt_pad    : (B, T)  True where tgt_in is padding
    """
    src_seqs = [torch.tensor(s, dtype=torch.long) for s, _ in batch]
    # decoder input drops the last token, target drops the first (teacher forcing)
    tgt_in_seqs = [torch.tensor(t[:-1], dtype=torch.long) for _, t in batch]
    tgt_out_seqs = [torch.tensor(t[1:], dtype=torch.long) for _, t in batch]

    src = _pad(src_seqs)
    tgt_in = _pad(tgt_in_seqs)
    tgt_out = _pad(tgt_out_seqs)
    src_pad = src.eq(PAD_ID)
    tgt_pad = tgt_in.eq(PAD_ID)
    return src, tgt_in, tgt_out, src_pad, tgt_pad


def _pad(seqs):
    maxlen = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), maxlen), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out