"""Decode with one or more checkpoints, ensembling by averaging the per-step
probability distributions (this is what the paper's OpenNMT ensemble does).

Pass several checkpoints to ``--models`` -- e.g. the 50k/100k/150k/200k
checkpoints from each of your 3 seeds -- to reproduce the paper's ensemble.

Example:
    python decode.py --models runs/seed1/ckpt_50000.pt runs/seed1/ckpt_100000.pt ... \
        --src prepared/dev.src --out dev.pred --beam 5
"""


import torch
import torch.nn.functional as F

from src.vocab import Vocab, tokenize_source, PAD_ID, BOS_ID, EOS_ID
from src.transformer import Transformer


def load_models(paths, device):
    models, src_vocab, tgt_vocab = [], None, None
    for p in paths:
        ckpt = torch.load(p, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        sv = Vocab.from_dict(ckpt["src_vocab"])
        tv = Vocab.from_dict(ckpt["tgt_vocab"])
        if src_vocab is None:
            src_vocab, tgt_vocab = sv, tv
        else:
            assert len(sv) == len(src_vocab) and len(tv) == len(tgt_vocab), \
                "ensemble checkpoints must share the same vocabulary"
        m = Transformer(len(sv), len(tv), d_model=cfg["d_model"],
                        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
                        n_layers=cfg["n_layers"], dropout=0.0, pad_id=PAD_ID).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models.append(m)
    return models, src_vocab, tgt_vocab


@torch.no_grad()
def beam_search(models, src_ids, tgt_vocab, device, beam=5, max_len=64, alpha=0.6):
    """Beam search over an ensemble for a single source sequence."""
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad = src.eq(PAD_ID)
    memories = [m.encode(src, src_pad) for m in models]

    # each beam: (tokens list, cumulative logprob)
    beams = [([BOS_ID], 0.0)]
    finished = []

    for _ in range(max_len):
        if not beams:
            break
        nb = len(beams)
        seqs = torch.tensor([b[0] for b in beams], dtype=torch.long, device=device)
        tgt_pad = seqs.eq(PAD_ID)

        # average softmax probabilities across the ensemble at the last position
        avg_probs = torch.zeros(nb, len(tgt_vocab), device=device)
        for m, mem in zip(models, memories):
            mem_b = mem.expand(nb, -1, -1)
            src_pad_b = src_pad.expand(nb, -1)
            dec = m.decode(seqs, mem_b, src_pad_b, tgt_pad)
            logits = m.generator(dec[:, -1])           # (nb, V)
            avg_probs += F.softmax(logits, dim=-1)
        avg_probs /= len(models)
        logprobs = torch.log(avg_probs + 1e-9)

        # expand every beam by the top-`beam` next tokens
        topv, topi = logprobs.topk(beam, dim=-1)
        candidates = []
        for bi, (toks, score) in enumerate(beams):
            for k in range(beam):
                tok = topi[bi, k].item()
                cand = (toks + [tok], score + topv[bi, k].item())
                candidates.append(cand)
        candidates.sort(key=lambda x: x[1], reverse=True)

        beams = []
        for toks, score in candidates:
            if toks[-1] == EOS_ID:
                lp = ((len(toks)) ** alpha)            # length normalization
                finished.append((toks, score / lp))
            else:
                beams.append((toks, score))
            if len(beams) >= beam:
                break

    if not finished:  # nothing emitted EOS within max_len; fall back to best beam
        finished = [(t, s / (len(t) ** alpha)) for t, s in beams]
    best = max(finished, key=lambda x: x[1])[0]
    ids = [t for t in best if t not in (BOS_ID, EOS_ID, PAD_ID)]
    return tgt_vocab.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    models, src_vocab, tgt_vocab = load_models(args.models, args.device)
    print(f"loaded {len(models)} model(s); decoding {args.src} with beam={args.beam}")

    preds = []
    with open(args.src, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    for i, line in enumerate(lines, 1):
        src_ids = src_vocab.encode(tokenize_source(line))
        phonemes = beam_search(models, src_ids, tgt_vocab, args.device,
                               beam=args.beam, max_len=args.max_len)
        preds.append(" ".join(phonemes))
        if i % 200 == 0:
            print(f"  decoded {i}/{len(lines)}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(preds) + "\n")
    print(f"wrote {len(preds)} predictions to {args.out}")

