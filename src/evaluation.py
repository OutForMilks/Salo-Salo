import torch
from src.vocab import Vocab, tokenize_source, PAD_ID
from src.transformer import Transformer
from src.decode import beam_search, batch_decode, batch_beam_decode

def load_models_fixed(paths, device):
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

        m = Transformer(len(sv), len(tv),
                        d_model=cfg["hidden_size"],
                        n_heads=cfg["num_heads"],
                        d_ff=cfg["ff_size"],
                        n_layers=cfg["n_layers"],
                        dropout=0.0, pad_id=PAD_ID).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models.append(m)
    return models, src_vocab, tgt_vocab

def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]


def compute_per(references, hypotheses):
    total_edits = sum(edit_distance(r, h) for r, h in zip(references, hypotheses))
    total_len = sum(len(r) for r in references)
    return total_edits / total_len


def compute_wer(references, hypotheses):
    n_correct = sum(1 for r, h in zip(references, hypotheses) if r == h)
    return 1 - (n_correct / len(references))


def evaluate_checkpoint(model_paths, src_path, tgt_path, device, beam=5, max_len=64):
    models, src_vocab, tgt_vocab = load_models_fixed(model_paths, device)

    with open(src_path, encoding="utf-8") as f:
        src_lines = [l.rstrip("\n") for l in f if l.strip()]
    with open(tgt_path, encoding="utf-8") as f:
        tgt_lines = [l.rstrip("\n") for l in f if l.strip()]

    assert len(src_lines) == len(tgt_lines), "src/tgt line count mismatch"

    references, hypotheses = [], []
    for i, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines), 1):
        src_ids = src_vocab.encode(tokenize_source(src_line))
        pred_phonemes = beam_search(models, src_ids, tgt_vocab, device,
                                     beam=beam, max_len=max_len)
        references.append(tgt_line.split())
        hypotheses.append(pred_phonemes)

        if i % 200 == 0:
            print(f"  decoded {i}/{len(src_lines)}")

    per = compute_per(references, hypotheses)
    wer = compute_wer(references, hypotheses)
    return per, wer, references, hypotheses

def batch_evaluate_checkpoint(model_paths, src_path, tgt_path, device, batch_size=64, max_len=64):
    models, src_vocab, tgt_vocab = load_models_fixed(model_paths, device)

    with open(src_path, encoding="utf-8") as f:
        src_lines = [l.rstrip("\n") for l in f if l.strip()]
    with open(tgt_path, encoding="utf-8") as f:
        tgt_lines = [l.rstrip("\n") for l in f if l.strip()]
    assert len(src_lines) == len(tgt_lines)

    references, hypotheses = [], []
    for start in range(0, len(src_lines), batch_size):
        batch_src_lines = src_lines[start:start + batch_size]
        batch_tgt_lines = tgt_lines[start:start + batch_size]

        encoded = [src_vocab.encode(tokenize_source(l)) for l in batch_src_lines]
        max_s_len = max(len(e) for e in encoded)

        src_padded = torch.full((len(encoded), max_s_len), PAD_ID, dtype=torch.long)
        for i, e in enumerate(encoded):
            src_padded[i, :len(e)] = torch.tensor(e, dtype=torch.long)
        src_padded = src_padded.to(device)
        src_pad_mask = src_padded.eq(PAD_ID)

        preds = batch_decode(models, src_padded, src_pad_mask, tgt_vocab, device, max_len=max_len)

        references.extend(l.split() for l in batch_tgt_lines)
        hypotheses.extend(preds)

        print(f"  decoded {min(start + batch_size, len(src_lines))}/{len(src_lines)}")

    per = compute_per(references, hypotheses)
    wer = compute_wer(references, hypotheses)
    return per, wer, references, hypotheses

def batch_evaluate_checkpoint_beam(model_paths, src_path, tgt_path, device,
                                    batch_size=64, beam=5, max_len=64):
    models, src_vocab, tgt_vocab = load_models_fixed(model_paths, device)
    with open(src_path, encoding="utf-8") as f:
        src_lines = [l.rstrip("\n") for l in f if l.strip()]
    with open(tgt_path, encoding="utf-8") as f:
        tgt_lines = [l.rstrip("\n") for l in f if l.strip()]
    assert len(src_lines) == len(tgt_lines)

    references, hypotheses = [], []
    for start in range(0, len(src_lines), batch_size):
        batch_src_lines = src_lines[start:start + batch_size]
        batch_tgt_lines = tgt_lines[start:start + batch_size]

        encoded = [src_vocab.encode(tokenize_source(l)) for l in batch_src_lines]
        max_s_len = max(len(e) for e in encoded)
        src_padded = torch.full((len(encoded), max_s_len), PAD_ID, dtype=torch.long)
        for i, e in enumerate(encoded):
            src_padded[i, :len(e)] = torch.tensor(e, dtype=torch.long)
        src_padded = src_padded.to(device)
        src_pad_mask = src_padded.eq(PAD_ID)

        preds = batch_beam_decode(models, src_padded, src_pad_mask, tgt_vocab, device,
                                   beam=beam, max_len=max_len)

        references.extend(l.split() for l in batch_tgt_lines)
        hypotheses.extend(preds)
        print(f"  decoded {min(start + batch_size, len(src_lines))}/{len(src_lines)}")

    per = compute_per(references, hypotheses)
    wer = compute_wer(references, hypotheses)
    return per, wer, references, hypotheses