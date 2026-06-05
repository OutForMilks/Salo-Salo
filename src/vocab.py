"""Vocabulary and tokenization for multilingual grapheme-to-phoneme conversion.

The source side is a sequence of graphemes (characters) with a language-ID
token prepended, e.g. ``<fre> f r o n t``.  The target side is a sequence of
space-separated phonemes (IPA), exactly as WikiPron / SIGMORPHON provide them,
e.g. ``f ʁ ɔ̃``.

We keep one shared source vocabulary (graphemes + language tokens) and one
shared target vocabulary (phonemes) across all languages.  This shared space is
what lets the model transfer between languages.
"""

import json

# Reserved special tokens. Indices are fixed so they are stable across runs.
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


class Vocab:
    """A simple bidirectional token <-> id mapping."""

    def __init__(self, tokens):
        # ``tokens`` is the list of non-special tokens, sorted for determinism.
        self.itos = list(SPECIALS) + list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, add_bos_eos=False):
        ids = [self.stoi.get(t, UNK_ID) for t in tokens]
        if add_bos_eos:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids, strip_specials=True):
        toks = [self.itos[i] for i in ids]
        if strip_specials:
            toks = [t for t in toks if t not in SPECIALS]
        return toks

    def to_dict(self):
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d):
        v = cls.__new__(cls)
        v.itos = d["itos"]
        v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v


def tokenize_source(line):
    """Split a prepared source line into tokens.

    A prepared line looks like ``<fre> f r o n t`` (already space-separated by
    ``prepare_data.py``), so a plain whitespace split recovers the tokens:
    the language tag plus one token per grapheme.
    """
    return line.strip().split()


def tokenize_target(line):
    """Split a prepared target (phoneme) line on whitespace."""
    return line.strip().split()


def build_vocabs(src_path, tgt_path):
    """Build source and target vocabularies from prepared training files."""
    src_tokens, tgt_tokens = set(), set()
    with open(src_path, encoding="utf-8") as f:
        for line in f:
            src_tokens.update(tokenize_source(line))
    with open(tgt_path, encoding="utf-8") as f:
        for line in f:
            tgt_tokens.update(tokenize_target(line))
    # sorted() makes the vocabulary deterministic regardless of file order.
    return Vocab(sorted(src_tokens)), Vocab(sorted(tgt_tokens))


def save_vocabs(src_vocab, tgt_vocab, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"src": src_vocab.to_dict(), "tgt": tgt_vocab.to_dict()},
                  f, ensure_ascii=False)


def load_vocabs(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return Vocab.from_dict(d["src"]), Vocab.from_dict(d["tgt"])