"""Convert SIGMORPHON-2020-Task-1 style .tsv files into a single pooled,
language-tagged corpus suitable for multilingual training.

Input: one directory containing files named like
    ady_train.tsv  ady_dev.tsv  ady_test.tsv
    arm_train.tsv  arm_dev.tsv  arm_test.tsv
    ...
Each .tsv line is:  <graphemes>\t<space-separated phonemes>
e.g.   front<TAB>f ʁ ɔ̃

Output (in --out-dir): for each split we write two parallel files plus a
per-line language id file::

    train.src   ->  <fre> f r o n t
    train.tgt   ->  f ʁ ɔ̃
    train.lang  ->  fre

The language tag is prepended to the source so a single model knows which
language it is transcribing (Johnson et al., 2017).  The .lang file lets us
slice predictions back out per language at evaluation time.

Usage:
    python prepare_data.py --in-dir DATA_DIR --out-dir prepared \
        --langs ady arm bul dut fre ...
"""

import argparse
import os
import glob


def discover_langs(in_dir):
    langs = set()
    for p in glob.glob(os.path.join(in_dir, "*_train.tsv")):
        base = os.path.basename(p)
        langs.add(base.split("_train.tsv")[0])
    return sorted(langs)


def graphemes(word):
    """Split a written form into graphemes.

    We use simple Unicode-character segmentation, which matches what the paper
    does for alphabets/alphasyllabaries.  Combining marks are kept attached to
    their base character so they are not split off as standalone tokens.
    """
    import unicodedata
    out = []
    for ch in word.strip():
        if unicodedata.combining(ch) and out:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def convert_split(in_dir, out_dir, langs, split):
    import os
    print(os.getcwd())
    src_lines, tgt_lines, lang_lines = [], [], []
    for lang in langs:
        path = os.path.join(in_dir, f"{lang}_{split}.tsv")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                word, phonemes = parts[0], parts[1]
                src_toks = [f"<{lang}>"] + graphemes(word)
                src_lines.append(" ".join(src_toks))
                tgt_lines.append(phonemes.strip())
                lang_lines.append(lang)
                n += 1
        print(f"  {lang} {split}: {n} pairs")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{split}.src"), "w", encoding="utf-8") as f:
        f.write("\n".join(src_lines) + "\n")
    with open(os.path.join(out_dir, f"{split}.tgt"), "w", encoding="utf-8") as f:
        f.write("\n".join(tgt_lines) + "\n")
    with open(os.path.join(out_dir, f"{split}.lang"), "w", encoding="utf-8") as f:
        f.write("\n".join(lang_lines) + "\n")
    print(f"  -> wrote {len(src_lines)} lines to {split}.{{src,tgt,lang}}")

