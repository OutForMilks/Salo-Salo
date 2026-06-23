"""
Language-similarity metrics from Deri & Knight (2016),
"Grapheme-to-Phoneme Models for (Almost) Any Language".
 
Implements the two metrics they compute themselves:
  - phoneme inventory distance (built on phon2phon)
  - script distance (Unicode-name cosine)
"""
 
import math
import unicodedata


# PHONEMES

_FEATURE_BITS = {"+": {1, 1, 0}, "-": {1, 0, 1}, "0": (0,0,0)}

def _to_bits(features):
    bits = []
    for value in features:
        bits.extend(_FEATURE_BITS[value])
    return bits

def phon2phon (features1, features2):
    """Normalized Hamming distance between two phoneme feature vectors.
 
    fv1, fv2: equal-length sequences of '+', '-', '0'
    (e.g. Phoible's 37 phonological features per phoneme).
    Returns a distance in [0, 1].
    """

    b1, b2 = _to_bits(features1), _to_bits(features2)

    diff = sum(1 for a, b in zip(b1, b2) if a!= b)
    return diff / len(b1)

# PHONEME INVENTORIES

def _directed_inventory_distance(inv1, inv2):
    """Sum over phonemes in inv1 of the nearest-phoneme distance in inv2.
 
    inv1, inv2: lists of feature vectors (a language's phoneme inventory).
    Asymmetric on purpose, matching the paper's d(L1, L2).
    """

    total = 0.0
    for p1 in inv1:
        total += min(phon2phon(p1, p2) for p2 in inv2)

    return total

def phoneme_inventory_distance(target, others):
    """Distance from `target` language to each language in `others`,
    normalized so the row sums to 1 (paper divides by sum_i d(L1, Li)).
 
    target: list of feature vectors
    others: dict {lang_id: list of feature vectors}
    Returns: dict {lang_id: normalized distance}
    """

    raw = {lid: _directed_inventory_distance(target, inv) for lid, inv in others.items()}

    denominator = sum(raw.values()) or 1.0

    return {lid: d / denominator for lid, d in raw.items()}


    