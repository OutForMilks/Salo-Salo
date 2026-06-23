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


# SCRIPTS (e.g. LATIN, etc.)
def reduced_char_name(ch):
    """
    Strip script/accent/form identifiers from a Unicode name.
 
    'DEVANAGARI LETTER KA' -> 'LETTER KA'
    'BENGALI LETTER KA'    -> 'LETTER KA'
    Returns None for characters with no usable name.
    """

    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    
    for anchor in ("LETTER", "SIGN", "DIGIT", "VOWEL", "SYLLABLE", "MARK"):
        idx = name.find(anchor)
        if idx != -1:
            return name[idx:]

def char_name_set(text):
    """
    Reduced-character-name feature set for a language sample (e.g. its
    named-entity text). The paper uses languages with >500 NE pairs.
    """

    names = set()
    for ch in text:
        if ch.isalpha():
            n = reduced_char_name(ch)
            if n:
                names.add(n)
    
    return names

def script_distance(set1, set2):
    """
    Cosine distance over binary character-name feature vectors:
        1 - (f1 . f2) / (||f1|| ||f2||)
    set1, set2: reduced-name sets from char_name_set(). Returns [0, 1].
    """

    if not set1 or not set2:
        return 1
    inter = len(set1 & set2)
    return 1.0 - inter / (math.sqrt(len(set1)) * math.sqrt(len(set2)))

def script_row(target_text, others_text):
    """
    {lang_id: script_distance} from a target to each candidate.
 
    target_text: an orthographic text sample for the target language.
    others_text: {lang_id: orthographic text sample}.
    Like the paper, this needs real text per language (they used named-entity
    lists); for a g2p project the grapheme side of your training data works.
    """

    t = char_name_set(target_text)
    return {lid: script_distance(t, char_name_set(s)) for lid, s in others_text.items()}

# LANGUAGE TO LANGUAGE DISTANCE METRIC
def normalize(row):
    """
    Min-max scale a {lang_id: distance} row to [0, 1].
 
    Metrics arrive on different scales (the phoneme row sums to 1, URIEL
    cosines sit in [0, 1], script distance in [0, 1]). Scaling each row to a
    common [0, 1] range before averaging stops any one metric from dominating.
    """

    if not row:
        return {}
    
    lo, hi = min(row.values()), max(row.values())
    if hi == lo:
        return {k: 0.0 for k in row}
    
    return {k: (v-lo) / (hi-lo) for k, v in row.items()}

def lang2lang_average(metric_rows):
    """
    Average several normalized distance metrics into one score per language.
 
    metric_rows: list of dicts, each {lang_id: distance}, one per metric
                 (phoneme inventory, script, URIEL genetic, geographic, ...).
    Returns: dict {lang_id: mean distance}, plus you can sort to get the
             closest related language.
    """    

    langs = set().union(*metric_rows)
    out = {}

    for lid in langs:
        vals = [row[lid] for row in metric_rows if lid in row]
        out[lid] = sum(vals) / len(vals) if vals else 1.0

    return out

def closest_languages(scores, k=3, candidates=None):
    """
    Top-k most similar languages (smallest distance).
    candidates: optional set to restrict to (e.g. same-script high-resource).
    """

    items = [(lid, d) for lid, d in scores.items()
             if candidates is None or lid in candidates
             ]
    return sorted(items, key=lambda x: x[1])[:k]
