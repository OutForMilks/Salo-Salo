l distance · PY
"""
URIEL distance rows via the lang2vec package.
 
Deri & Knight use URIEL's genetic and geographic distances (among others) as
part of the averaged lang2lang metric. lang2vec exposes URIEL as feature
vectors; the published distances are cosine distances over those vectors, so
we compute them the same way here.
 
    pip install lang2vec
 
Feature sets used:
    'fam' -> binary language-family membership  -> genetic distance
    'geo' -> vector of geographic anchor coords -> geographic distance
Both are keyed by ISO 639-3 codes, matching the Phoible loader.
"""
 
import math
import lang2vec.lang2vec as l2v
 
# Map a friendly metric name to the lang2vec feature set behind it.
# Deri & Knight average genetic, geographic, syntactic and phonetic distances
# from URIEL. The *_knn sets are kNN-imputed, so they are dense (no missing
# values) and match how lang2vec's precomputed distances are derived.
_FEATURE_SET = {
    "genetic": "fam",
    "geographic": "geo",
    "syntactic": "syntax_knn",
    "phonological": "phonology_knn",
}
 
 
def _is_numeric(vec):
    """
    lang2vec returns '--' strings for languages it has no data for.
    """
    return all(not isinstance(x, str) for x in vec)

def _cosine_distance(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 1.0
    return 1.0 - dot / (n1 * n2)

def uriel_row(metric, target, candidates)
    """
    {cand: distance} from `target` to each candidate for one URIEL metric.
    metric: 'genetic' or 'geographic'.
    target: ISO 639-3 code (e.g. 'tgl').
    candidates: iterable of ISO 639-3 codes.
    Candidates URIEL cannot cover for this metric are silently skipped.
    """
    fs = _FEATURE_SET[metric]
    langs = [target] + [c for c in candidates if c != target]
    feats = l2v.get_features(langs, fs)

    tvec = feats[target]
    if not _is_numeric(tvec):
        raise ValueError(f"URIELD has no '{metric}' data for target {target!r}")
    
    row = {}
    for c in langs[1:]:
        cvec = feats.get(c)
        if cvec is not None and _is_numeric(cvec):
            row[c] = _cosine_distance(tvec, cvec)
    
    return row


def covered_languages():
    """
    ISO 639-3 codes URIEL knows about (use to prefilter candidates).
    """
    return set(l2v.available_uriel_languages())

