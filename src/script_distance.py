"""
Descriptor-based script distance: label each language with its writing system
(ISO 15924, e.g. 'Latn', 'Cyrl', 'Jpan') and score 0 if two languages share a
script, 1 if they do not.
 
This is a coarser stand-in for Deri & Knight's character-name cosine distance.
Theirs captures partial overlap between related scripts; this captures only
"same writing system or not", which is the dominant signal for deciding whether
a g2p model can transfer. The win is that it needs no text per language: the
script label comes from CLDR's likely-subtags data via `langcodes`.
 
    pip install langcodes language_data
"""
 
from functools import lru_cache
import langcodes
 
 
@lru_cache(maxsize=None)
def language_script(code):
    """ISO 15924 script for a language code (e.g. 'tgl' -> 'Latn').
 
    Returns None if the script cannot be determined.
    """
    try:
        return langcodes.Language.get(code).maximize().script
    except Exception:
        return None
 
 
def script_distance_by_label(s1, s2):
    """0.0 if same script, 1.0 if different, 1.0 if either is unknown."""
    if s1 is None or s2 is None:
        return 1.0
    return 0.0 if s1 == s2 else 1.0
 
 
def script_row_auto(target, candidates):
    """{lang_id: script_distance} from `target` to each candidate, using
    CLDR script labels. Candidates with no resolvable script are skipped."""
    ts = language_script(target)
    row = {}
    for c in candidates:
        if c == target:
            continue
        cs = language_script(c)
        if cs is not None:
            row[c] = script_distance_by_label(ts, cs)
    return row