# Salo-salo

Multilingual grapheme-to-phoneme (G2P) conversion for **Filipino**, studying
cross-lingual transfer: does adding training data from *related* languages
improve Tagalog G2P, and does *how related* they are matter?

A Transformer encoder-decoder (written from scratch, Vaswani et al. 2017) is
trained on pooled, language-tagged WikiPron pronunciation data. Auxiliary
languages are mixed in with temperature sampling, and every model is scored on
the same held-out Tagalog test set.

## So far

- A Tagalog-only baseline model
- A language-ablation study comparing training mixes: Tagalog + Philippine
  languages, Tagalog + non-Philippine Austronesian languages, and everything
  combined
- Adding Castilian Spanish (a major loanword source for Tagalog) to the best
  mix, with tuning of the language-sampling temperature
- A language-similarity analysis (phoneme inventory, script, and URIEL
  genetic/geographic distances) to relate transfer to language relatedness
