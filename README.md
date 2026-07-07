# Salo-salo: Grapheme-to-Phoneme Conversion Project on Filipino.

**By**: Enrique Aragon, Stephen Borja, Justin Ethan Ching, and Erin Gabrielle Chua.

**Dataset**: Lee, J. L., Ashby, L. F.E., Garza, M. E., Lee-Sikka, Y., Miller, S., Wong, A., McCarthy, A. D., & Gorman, K. (2020). Massively multilingual pronunciation mining with WikiPron [Dataset]. https://github.com/CUNY-CL/wikipron

**Motivation**: Grapheme-to-phoneme (G2P) conversion is a core component of speech technologies such as text-to-speech and automatic speech recognition, yet low-resource languages like Filipino have little pronunciation data to train on. Cross-lingual transfer (pooling training data from related languages) offers a way to close this gap.

**Goal**: By the end of the project, our goal is to determine whether adding training data from *related* languages improves Tagalog G2P conversion, and whether *how related* those languages are matters.

## Set Up

> This project requires [uv](https://docs.astral.sh/uv/getting-started/installation/) as its package manager. The remaining dependencies are handled by uv itself (you can view these dependencies in [`pyproject.toml`](/pyproject.toml)).

1. Simply clone the repository.
2. Run `uv sync` which should catch your environment up.
3. If your text editor does not support JupyterNotebooks run browser-based interface with `uv run --with jupyter jupyter lab`. No need to do this if you're using PyCharm or VSCode.

## Machine Learning Model

We use a **Transformer encoder-decoder** (Vaswani et al., 2017) written from scratch, trained on pooled, language-tagged WikiPron pronunciation data. Auxiliary languages are mixed in with temperature sampling, and every model is scored on the same held-out Tagalog test set.

## Notebooks

View the notebooks enumerated below, also view the notebooks in the order indicated.

1. `g2p.ipynb`: contains the setup, data preparation, and training pipeline for the Transformer G2P model (Colab-ready).
1. `notebooks/language_abl.ipynb`: contains the language-ablation study comparing training mixes — Tagalog + Philippine languages, Tagalog + non-Philippine Austronesian languages, and everything combined — plus the addition of Castilian Spanish (a major loanword source for Tagalog) and tuning of the language-sampling temperature.
1. `notebooks/languages.ipynb`: contains the language-similarity analysis (phoneme inventory, script, and URIEL genetic/geographic distances) relating transfer to language relatedness.

## Data

The pronunciation data used in the project is mined from Wiktionary by the WikiPron project (Lee et al., 2020), available at [github.com/CUNY-CL/wikipron](https://github.com/CUNY-CL/wikipron) under the Apache 2.0 license (the underlying Wiktionary data is CC BY-SA 3.0).

The dataset is also available as `.tsv` files in the repository: per-language WikiPron data in the `wikipron/` directory (grouped into `filipino/` and `austronesian/`), and the Tagalog train/dev/test splits in the `data/` directory.
