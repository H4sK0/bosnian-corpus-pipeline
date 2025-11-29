# Bosnian Corpus Pipeline

This repository contains the Python scripts used to construct, clean and
genre-organise the **Bosnian Corpus (v1.0)** published on Zenodo.

- Zenodo dataset (text + PDFs): https://doi.org/10.5281/zenodo.17757098  
- Main paper (EN): `bosnian_corpus_en.pdf`  
- Paper (BS): `bosanski_korpus_bs.pdf`

The goal of this pipeline is to turn several public CLARIN.SI resources
(Sarajevo SMS 1.1, bsWaC 1.1, CLASSLA-web.bs 1.0) into a single,
well-documented, plain-text corpus suitable for entropy, “language energy”
and general NLP experiments.

## Repository contents

Core scripts (all in this repository):

- `bswac_xml_to_text_all.py`  
  Convert original bsWaC XML/vert files to plain text (one sentence per line).

- `classla_vert_to_genre_txt.py`  
  Extract genre-labelled plain text from CLASSLA-web.bs `.vert` files and
  write one text file per super-genre.

- `sms_conllu_to_txt.py`  
  Extract plain text messages from Sarajevo SMS CONLLU files.

- `preclean_for_entropy.py`  
  Minimal pre-cleaning for entropy work (Unicode NFC, whitespace, basic noise).

- `preclean_paper_grade.py`  
  “Paper-grade” cleaning: mojibake fixes, removal of CMS/meta lines, aggressive
  filtering of non-letter lines etc.

- `auto_genre_from_bswac.py`  
  Train a TF–IDF + logistic regression classifier on CLASSLA genres and
  assign super-genre labels to bsWaC text blocks.

- `final_sanity_clean.py`  
  Final pass over all genre files: NFC again, strip non-printing characters,
  drop very short or noisy lines, optional deduplication.

- `bosnian_corpus_merge.py`  
  Merge all super-genre files into a single `bosnian_corpus.txt` file.

You can adapt these scripts to re-run the pipeline on the original CLARIN.SI
resources or on your own Bosnian data.

## Installation

```bash
# create and activate a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
