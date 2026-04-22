# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

Run all notebooks:
```bash
jupyter notebook notebooks/
```

## Data Paths

Data lives in `docs/input/` (not `data/` as README suggests — `data/` is gitignored):
- Train: `docs/input/train/imdb_reviews_train.csv` — 21,754 rows, cols: `text`, `label` (pos/neg), `nr_tokens`
- Test: `docs/input/test/imdb_reviews_test.csv` — 21,996 rows, same columns
- NRC Lexicon: `data/en/NCR-lexicon.csv` — cols: `English`, `Positive`, `Negative` + emotion columns (obtain from Moodle)
- LaTeX report template: `docs/LaTEX_Template_EN.tex` (amsbook class, uses `\chapter` as top level)

## Architecture

All notebooks import shared helpers from `src/utils.py`:
- `load_data(split)` — loads train/test CSV, returns `(texts, labels)`
- `evaluate_predictions(y_true, y_pred)` — returns dict with accuracy, precision, recall, F1
- `preprocess_text(text, lowercase, remove_stopwords, lemmatize, handle_negation)` — configurable pipeline
- `save_results(name, metrics, conditions)` — appends one row to `results/all_results.csv`

Notebook execution order (each maps to a task in `docs/Trabalho.md`):

| Notebook | Task | Description |
|----------|------|-------------|
| `00_dataset_selection` | — | Dataset comparison (done) |
| `01_eda` | — | Data characterisation for report section 3 |
| `02_lexicon_rules` | 2.1.1 | TextBlob, VADER, Stanza |
| `03_transformer_baseline` | 2.1.2 | Pre-trained DistilBERT (no fine-tuning) |
| `04_nrc_lexicon` | 2.2 | NRC lexicon ± negation handling |
| `05_classical_ml` | 2.3 | LR, NB, SVM with BoW / TF-IDF |
| `06_transformer_finetuning` | 2.3 | Fine-tune DistilBERT on train set |
| `07_llm_prompting` | 2.4 | 3 prompt strategies on 200-sample subset |

All experiment results accumulate in `results/all_results.csv`.

## Project Context

- Course: TMCD 2025/2026, ISCTE — Dep. de Ciências e Tecnologias da Informação
- Assignment: `docs/Trabalho.md` · group of 3 (n=3) → need ≥6 articles, ≥3 lexicon tools
- Report: max 13 pages (10+n), PDF using `docs/LaTEX_Template_EN.tex`
- Submission: 24 April 2026 · Presentations: 29–30 April 2026
- Dataset is perfectly balanced (50/50 pos/neg) → accuracy is the primary metric; also report precision, recall, F1
