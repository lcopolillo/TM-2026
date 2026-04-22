# Text Mining Final Project — Sentiment Analysis

Sentiment analysis project for the MSc curricular unit TMCD (2025/2026, 2nd semester), ISCTE — Instituto Universitário de Lisboa.

The goal is to explore and compare multiple approaches to binary sentiment classification (positive/negative) on the IMDB Movie Reviews dataset.

**Authors:** Luiza Coelho · Pedro Louro · Tiago Vieira

---

## Project Structure

```
TM-2026/
├── data/                        # Dataset (not committed — source: Moodle)
│   ├── train/imdb_reviews_train.csv
│   └── test/imdb_reviews_test.csv
├── docs/                        # Assignment, report, and reference material
│   ├── Trabalho.md              # Assignment specification
│   ├── report.md                # Full report (Markdown)
│   ├── report.tex               # Full report (LaTeX)
│   └── references-analyze.md    # Annotated bibliography
├── models/                      # Fine-tuned model checkpoints (not committed — 2.2 GB)
│   └── distilbert_finetuned/
├── notebooks/                   # One notebook per task, run in order
│   ├── 00_dataset_selection.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_lexicon_rules.ipynb
│   ├── 03_transformer_baseline.ipynb
│   ├── 04_nrc_lexicon.ipynb
│   ├── 05_classical_ml.ipynb
│   ├── 06_transformer_finetuning.ipynb
│   ├── 07_llm_prompting.ipynb
│   └── 08_results_summary.ipynb
├── results/                     # All outputs: metrics CSV, figures, LaTeX table
├── src/
│   └── utils.py                 # Shared utilities (data loading, preprocessing, evaluation)
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.11+

### 1. Clone the repository

```bash
git clone https://github.com/lcopolillo/TM-2026
cd TM-2026
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

Run once after installing:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 5. Add the dataset

Place the IMDB dataset files (available on Moodle) in the `data/` directory:

```
data/
├── train/
│   └── imdb_reviews_train.csv
└── test/
    └── imdb_reviews_test.csv
```

### 6. (Task 2.4 only) Set the Anthropic API key

Create a `.env` file at the project root:

```bash
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

### 7. Run the notebooks

Open the project in VS Code (or run `jupyter notebook`), select the `.venv` Python interpreter, and run notebooks in order (00 → 08). Each notebook saves its results to `results/all_results.csv`.

> **Note — notebook 06:** Fine-tuning DistilBERT takes ~6.5 hours on CPU. Skip and use the pre-trained checkpoint if already available in `models/distilbert_finetuned/`.

---

## Notebooks

| Notebook | Task | Description |
|---|---|---|
| `00_dataset_selection` | — | Dataset statistics and selection |
| `01_eda` | — | Exploratory data analysis (class distribution, word frequencies) |
| `02_lexicon_rules` | 2.1.1 | TextBlob, VADER, Stanza baselines |
| `03_transformer_baseline` | 2.1.2 | Pre-trained DistilBERT (SST-2, zero-shot) |
| `04_nrc_lexicon` | 2.2 | NRC EmoLex classifier ± negation handling |
| `05_classical_ml` | 2.3 | Logistic Regression, Naive Bayes, SVM with BoW / TF-IDF |
| `06_transformer_finetuning` | 2.3 | DistilBERT fine-tuned on IMDB train set |
| `07_llm_prompting` | 2.4 | Claude Haiku — generic, domain-aware, few-shot prompts |
| `08_results_summary` | — | Consolidated results table and figures |

---

## Results Summary

| Approach | Accuracy | F1 |
|---|---|---|
| Claude Haiku — domain prompt | **0.9550** | **0.9557** |
| DistilBERT fine-tuned | 0.9475 | 0.9485 |
| SVM TF-IDF bigrams + negation | 0.9045 | 0.9062 |
| DistilBERT pre-trained (SST-2) | 0.9000 | 0.8997 |
| Stanza | 0.8335 | 0.8167 |
| VADER | 0.7015 | 0.7456 |
| TextBlob | 0.7000 | 0.7628 |
| NRC Lexicon + Negation | 0.6550 | 0.7059 |

> Note: Claude Haiku (notebook 07) requires an Anthropic API key and evaluates on a stratified 500-sample subset (~1 500 API calls, ~15–20 min).

---

## Deactivate the virtual environment

```bash
deactivate
```
