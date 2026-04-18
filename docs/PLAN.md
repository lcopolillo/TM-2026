# Action Plan — Sentiment Analysis (IMDB Reviews)

**Group:** Luiza Coelho, Pedro Louro, Tiago Vieira (n = 3)
**Dataset:** IMDB reviews — 21,754 train / 21,996 test · balanced 50/50 pos/neg
**Deadline:** 24 April 2026 · Presentations: 29-30 April

---

## Repository Structure (to create)

```
notebooks/
  00_dataset_selection.ipynb   ← already done
  01_eda.ipynb
  02_lexicon_rules.ipynb
  03_transformer_baseline.ipynb
  04_nrc_lexicon.ipynb
  05_classical_ml.ipynb
  06_transformer_finetuning.ipynb
  07_llm_prompting.ipynb
src/
  utils.py                     ← shared functions for all notebooks
results/
  all_results.csv              ← one row per experiment
docs/
  Trabalho.md / Trabalho.pdf
  input/train/imdb_reviews_train.csv
  input/test/imdb_reviews_test.csv
  Text-Mining-main/data/NRC-lexicon.csv
```

---

## Step 0 — Setup

- **Update `requirements.txt`** — add: `textblob`, `vaderSentiment`, `stanza`, `transformers`, `torch`, `datasets`, `evaluate`, `accelerate`, `openai`, `wordcloud`, `spacy`
- **Create `src/utils.py`** with shared functions used by all notebooks:
  - `load_data(split)` — loads train or test CSV
  - `evaluate_predictions(y_true, y_pred)` — returns accuracy, precision, recall, F1
  - `preprocess_text(text, lowercase, remove_stopwords, lemmatize, handle_negation)` — configurable pipeline
  - `save_results(name, metrics, conditions)` — appends a row to `results/all_results.csv`
- **Create `CLAUDE.md`** at project root with commands, data paths, and architecture overview

---

## Step 1 — EDA (`01_eda.ipynb`)

*For report section 3 — "Dados utilizados"*

- Class distribution chart
- Token count distribution per class (avg ~179 words)
- Most frequent words (word cloud)
- Sample reviews (positive and negative)

---

## Step 2 — Task 2.1.1: Lexicon & Rule-based (`02_lexicon_rules.ipynb`)

Apply **3 tools** to the full test set (21,996 reviews):

| Tool | Mapping logic |
|------|--------------|
| **TextBlob** | `polarity > 0` → pos, else → neg |
| **VADER** | `compound ≥ 0.05` → pos, else → neg |
| **Stanza** | built-in sentiment pipeline, map to pos/neg |

Metrics per tool: accuracy, precision, recall, F1.
Expected range: 65–75% accuracy (IMDB reviews are long; these tools are optimised for short text — document this in the report).

---

## Step 3 — Task 2.1.2: Transformer Baseline (`03_transformer_baseline.ipynb`)

Pre-trained model, **no fine-tuning**:

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis",
                      model="distilbert-base-uncased-finetuned-sst-2-english")
```

- Batch inference on test set (batch_size=32)
- Map POSITIVE/NEGATIVE → pos/neg
- Expected accuracy: ~85–88%
- This is the transformer *baseline* before fine-tuning

---

## Step 4 — Task 2.2: NRC Lexicon Classifier (`04_nrc_lexicon.ipynb`)

Lexicon: `docs/Text-Mining-main/data/NRC-lexicon.csv` (columns used: `English`, `Positive`, `Negative`)

**Experiment 1 — no negation:**
1. Tokenise + lowercase + lemmatise (NLTK WordNetLemmatizer)
2. Count positive and negative tokens via NRC lookup
3. Assign `pos` if total_pos > total_neg, else `neg`

**Experiment 2 — with negation handling:**
- Same as Exp 1, but after negation words (`not`, `no`, `never`, `n't`, `nor`, `neither`, `hardly`, `barely`, `scarcely`), flip polarity of the next 3 tokens

Compare results of both experiments. Expected accuracy: 68–76%.

> Note: n=3 group only needs 1 lexicon. Optionally add SentiWordNet as second lexicon for extra depth.

---

## Step 5 — Task 2.3a: Classical ML (`05_classical_ml.ipynb`)

**Preprocessing pipeline** (configurable flags in `src/utils.py`):
- Tokenisation → lowercase → stopword removal → lemmatisation → optional negation marking (`_NEG` suffix)

**Feature representations:**
| Representation | Library | Config |
|----------------|---------|--------|
| Bag-of-Words | `CountVectorizer` | top 10K unigrams |
| TF-IDF | `TfidfVectorizer` | top 10K, unigrams + bigrams |
| Embeddings | avg GloVe/Word2Vec | 100-dim vectors |

**Models** (trained on train set, 5-fold CV, evaluated on test):
| Model | Hyperparams to try |
|-------|-------------------|
| Logistic Regression | C ∈ {0.1, 1.0, 10} |
| Naive Bayes | alpha ∈ {0.1, 1.0} |
| Linear SVM | C ∈ {0.1, 1.0, 10} |

Best expected: **SVM + TF-IDF + lemma + negation ≈ 89–92%**

---

## Step 6 — Task 2.3b: Transformer Fine-tuning (`06_transformer_finetuning.ipynb`)

Fine-tune `distilbert-base-uncased` on IMDB train set:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
```

- max_length=512 (truncate), epochs=3, batch_size=16, lr=2e-5
- Evaluate on test set
- Expected accuracy: **~93–95%** (best result overall)

> Compute note: ~30–90 min on CPU. Use Google Colab GPU if needed. Document resource usage in report.

---

## Step 7 — Task 2.4: LLM Prompting (`07_llm_prompting.ipynb`)

**Subset:** 200 stratified test samples (100 pos + 100 neg) to control API costs.

**API:** OpenRouter free tier (`deepseek/deepseek-chat` or similar free model) or `iaedu.pt`.

**3 prompt strategies:**

**Prompt 1 — Generic:**
```
Classify the sentiment of the following text as "positive" or "negative".
Reply with only one word.

Text: {review}
```

**Prompt 2 — Domain-aware:**
```
You are analyzing movie reviews. Classify the sentiment of the following
movie review as "positive" or "negative". Reply with only one word.

Review: {review}
```

**Prompt 3 — Few-shot (3 pos + 3 neg examples from train set):**
```
Classify movie review sentiment as "positive" or "negative".

Examples:
[POS] {example_pos_1}
[NEG] {example_neg_1}
[POS] {example_pos_2}
[NEG] {example_neg_2}
[POS] {example_pos_3}
[NEG] {example_neg_3}

Now classify: {review}
Reply with only one word: positive or negative.
```

Evaluate each prompt on the 200-sample subset. Include prompts as appendix in the report.

---

## Results Summary

All experiments write to `results/all_results.csv`. Expected final ranking:

| Rank | Approach | Expected Accuracy |
|------|----------|------------------|
| 1 | Fine-tuned DistilBERT | ~93–95% |
| 2 | SVM + TF-IDF + preprocessing | ~89–92% |
| 3 | Few-shot LLM prompting | ~85–90% |
| 4 | Pre-trained DistilBERT (baseline) | ~85–88% |
| 5 | Logistic Regression + TF-IDF | ~85–88% |
| 6 | NRC Lexicon + negation | ~70–76% |
| 7 | Stanza | ~65–75% |
| 8 | TextBlob / VADER | ~65–70% |

---

## Report Structure (Springer LNCS, max 13 pages)

1. **Introdução** — motivation, task, approach overview
2. **Trabalho relacionado** — ≥6 scientific articles (2×n), compare methods and results
3. **Dados utilizados** — IMDB description, stats from EDA notebook, class balance
4. **Trabalho realizado** — one sub-section per task (2.1.1, 2.1.2, 2.2, 2.3, 2.4), preprocessing choices documented
5. **Resultados** — ≤5 tables (one consolidated table recommended); analyse and discuss
6. **Conclusões** — best approach, limitations, future improvements
7. **Bibliografia**

> Last paragraph of abstract: contribution % per member (e.g. Tiago: 34%, Luiza: 33%, Pedro: 33%)

---

## Presentation (8 min max, PDF/PPTX)

Cover: data description · approaches · key results · future work
Submit to `GrupoXX/` folder via https://t.ly/HQo5v before 17h00 on presentation day.
