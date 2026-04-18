# Sentiment Analysis of IMDB Movie Reviews

**A Comparative Study of Lexicon-based, Machine Learning, and Large Language Model Approaches**

TMCD 2025/2026 — 2nd Semester  
Departamento de Ciências e Tecnologias da Informação  
ISCTE — Instituto Universitário de Lisboa

**Luiza Coelho · Pedro Louro · Tiago Vieira** — April 2026

---

## Resumo

Este trabalho explora e compara múltiplas abordagens de análise de sentimento binária (positivo/negativo) aplicadas ao conjunto de dados IMDB Movie Reviews. O conjunto de treino contém 41 750 críticas e o conjunto de teste 2 000 críticas, com distribuição equilibrada entre as duas classes.

Foram implementadas e avaliadas quatro famílias de métodos: (i) ferramentas baseadas em léxicos e regras (TextBlob, VADER e Stanza), com acurácias entre 70% e 83%; (ii) um modelo transformador pré-treinado sem ajuste fino (`distilbert-base-uncased-finetuned-sst-2-english`), que alcançou 90%; (iii) um classificador baseado no léxico NRC EmoLex com e sem tratamento da negação, atingindo 64–65%; (iv) modelos de aprendizagem automática clássica (Regressão Logística, Naive Bayes, SVM) com representações BoW e TF-IDF, chegando a 90,5%, e o DistilBERT com ajuste fino, que obteve 94,8%; (v) utilização do modelo de língua Claude Haiku baseado em instruções, que alcançou 95,5% com uma instrução orientada ao domínio.

Os resultados mostram que abordagens baseadas em instruções com modelos de língua de grande dimensão, sem qualquer treino específico no conjunto de dados, atingiram desempenho comparável ou superior ao ajuste fino de modelos transformadores.

**Contribuições:**
Luiza Coelho: 33,3% — implementação das tarefas 2.1.1 (Stanza), 2.2 (NRC Lexicon) e contribuição para a redação do relatório.
Pedro Louro: 33.3% — implementação das tarefas 2.1.2 (DistilBERT baseline) e 2.3 (modelos clássicos e ajuste fino), análise de resultados.
Tiago Vieira: 33,3% — implementação da tarefa 2.4 (LLM prompting), coordenação do projeto, implementação de utilitários partilhados e redação do relatório.

---

## Abstract

This work explores and compares multiple approaches to binary sentiment analysis (positive/negative) applied to the IMDB Movie Reviews dataset. The training set contains 41 750 reviews and the test set 2 000 reviews, with a balanced class distribution.

Four families of methods were implemented and evaluated: (i) lexicon- and rule-based tools (TextBlob, VADER, and Stanza), achieving 70–83% accuracy; (ii) a pre-trained transformer without fine-tuning (`distilbert-base-uncased-finetuned-sst-2-english`), reaching 90%; (iii) an NRC EmoLex-based classifier with and without negation handling, reaching 64–65%; (iv) classical machine learning models (Logistic Regression, Naive Bayes, SVM) with BoW and TF-IDF representations reaching 90.5%, and fine-tuned DistilBERT achieving 94.8%; (v) instruction-based LLM inference using Claude Haiku, which reached 95.5% with a domain-aware prompt.

The results demonstrate that instruction-based LLMs, without any task-specific training, can match or exceed the performance of fine-tuned transformer models on this dataset.

---

## Table of Contents

- [Sentiment Analysis of IMDB Movie Reviews](#sentiment-analysis-of-imdb-movie-reviews)
  - [Resumo](#resumo)
  - [Abstract](#abstract)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
  - [2. Data](#2-data)
  - [3. Tasks](#3-tasks)
    - [3.1 Baseline — Pre-existing Tools (Task 2.1)](#31-baseline--pre-existing-tools-task-21)
      - [TextBlob](#textblob)
      - [VADER](#vader)
      - [Stanza](#stanza)
      - [DistilBERT Pre-trained (SST-2) — Task 2.1.2](#distilbert-pre-trained-sst-2--task-212)
    - [3.2 Sentiment Lexicon — NRC EmoLex (Task 2.2)](#32-sentiment-lexicon--nrc-emolex-task-22)
    - [3.3 Classical Machine Learning (Task 2.3)](#33-classical-machine-learning-task-23)
    - [3.4 Transformer Fine-tuning (Task 2.3)](#34-transformer-fine-tuning-task-23)
    - [3.5 Generative Models — LLM Prompting (Task 2.4)](#35-generative-models--llm-prompting-task-24)
  - [4. Results — Overall Comparison](#4-results--overall-comparison)
  - [5. Conclusions](#5-conclusions)
    - [Future Work](#future-work)
  - [Appendix A — Prompt Definitions](#appendix-a--prompt-definitions)
    - [Prompt 1 — Generic](#prompt-1--generic)
    - [Prompt 2 — Domain-aware](#prompt-2--domain-aware)
    - [Prompt 3 — Few-shot](#prompt-3--few-shot)

---

## List of Figures

- [Figure 1 — Class distribution across train and test splits](#2-data)
- [Figure 2 — Token count distribution by class (train set)](#2-data)
- [Figure 3 — Top 20 words by class (train set, stopwords removed)](#2-data)
- [Figure 4 — Word clouds — positive and negative reviews](#2-data)
- [Figure 5 — Task 2.1: Accuracy and F1 comparison across baseline tools](#31-baseline--pre-existing-tools-task-21)
- [Figure 6 — DistilBERT pre-trained (SST-2): confidence distribution on test set](#31-baseline--pre-existing-tools-task-21)
- [Figure 7 — Task 2.2: NRC lexicon classifier results with and without negation handling](#32-sentiment-lexicon--nrc-emolex-task-22)
- [Figure 8 — Task 2.3: Classical ML accuracy across feature representations and classifiers](#33-classical-machine-learning-task-23)
- [Figure 9 — Task 2.3: DistilBERT fine-tuning — training loss and validation accuracy per epoch](#34-transformer-fine-tuning-task-23)
- [Figure 10 — Task 2.4: Claude Haiku metrics across the three prompt strategies](#35-generative-models--llm-prompting-task-24)
- [Figure 11 — All approaches ranked by accuracy, colour-coded by task family](#4-results--overall-comparison)

---

## 1. Introduction

Sentiment analysis is the computational task of identifying and extracting subjective information from textual data — most commonly, the overall polarity (positive or negative) expressed by an author toward a subject. It has become one of the most actively researched areas in Natural Language Processing (NLP) and Text Mining, driven by the vast amount of opinionated content produced daily on social media, review platforms, and news outlets.

Movie review datasets, such as IMDB, present a particularly rich testbed for sentiment analysis: the reviews are long (average ≈175 words), linguistically varied, and may contain complex constructions such as irony, negation, and domain-specific terminology. These characteristics make the task simultaneously more challenging and more realistic than classification on short texts such as tweets.

This report describes the work carried out for the TMCD 2025/2026 Sentiment Analysis assignment. The dataset selected was the **IMDB Movie Reviews** corpus, a widely used binary classification benchmark. The following approaches were implemented and compared:

- **Task 2.1** — Pre-existing off-the-shelf tools: three lexicon/rule-based tools (TextBlob, VADER, Stanza) and one pre-trained transformer (DistilBERT SST-2);
- **Task 2.2** — A custom lexicon-based classifier using the NRC Word-Emotion Association Lexicon, with and without negation handling;
- **Task 2.3** — Trained machine learning models: classical approaches (Logistic Regression, Naive Bayes, SVM) with BoW and TF-IDF features, and DistilBERT fine-tuned on the training set;
- **Task 2.4** — Instruction-based inference using Claude Haiku (Anthropic API) with three distinct prompt strategies.

All experiments were implemented in Python and tracked in `results/all_results.csv`. Shared utilities (data loading, preprocessing, evaluation, result logging) are in `src/utils.py`.

---

## 2. Data

The **IMDB Movie Reviews** dataset was distributed as two CSV files — one for training and one for testing — each containing a `text` column (the review) and a `label` column (`pos` or `neg`).

| Split | Total  | Pos    | Neg    | Avg words | Min words | Max words |
|-------|--------|--------|--------|-----------|-----------|-----------|
| Train | 41 750 | 20 700 | 21 050 | 178.4     | 4         | 466       |
| Test  |  2 000 |  1 022 |    978 | 175.3     | 9         | 426       |

Both splits are near-perfectly balanced (≈50/50), which makes the majority-class baseline 51.1% and accuracy a valid primary metric.

![Figure 1 — Class distribution across train and test splits](../results/fig_class_distribution.png)

Reviews are considerably longer than other sentiment datasets: the average of 178 words per review exceeds typical tweet lengths (≈20 words) by almost an order of magnitude.

![Figure 2 — Token count distribution by class (train set)](../results/fig_token_distribution.png)

The most frequent words after stopword removal reflect typical movie review vocabulary. Word clouds for positive and negative reviews reveal the domain-specific vocabulary that lexicon tools partially miss.

![Figure 3 — Top 20 words by class (train set, stopwords removed)](../results/fig_top_words.png)

![Figure 4 — Word clouds — positive (left) and negative (right) reviews](../results/fig_wordclouds.png)

This length has direct consequences for each method:

- **Lexicon-based tools** aggregate polarity over many tokens, making them susceptible to noise from neutral words and reducing the signal-to-noise ratio.
- **Transformer models** truncate reviews longer than 512 tokens (approximately 15% of the dataset).
- **Classical ML models** benefit from the richer per-sample vocabulary, favouring large TF-IDF feature spaces.
- **LLM inference** used reviews truncated to 1 500 characters to keep API costs manageable.

---

## 3. Tasks

### 3.1 Baseline — Pre-existing Tools (Task 2.1)

The first set of experiments applied off-the-shelf tools to the raw test set without any task-specific training. These results establish the performance floor for our subsequent trained models.

#### TextBlob

TextBlob computes a polarity score in the range [−1, 1] by averaging word-level polarities from its built-in lexicon (derived from the Pattern library). A polarity score > 0 is classified as `pos`; ≤ 0 as `neg`. No preprocessing was applied — the tool receives the raw review text.

TextBlob achieved **70.0% accuracy**, with notably high recall (0.944) but low precision (0.640). This imbalance indicates a strong positive bias: the tool tends to classify most reviews as positive, which is typical for tools built on general-purpose web text where positive language is more frequent than in balanced benchmark datasets.

#### VADER

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based tool designed specifically for short social media texts. It produces a compound score in [−1, 1] via a combination of word-level valence and grammatical rules for modifiers (capitalisation, punctuation, degree words). A compound score ≥ 0.05 was mapped to `pos`; all other scores to `neg`.

VADER achieved **70.2% accuracy**, very close to TextBlob. Like TextBlob, it shows high recall (0.856) and low precision (0.660). The tool was not designed for long-form reviews and this limitation is reflected in the results. The aggregation of polarity signals across 175 words per review introduces noise, and the rule-based modifiers were calibrated for tweet-length contexts.

#### Stanza

Stanza applies a full neural NLP pipeline, including a sentence-level sentiment model trained on SST (Stanford Sentiment Treebank). It produces a three-class prediction (negative=0, neutral=1, positive=2) per sentence. We averaged sentence-level scores across the review and applied a threshold of > 1 for `pos`.

Stanza achieved **83.4% accuracy**, clearly outperforming the other lexicon/rule-based tools. Its neural components allow it to capture richer contextual patterns than purely word-list-based approaches. However, it shows the opposite precision–recall trade-off: high precision (0.933) at the cost of lower recall (0.726), meaning it tends to be conservative with positive predictions.

#### DistilBERT Pre-trained (SST-2) — Task 2.1.2

The Hugging Face `pipeline` API was used with `distilbert-base-uncased-finetuned-sst-2-english`, a DistilBERT model fine-tuned on the Stanford Sentiment Treebank (SST-2). No additional training was performed on IMDB data, making this a zero-shot transfer evaluation.

DistilBERT pre-trained achieved **90.0% accuracy**, substantially above all rule-based tools. Despite being trained on short movie snippets from SST-2, the model transfers well to full-length IMDB reviews. Reviews were truncated to 512 tokens to satisfy the model's maximum sequence length.

**Task 2.1 results summary:**

| Approach                        | Acc.   | Prec.  | Rec.   | F1     |
|---------------------------------|--------|--------|--------|--------|
| DistilBERT (pre-trained, SST-2) | 0.9000 | 0.9228 | 0.8777 | 0.8997 |
| Stanza                          | 0.8335 | 0.9333 | 0.7260 | 0.8167 |
| VADER                           | 0.7015 | 0.6604 | 0.8562 | 0.7456 |
| TextBlob                        | 0.7000 | 0.6399 | 0.9442 | 0.7628 |

![Figure 5 — Task 2.1: Accuracy and F1 comparison across baseline tools](../results/fig_lexicon_rules.png)

![Figure 6 — DistilBERT pre-trained (SST-2): confidence distribution on test set](../results/fig_distilbert_baseline_confidence.png)

---

### 3.2 Sentiment Lexicon — NRC EmoLex (Task 2.2)

The NRC Word-Emotion Association Lexicon (EmoLex) contains 14 182 English words annotated with binary flags for positive and negative polarity (plus eight emotion categories). Only the `Positive` and `Negative` columns were used.

**Preprocessing:** reviews were lowercased and lemmatized using NLTK's `WordNetLemmatizer` before lexicon lookup.

**Classification rule:** count the number of positive-annotated and negative-annotated tokens in the review. Assign the class with the higher count; ties default to `neg`.

Two experiments were conducted:

**Experiment 1 — Without negation handling**

Raw polarity counts after lowercasing and lemmatization. The classifier achieved **64.6% accuracy**.

**Experiment 2 — With negation handling (window = 3)**

After any negation trigger word (*not, no, never, n't, nor, neither, hardly, barely, scarcely*), the polarity of the following three tokens is flipped before counting. This converts "not good" into a negative signal instead of a positive one. The classifier achieved **65.5% accuracy**, a +0.9 pp gain.

| Approach                          | Acc.   | Prec.  | Rec.   | F1     |
|-----------------------------------|--------|--------|--------|--------|
| NRC Lexicon + Negation (window=3) | 0.6550 | 0.6254 | 0.8102 | 0.7059 |
| NRC Lexicon (no negation)         | 0.6460 | 0.6160 | 0.8160 | 0.7020 |

![Figure 7 — Task 2.2: NRC lexicon classifier results with and without negation handling](../results/fig_nrc_lexicon.png)

Both results are well above the 51.1% majority baseline but substantially below all other approaches tested. The main limitation is vocabulary coverage: a large fraction of IMDB review vocabulary — character names, genre-specific terms, cinematic jargon — has no NRC polarity annotation, so those tokens contribute nothing to the count. The classifier is also context-free: it assigns the same polarity regardless of how a word is used syntactically.

Negation handling provides a consistent but modest improvement. The window approach is a valid heuristic but is insufficient to capture the complexity of negation in longer sentences.

---

### 3.3 Classical Machine Learning (Task 2.3)

Supervised classifiers were trained on the full 41 750-review training set and evaluated on the 2 000-review test set. All models were implemented with `scikit-learn`, and hyperparameter selection used 5-fold cross-validation.

#### Preprocessing Pipeline

A configurable pipeline was implemented in `src/utils.py` with the following steps, each independently toggleable:

1. **Lowercasing** — normalises capitalisation.
2. **Punctuation removal** — strips non-alphabetic characters.
3. **Stopword removal** — removes NLTK English stopwords.
4. **Lemmatization** — applies NLTK `WordNetLemmatizer` to reduce inflected forms to their base.
5. **Negation marking** — after a negation trigger word (*not, no, never, n't, nor, neither, hardly, barely, scarcely*), appends the suffix `_NEG` to the following three tokens. This makes negated contexts appear as distinct features (e.g., `good` and `good_NEG` become separate vocabulary entries).

#### Feature Representations

- **Bag-of-Words (BoW):** `CountVectorizer` with the top 10 000 most frequent features.
- **TF-IDF unigrams:** `TfidfVectorizer` with unigrams, top 10 000 features.
- **TF-IDF bigrams:** `TfidfVectorizer` with unigrams + bigrams, top 10 000 features.

All representations were built with lowercase + stopword removal + lemmatization as the baseline preprocessing. The best-performing configuration also applied negation marking.

#### Classifiers

- **Logistic Regression (LR):** L2 regularisation, C ∈ {0.1, 1, 10}.
- **Multinomial Naive Bayes (NB):** smoothing parameter α ∈ {0.1, 1.0}.
- **Linear SVM:** hinge loss, C ∈ {0.1, 1, 10}.

| Model          | Features                   | Acc.   | Prec.  | Rec.   | F1     |
|----------------|----------------------------|--------|--------|--------|--------|
| SVM (C=1)      | TF-IDF bigrams + negation  | 0.9045 | 0.9094 | 0.9031 | 0.9062 |
| SVM (C=0.1)    | TF-IDF bigrams + negation  | 0.9035 | 0.9060 | 0.9051 | 0.9055 |
| LR (C=1)       | TF-IDF bigrams             | 0.9000 | 0.9014 | 0.9031 | 0.9022 |
| SVM (C=1)      | TF-IDF unigrams            | 0.9000 | 0.9014 | 0.9031 | 0.9022 |
| LR (C=1)       | TF-IDF unigrams            | 0.8925 | 0.8929 | 0.8973 | 0.8951 |
| LR (C=1)       | BoW                        | 0.8825 | 0.8885 | 0.8806 | 0.8845 |
| LR (C=1)       | BoW (lowercase only)       | 0.8815 | 0.8875 | 0.8796 | 0.8835 |
| NB (α=1)       | BoW                        | 0.8580 | 0.8758 | 0.8415 | 0.8583 |
| SVM (C=1)      | BoW                        | 0.8540 | 0.8614 | 0.8513 | 0.8563 |

![Figure 8 — Task 2.3: Classical ML accuracy across feature representations and classifiers](../results/fig_classical_ml.png)

**Feature representation matters more than classifier choice.** Moving from BoW to TF-IDF unigrams gains ≈+1 pp for LR; adding bigrams gains a further +0.8 pp. Bigrams capture short collocations such as "not bad" or "highly recommended" that unigrams treat as independent signals.

**Negation marking consistently helps.** The best classical model — SVM with TF-IDF bigrams and negation marking — reaches 90.45%, a +0.45 pp gain over the same configuration without negation. This confirms that explicit negation handling at the feature level captures information that the bigram vocabulary alone does not fully encode.

**Naive Bayes underperforms** relative to LR and SVM on the same features. The conditional independence assumption is violated by word co-occurrences in natural text, and NB does not benefit as strongly from TF-IDF weighting.

**GridSearchCV results.** A 5-fold grid search over C ∈ {0.1, 1, 10} for the best SVM configuration (TF-IDF bigrams + negation) selected C = 0.1 as optimal (90.35%), effectively tied with C = 1 (90.45%). This confirms the SVM is not sensitive to regularisation strength on this dataset.

---

### 3.4 Transformer Fine-tuning (Task 2.3)

The model `distilbert-base-uncased` was fine-tuned on the full 41 750-review training set using the Hugging Face `Trainer` API.

**Tokenisation:** reviews were truncated to a maximum of 512 tokens. Dynamic padding was applied per batch using `DataCollatorWithPadding`, avoiding unnecessary padding to the full 512-token length on shorter examples.

**Training configuration:**

| Parameter         | Value                     |
|-------------------|---------------------------|
| Epochs            | 3                         |
| Learning rate     | 2 × 10⁻⁵                 |
| Batch size        | 16                        |
| Weight decay      | 0.01                      |
| Early stopping    | patience = 1 (val. acc.)  |
| Seed              | 42                        |
| Hardware          | CPU (MPS disabled — OOM)  |
| Training loss     | 0.146                     |
| Runtime           | ≈ 6.5 hours               |

The best checkpoint (by validation accuracy at end of each epoch) was loaded for final evaluation.

| Model                                  | Acc.    | Prec.   | Rec.    | F1      |
|----------------------------------------|---------|---------|---------|---------|
| DistilBERT fine-tuned (3 epochs, IMDB) | 0.9475  | 0.9508  | 0.9462  | 0.9485  |
| DistilBERT pre-trained (SST-2)         | 0.9000  | 0.9228  | 0.8777  | 0.8997  |
| **Gain**                               | +0.0475 | +0.0280 | +0.0685 | +0.0488 |

![Figure 9 — Task 2.3: DistilBERT fine-tuning — training loss and validation accuracy per epoch](../results/fig_finetuning_curves.png)

Fine-tuning on IMDB data improves accuracy by 4.75 percentage points. The most significant gain is in recall (+6.9 pp): the SST-2 pre-trained model was overly conservative about positive predictions (trained on shorter SST-2 snippets, it under-counts positive signals in long reviews). Fine-tuning on IMDB data corrects this bias, bringing precision and recall into better balance.

The ≈6.5 h CPU training time is the main practical limitation of this approach. The same training on a modern GPU (e.g., Google Colab T4) typically completes in under 30 minutes.

---

### 3.5 Generative Models — LLM Prompting (Task 2.4)

Due to API cost constraints, this task was evaluated on a **stratified 200-sample subset** of the test set: 100 positive and 100 negative reviews, drawn at random with seed 42.

**Model:** `claude-haiku-4-5-20251001` (Anthropic API), called with `max_tokens=10` and default temperature. Reviews were truncated to 1 500 characters before insertion into the prompt. Zero API errors were encountered across all 600 calls (200 reviews × 3 prompts).

**Response parsing:** the first occurrence of the word `positive` or `negative` (case-insensitive regex) in the model's response determined the predicted label.

Three prompt strategies were tested (full prompts in Appendix A):

**Prompt 1 — Generic**
A concise instruction to classify the sentiment as "positive" or "negative", with no additional context about the domain.

**Prompt 2 — Domain-aware**
Identical to the generic prompt but adds the context that the text is a movie review and that the model is acting as a movie review analyst.

**Prompt 3 — Few-shot**
Provides 3 labelled positive and 3 labelled negative examples from the training set (each truncated to 300 characters) before asking for the classification of the target review.

| Prompt strategy | Acc.       | Prec.  | Rec.       | F1         |
|-----------------|------------|--------|------------|------------|
| Domain-aware    | **0.9550** | 0.9417 | **0.9700** | **0.9557** |
| Generic         | 0.9500     | 0.9412 | 0.9600     | 0.9505     |
| Few-shot        | 0.9500     | 0.9412 | 0.9600     | 0.9505     |

![Figure 10 — Task 2.4: Claude Haiku metrics across the three prompt strategies](../results/fig_llm_prompting.png)

All three strategies achieve ≥ 95% accuracy on the 200-sample subset. The domain-aware prompt is the best performer at 95.5%, outperforming both the generic and few-shot variants by 0.5 percentage points.

The equivalence of the generic and few-shot prompts is a notable finding. Despite providing 6 labelled examples, the few-shot prompt offers no advantage over simply asking the question directly. This suggests that Claude Haiku already has strong domain knowledge about movie review sentiment from pre-training, and that additional examples do not shift its decision boundary on this task. Specifying the domain explicitly (Prompt 2) is more informative than providing examples.

---

## 4. Results — Overall Comparison

The table below summarises the best result from each task, ordered by accuracy.

| Task  | Best approach                     | Acc.       | Prec.  | Rec.   | F1     |
|-------|-----------------------------------|------------|--------|--------|--------|
| 2.4   | Claude Haiku (domain prompt)      | **0.9550** | 0.9417 | 0.9700 | 0.9557 |
| 2.3   | DistilBERT (fine-tuned, 3 epochs) | 0.9475     | 0.9508 | 0.9462 | 0.9485 |
| 2.3   | SVM — TF-IDF bigrams + negation   | 0.9045     | 0.9094 | 0.9031 | 0.9062 |
| 2.1.2 | DistilBERT (pre-trained, SST-2)   | 0.9000     | 0.9228 | 0.8777 | 0.8997 |
| 2.1.1 | Stanza                            | 0.8335     | 0.9333 | 0.7260 | 0.8167 |
| 2.1.1 | VADER                             | 0.7015     | 0.6604 | 0.8562 | 0.7456 |
| 2.1.1 | TextBlob                          | 0.7000     | 0.6399 | 0.9442 | 0.7628 |
| 2.2   | NRC Lexicon + Negation            | 0.6550     | 0.6254 | 0.8102 | 0.7059 |

![Figure 11 — All approaches ranked by accuracy, colour-coded by task family](../results/fig_all_results.png)

The results follow an intuitive progression from weakest to strongest. The NRC lexicon classifier is the weakest (64–65%), constrained by vocabulary coverage and the absence of contextual modelling. The rule-based tools VADER and TextBlob reach ≈70%, while Stanza's neural pipeline achieves 83.4%. The pre-trained DistilBERT SST-2 baseline reaches 90% with no IMDB training at all.

Fine-tuned DistilBERT and Claude Haiku are the two top-performing systems, both exceeding 94.7%. Claude Haiku achieves 95.5% on the 200-sample test subset, representing the highest accuracy in the study.

**Important caveat:** Task 2.4 was evaluated on a 200-sample subset rather than the full 2 000-review test set. The 0.75 pp gap between Claude Haiku and fine-tuned DistilBERT is within the margin of uncertainty at this sample size and should not be interpreted as a definitive superiority claim. A full 2 000-review LLM evaluation would be needed to confirm this difference statistically.

---

## 5. Conclusions

This work carried out a systematic comparison of eight sentiment analysis configurations across four paradigms on the IMDB Movie Reviews dataset. The main findings are:

1. **Lexicon-based tools are fast but limited for long-form text.** TextBlob and VADER reach only 70% accuracy on IMDB reviews. Stanza, which incorporates a neural pipeline, performs substantially better (83.4%), confirming that even nominally "rule-based" tools benefit from neural components. None of the rule-based tools come close to trained models.

2. **The NRC lexicon is the weakest approach tested.** Its 64–65% accuracy reflects limited vocabulary coverage and the inability to capture context. Negation handling provides a consistent but small gain (+0.9 pp) — useful, but not sufficient to bridge the gap to learning-based methods.

3. **Classical ML with TF-IDF and negation marking is competitive with the pre-trained transformer baseline.** SVM with TF-IDF bigrams and negation reaches 90.5% — matching DistilBERT SST-2 — while requiring no GPU and training in minutes rather than hours. For resource-constrained environments this is a strong practical choice.

4. **Fine-tuned DistilBERT is the strongest fully supervised system.** Three epochs of fine-tuning on IMDB data pushes accuracy to 94.8%, a 4.8 pp gain over zero-shot transfer. The improvement is largest in recall, suggesting that domain-specific fine-tuning corrects a positive-class bias present in the SST-2 checkpoint.

5. **Instruction-tuned LLMs achieve top-tier performance with no task-specific training.** Claude Haiku with a domain-aware prompt reaches 95.5% on the 200-sample test subset — the highest single accuracy in this study. The finding that a domain description outperforms few-shot examples suggests the model's prior knowledge of movie review sentiment is already strong; additional examples add marginal information.

### Future Work

- **Evaluate LLMs on the full test set** (2 000 reviews) to confirm the performance gap with fine-tuned models.
- **Combine NRC with a second lexicon** (SentiWordNet or AFINN) to improve vocabulary coverage.
- **Add POS-tag filtering or character n-grams** to the TF-IDF pipeline.
- **Fine-tune on GPU** to enable hyperparameter search and experimentation with larger models (BERT-large, RoBERTa).
- **Aspect-level sentiment analysis** (acting, screenplay, direction) for richer insights.

---

## Appendix A — Prompt Definitions

The following prompts were used verbatim in the Claude Haiku experiments (Task 2.4). Review text was truncated to 1 500 characters before insertion. All calls used `max_tokens=10` and default temperature.

### Prompt 1 — Generic

```
Classify the sentiment of the following text as "positive" or "negative".
Reply with only one word.

Text: {review}
```

### Prompt 2 — Domain-aware

```
You are analyzing movie reviews. Classify the sentiment of the following
movie review as "positive" or "negative".
Reply with only one word.

Review: {review}
```

### Prompt 3 — Few-shot

```
Classify movie review sentiment as "positive" or "negative".

Examples:
[POS] {example_pos_1}
[NEG] {example_neg_1}
[POS] {example_pos_2}
[NEG] {example_neg_2}
[POS] {example_pos_3}
[NEG] {example_neg_3}

Now classify:
{review}

Reply with only one word: positive or negative.
```

The six few-shot examples were sampled from the training set (random seed 42): 3 positive and 3 negative reviews, each truncated to 300 characters within the prompt.
