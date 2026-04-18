# References — Sentiment Analysis

Working list of references for the report. Each entry includes a brief note on how it relates to our work.

---

## [1] Maas et al. (2011) — IMDB Dataset

**A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts (2011)**
"Learning Word Vectors for Sentiment Analysis"
*Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011)*, Portland, OR, pp. 142–150.

**Relevance:** Foundational paper that introduced the IMDB Large Movie Review Dataset we use. Their SVM + BoW baseline achieved 87.8% accuracy on 25 000 test reviews; their word-vector model reached 88.9%. Establishes IMDB as a standard benchmark and documents the challenge of long-form reviews vs. short texts.

---

## [2] Pang, Lee & Vaithyanathan (2002) — ML for Sentiment

**B. Pang, L. Lee, and S. Vaithyanathan (2002)**
"Thumbs up? Sentiment Classification using Machine Learning Techniques"
*Proceedings of the 2002 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 79–86.

**Relevance:** One of the earliest ML studies on sentiment classification using movie reviews. Compared Naive Bayes, Maximum Entropy, and SVM with BoW and bigram features. SVMs achieved up to 82.9% accuracy. Also showed the benefit of negation handling as a feature — directly relevant to our Task 2.3 experiments.

---

## [3] Pang & Lee (2008) — Opinion Mining Survey

**B. Pang and L. Lee (2008)**
"Opinion Mining and Sentiment Analysis"
*Foundations and Trends in Information Retrieval*, 2(1–2), pp. 1–135.

**Relevance:** Comprehensive survey of lexicon-based and ML approaches to sentiment analysis. Covers preprocessing choices (negation, stopwords, lemmatization) and evaluation metrics. Informs our method selection across Tasks 2.1–2.3.

---

## [4] Hutto & Gilbert (2014) — VADER

**C. J. Hutto and E. Gilbert (2014)**
"VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text"
*Proceedings of the 8th International AAAI Conference on Weblogs and Social Media (ICWSM-14)*, Ann Arbor, MI.

**Relevance:** Original paper for the VADER tool used in Task 2.1.1. Designed for short social media text with grammatical rules for capitalisation, punctuation, and degree modifiers. Their results on Twitter data confirm VADER's strength on short texts and expected weakness on long-form reviews like IMDB.

---

## [5] Mohammad & Turney (2013) — NRC EmoLex

**S. M. Mohammad and P. D. Turney (2013)**
"Crowdsourcing a Word-Emotion Association Lexicon"
*Computational Intelligence*, 29(3), pp. 436–465.

**Relevance:** Describes the NRC Word-Emotion Association Lexicon (EmoLex) used in Task 2.2. Contains 14 182 English words with binary polarity flags and emotion annotations. Validated on tweet-level tasks; aggregating word polarities provides competitive baselines on short text.

---

## [6] Devlin et al. (2019) — BERT

**J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova (2019)**
"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
*Proceedings of NAACL-HLT 2019*, Minneapolis, MN, pp. 4171–4186.

**Relevance:** Original BERT paper. DistilBERT (Tasks 2.1.2 and 2.3) is derived from it. Establishes the pre-train + fine-tune paradigm and shows that even 2–3 epochs of fine-tuning on task-specific data yields strong results for sequence classification.

---

## [7] Sanh et al. (2019) — DistilBERT

**V. Sanh, L. Debut, J. Chaumond, and T. Wolf (2019)**
"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"
*arXiv preprint arXiv:1910.01108*.

**Relevance:** Directly relevant — we use `distilbert-base-uncased` for fine-tuning (Task 2.3) and `distilbert-base-uncased-finetuned-sst-2-english` as a zero-shot baseline (Task 2.1.2). DistilBERT retains 97% of BERT's language understanding at 40% smaller size.

---

## [8] Brown et al. (2020) — GPT-3 / Few-shot LLMs

**T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan et al. (2020)**
"Language Models are Few-Shot Learners"
*Advances in Neural Information Processing Systems (NeurIPS 2020)*, vol. 33, pp. 1877–1901.

**Relevance:** Establishes the paradigm for instruction-based and few-shot prompting used in Task 2.4. Demonstrates that large language models can perform competitively on NLP tasks via in-context examples without gradient updates — the foundation for our Claude Haiku prompting experiments.

---

## [9] Qi et al. (2020) — Stanza

**P. Qi, Y. Zhang, Y. Zhang, J. Bolton, and C. D. Manning (2020)**
"Stanza: A Python Natural Language Processing Toolkit for Many Human Languages"
*Proceedings of ACL 2020, System Demonstrations*, pp. 101–108.

**Relevance:** Original paper for the Stanza library used in Task 2.1.1. Describes its neural NLP pipeline including the sentiment model applied at sentence level. Stanza outperformed VADER and TextBlob in our experiments (83.4% vs ≈70%) due to its neural components.

---

## [10] Loria (2018) — TextBlob

**S. Loria (2018)**
"TextBlob: Simplified Text Processing"
Available: https://textblob.readthedocs.io/

**Relevance:** Documentation reference for the TextBlob library used in Task 2.1.1. TextBlob uses the Pattern library's lexicon internally to compute polarity scores.

---

## Notes for report integration

When citing in the final PDF (LaTeX version), use numbered citations `[1]`–`[10]` matching the order above. The Markdown report intentionally omits inline citations to keep the text readable during drafting — add them back when exporting to LaTeX.

Articles to potentially add before final submission:
- A study on negation handling in sentiment analysis (to strengthen the 2.2 and 2.3 discussion)
- A study comparing LLMs vs. fine-tuned models for text classification (to strengthen the 2.4 discussion)
