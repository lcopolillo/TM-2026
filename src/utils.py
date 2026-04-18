"""Shared utilities for all sentiment analysis notebooks."""

import os
import csv
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_CSV = ROOT / "results" / "all_results.csv"
NRC_LEXICON_PATH = ROOT / "docs" / "Text-Mining-main" / "data" / "NRC-lexicon.csv"

_RESULTS_HEADER = ["task", "approach", "preprocessing", "accuracy", "precision", "recall", "f1", "notes"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(split: str = "test") -> tuple[list[str], list[str]]:
    """Load train or test split. Returns (texts, labels)."""
    assert split in ("train", "test"), "split must be 'train' or 'test'"
    path = DATA_DIR / split / f"imdb_reviews_{split}.csv"
    df = pd.read_csv(path)
    return df["text"].tolist(), df["label"].tolist()


def load_nrc_lexicon() -> dict[str, dict[str, int]]:
    """Load NRC lexicon. Returns {word: {'positive': int, 'negative': int}}."""
    df = pd.read_csv(NRC_LEXICON_PATH)
    df.columns = [c.strip() for c in df.columns]
    lexicon = {}
    for _, row in df.iterrows():
        word = str(row["English"]).strip().lower()
        lexicon[word] = {
            "positive": int(row["Positive"]),
            "negative": int(row["Negative"]),
        }
    return lexicon

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(y_true: list[str], y_pred: list[str]) -> dict:
    """Return accuracy, precision, recall, F1 (macro for multi-class)."""
    labels = sorted(set(y_true))
    avg = "binary" if len(labels) == 2 else "macro"
    pos_label = labels[-1]  # alphabetically last: 'pos' > 'neg'

    kwargs = {"average": avg, "zero_division": 0}
    if avg == "binary":
        kwargs["pos_label"] = pos_label

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, **kwargs), 4),
        "recall": round(recall_score(y_true, y_pred, **kwargs), 4),
        "f1": round(f1_score(y_true, y_pred, **kwargs), 4),
    }

# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(task: str, approach: str, metrics: dict,
                 preprocessing: str = "", notes: str = "") -> None:
    """Append one experiment row to results/all_results.csv."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RESULTS_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "task": task,
            "approach": approach,
            "preprocessing": preprocessing,
            "accuracy": metrics.get("accuracy", ""),
            "precision": metrics.get("precision", ""),
            "recall": metrics.get("recall", ""),
            "f1": metrics.get("f1", ""),
            "notes": notes,
        })
    print(f"Saved: [{task}] {approach} — acc={metrics.get('accuracy')} f1={metrics.get('f1')}")


def load_results() -> pd.DataFrame:
    """Load all results into a DataFrame."""
    if not RESULTS_CSV.exists():
        return pd.DataFrame(columns=_RESULTS_HEADER)
    return pd.read_csv(RESULTS_CSV)

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

import re
import string

_NEGATION_WORDS = frozenset([
    "not", "no", "never", "nor", "neither", "hardly", "barely", "scarcely",
    "n't", "nt",
])
_NEGATION_WINDOW = 3  # tokens to flip after a negation word


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
    handle_negation: bool = False,
) -> str:
    """
    Configurable text preprocessing pipeline.
    Returns a single cleaned string.
    """
    import nltk
    # lazy downloads
    for resource in ("punkt", "stopwords", "wordnet", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords as sw
    from nltk.stem import WordNetLemmatizer

    if lowercase:
        text = text.lower()

    tokens = word_tokenize(text)

    if remove_punctuation:
        tokens = [t for t in tokens if t not in string.punctuation]

    if remove_stopwords:
        stop = sw.words("english")
        # keep negation words even when removing stopwords
        tokens = [t for t in tokens if t not in stop or t in _NEGATION_WORDS]

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    if handle_negation:
        tokens = _apply_negation(tokens)

    return " ".join(tokens)


def _apply_negation(tokens: list[str]) -> list[str]:
    """
    Append _NEG suffix to tokens within a window after a negation word.
    E.g. ["not", "good"] → ["not", "good_NEG"]
    """
    result = []
    neg_counter = 0
    for token in tokens:
        if token in _NEGATION_WORDS:
            result.append(token)
            neg_counter = _NEGATION_WINDOW
        elif neg_counter > 0:
            result.append(token + "_NEG")
            neg_counter -= 1
            if token in string.punctuation:
                neg_counter = 0  # reset at sentence boundary
        else:
            result.append(token)
    return result


def preprocess_corpus(
    texts: list[str], **kwargs
) -> list[str]:
    """Apply preprocess_text to a list of texts with a progress bar."""
    from tqdm import tqdm
    return [preprocess_text(t, **kwargs) for t in tqdm(texts, desc="Preprocessing")]
