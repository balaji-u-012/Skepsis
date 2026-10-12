"""
Lightweight text utilities used for ranking and deduplication.

Deliberately dependency-free: no scikit-learn, no sentence-transformers,
no PyTorch. A genuine pretrained neural embedding model would need one of
those (or a GPU-hosted API), which the project's hardware constraints rule
out. Instead this module implements a small, transparent TF-IDF + cosine
similarity scorer in plain Python, which is fast enough for a handful of
candidate abstracts on a CPU-only laptop and needs no training step or
model download.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, Iterable, List

logger = logging.getLogger("skepsis.text")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "for",
    "to", "is", "are", "was", "were", "be", "been", "being", "with",
    "as", "by", "at", "from", "this", "that", "these", "those", "it",
    "its", "into", "we", "our", "using", "based", "via",
    "paper", "study", "approach", "method", "methods", "novel", "model",
    "models", "can", "which", "their", "also", "such", "more", "than",
}

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{1,}")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def term_frequency(tokens: Iterable[str]) -> Dict[str, float]:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {term: count / total for term, count in counts.items()}


def build_idf(documents_tokens: List[List[str]]) -> Dict[str, float]:
    n_docs = len(documents_tokens) or 1
    df = Counter()
    for tokens in documents_tokens:
        for term in set(tokens):
            df[term] += 1
    return {term: math.log((1 + n_docs) / (1 + count)) + 1 for term, count in df.items()}


def tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = term_frequency(tokens)
    return {term: weight * idf.get(term, 0.0) for term, weight in tf.items()}


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    shared_terms = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in shared_terms)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_by_relevance(query: str, documents: List[str]) -> List[float]:
    """Returns a cosine-similarity score (0-1) for each document against query,
    using a corpus-local TF-IDF built from query + all documents together.

    Note: with very short queries (1-2 content words) against long abstracts,
    the absolute cosine value is naturally low (0.1-0.5) even for excellent
    matches. Use these scores for *ranking*, not as an intuitive "match %".
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        logger.warning("Query %r produced no tokens after stopword removal; "
                       "all relevance scores will be 0.", query)
        return [0.0] * len(documents)

    doc_tokens_list = [tokenize(doc) for doc in documents]
    idf = build_idf([query_tokens] + doc_tokens_list)
    query_vec = tfidf_vector(query_tokens, idf)
    scores = []
    for doc_tokens in doc_tokens_list:
        doc_vec = tfidf_vector(doc_tokens, idf)
        scores.append(cosine_similarity(query_vec, doc_vec))
    return scores


def normalize_title(title: str) -> str:
    if not title:
        return ""
    cleaned = re.sub(r"[^a-z0-9\s]", "", title.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def title_similarity(title_a: str, title_b: str) -> float:
    return SequenceMatcher(None, normalize_title(title_a), normalize_title(title_b)).ratio()


def author_overlap(authors_a: List[str], authors_b: List[str]) -> float:
    """Jaccard similarity between two author sets. Using the union as the
    denominator (rather than the smaller set) prevents a single shared
    common surname from producing a spurious 1.0 overlap."""
    if not authors_a or not authors_b:
        return 0.0
    set_a = {a.strip().lower() for a in authors_a if a.strip()}
    set_b = {b.strip().lower() for b in authors_b if b.strip()}
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0