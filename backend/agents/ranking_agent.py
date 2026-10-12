from __future__ import annotations

import datetime
from dataclasses import replace
from typing import List, Optional

from backend.models.schemas import Paper
from backend.utils.text_utils import rank_by_relevance

RELEVANCE_WEIGHT = 0.7
RECENCY_WEIGHT = 0.3
RECENCY_WINDOW_YEARS = 10  # papers older than this get the minimum recency score


def _current_year() -> int:
    """Computed per call so a long-running server stays correct across
    a New Year boundary."""
    return datetime.date.today().year


def _recency_score(year: Optional[int]) -> float:
    if not year:
        return 0.0

    today_year = _current_year()

    # Papers dated in the future are almost always metadata errors (we've
    # seen OpenAlex emit years like 2027/2028). Don't reward them with a
    # perfect recency score — treat them as "this year" so they don't
    # artificially jump to the top of the ranking.
    if year > today_year:
        year = today_year

    age = max(today_year - year, 0)
    if age >= RECENCY_WINDOW_YEARS:
        return 0.0
    return 1 - (age / RECENCY_WINDOW_YEARS)


def filter_by_year(papers: List[Paper], year_from: Optional[int], year_to: Optional[int]) -> List[Paper]:
    if not year_from and not year_to:
        return papers
    filtered = []
    for p in papers:
        if p.year is None:
            continue  # can't verify it's in range, so exclude rather than guess
        if year_from and p.year < year_from:
            continue
        if year_to and p.year > year_to:
            continue
        filtered.append(p)
    return filtered


def _clamp01(value: float) -> float:
    """Guard against rank_by_relevance returning raw TF-IDF values > 1."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def rank_papers(query: str, papers: List[Paper], top_n: int = 5) -> List[Paper]:
    """Scores every candidate with TF-IDF relevance (title + abstract vs.
    query) blended with a recency score, then returns the top N in
    descending order. Deterministic and requires no external model."""
    if not papers:
        return []

    documents = [f"{p.title or ''} {p.abstract or ''}" for p in papers]
    relevance_scores = rank_by_relevance(query, documents)

    scored: List[Paper] = []
    for paper, relevance in zip(papers, relevance_scores):
        # Clamp relevance into [0, 1] so final_score is always a valid
        # percentage the frontend can render without overflow.
        relevance = _clamp01(float(relevance))
        recency = _recency_score(paper.year)
        final = RELEVANCE_WEIGHT * relevance + RECENCY_WEIGHT * recency
        p = replace(
            paper,
            relevance_score=round(relevance, 4),
            recency_score=round(recency, 4),
            final_score=round(final, 4),
        )
        scored.append(p)

    # Stable, deterministic ordering: primary key is final_score, tie-break
    # on recency, then on year (newest first) so equal-score papers have a
    # predictable order.
    scored.sort(
        key=lambda p: (p.final_score or 0.0, p.recency_score or 0.0, p.year or 0),
        reverse=True,
    )
    return scored[:top_n]