from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

NOT_REPORTED = "Not reported"


def _get(d: dict, key: str, default=None):
    return d.get(key, default) if d else default


# ---------------------------------------------------------------------------
# Raw / normalized paper record (output of discovery + verification + ranking)
# ---------------------------------------------------------------------------
@dataclass
class Paper:
    id: str
    title: str
    source: str  # arXiv | Semantic Scholar | OpenAlex | Crossref
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    published_date: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = None

    # scoring, filled in by the ranking agent
    relevance_score: Optional[float] = None
    recency_score: Optional[float] = None
    final_score: Optional[float] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        return cls(
            id=_get(d, "id", ""),
            title=_get(d, "title", ""),
            source=_get(d, "source", ""),
            authors=_get(d, "authors", []) or [],
            year=_get(d, "year"),
            published_date=_get(d, "published_date"),
            doi=_get(d, "doi"),
            url=_get(d, "url"),
            abstract=_get(d, "abstract"),
            venue=_get(d, "venue"),
            citation_count=_get(d, "citation_count"),
            relevance_score=_get(d, "relevance_score"),
            recency_score=_get(d, "recency_score"),
            final_score=_get(d, "final_score"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Structured analysis extracted per paper (the "detailed summary")
# ---------------------------------------------------------------------------
@dataclass
class PaperAnalysis:
    paper_id: str
    abstract: str = NOT_REPORTED
    problem: str = NOT_REPORTED
    methodology: str = NOT_REPORTED
    algorithm: str = NOT_REPORTED
    dataset: str = NOT_REPORTED
    results: str = NOT_REPORTED
    limitations: str = NOT_REPORTED
    novelty: str = NOT_REPORTED
    source_of_analysis: str = "heuristic"

    @classmethod
    def from_dict(cls, d: dict) -> "PaperAnalysis":
        return cls(
            paper_id=_get(d, "paper_id", ""),
            abstract=_get(d, "abstract", NOT_REPORTED),
            problem=_get(d, "problem", NOT_REPORTED),
            methodology=_get(d, "methodology", NOT_REPORTED),
            algorithm=_get(d, "algorithm", NOT_REPORTED),
            dataset=_get(d, "dataset", NOT_REPORTED),
            results=_get(d, "results", NOT_REPORTED),
            limitations=_get(d, "limitations", NOT_REPORTED),
            novelty=_get(d, "novelty", NOT_REPORTED),
            source_of_analysis=_get(d, "source_of_analysis", "heuristic"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalyzedPaper:
    paper: Paper
    analysis: PaperAnalysis

    @classmethod
    def from_dict(cls, d: dict) -> "AnalyzedPaper":
        return cls(
            paper=Paper.from_dict(d["paper"]),
            analysis=PaperAnalysis.from_dict(d["analysis"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper": self.paper.to_dict(),
            "analysis": self.analysis.to_dict(),
        }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
@dataclass
class ComparisonRow:
    paper_id: str
    title: str
    problem: str = NOT_REPORTED
    methodology: str = NOT_REPORTED
    algorithm: str = NOT_REPORTED
    dataset: str = NOT_REPORTED
    results: str = NOT_REPORTED
    limitations: str = NOT_REPORTED
    novelty: str = NOT_REPORTED

    @classmethod
    def from_dict(cls, d: dict) -> "ComparisonRow":
        return cls(
            paper_id=_get(d, "paper_id", ""),
            title=_get(d, "title", ""),
            problem=_get(d, "problem", NOT_REPORTED),
            methodology=_get(d, "methodology", NOT_REPORTED),
            algorithm=_get(d, "algorithm", NOT_REPORTED),
            dataset=_get(d, "dataset", NOT_REPORTED),
            results=_get(d, "results", NOT_REPORTED),
            limitations=_get(d, "limitations", NOT_REPORTED),
            novelty=_get(d, "novelty", NOT_REPORTED),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonResult:
    rows: List[ComparisonRow]
    similarities: List[str] = field(default_factory=list)
    differences: List[str] = field(default_factory=list)
    research_gaps: List[str] = field(default_factory=list)
    source_of_analysis: str = "heuristic"

    @classmethod
    def from_dict(cls, d: dict) -> "ComparisonResult":
        return cls(
            rows=[ComparisonRow.from_dict(r) for r in _get(d, "rows", [])],
            similarities=_get(d, "similarities", []) or [],
            differences=_get(d, "differences", []) or [],
            research_gaps=_get(d, "research_gaps", []) or [],
            source_of_analysis=_get(d, "source_of_analysis", "heuristic"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "similarities": list(self.similarities),
            "differences": list(self.differences),
            "research_gaps": list(self.research_gaps),
            "source_of_analysis": self.source_of_analysis,
        }


# ---------------------------------------------------------------------------
# Final pipeline output
# ---------------------------------------------------------------------------
@dataclass
class SearchResponse:
    search_id: str
    topic: str
    total_candidates: int
    papers: List[AnalyzedPaper]
    comparison: ComparisonResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "search_id": self.search_id,
            "topic": self.topic,
            "total_candidates": self.total_candidates,
            "papers": [p.to_dict() for p in self.papers],
            "comparison": self.comparison.to_dict(),
        }