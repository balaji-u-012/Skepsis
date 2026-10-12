from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from backend.agents.analysis_agent import analyze_paper
from backend.agents.comparison_agent import compare_papers
from backend.agents.discovery_agent import discover_papers
from backend.agents.gap_agent import identify_research_gaps
from backend.agents.query_agent import normalize_query
from backend.agents.ranking_agent import filter_by_year, rank_papers
from backend.agents.verification_agent import deduplicate_papers
from backend.models.schemas import AnalyzedPaper, PaperAnalysis, SearchResponse
from backend.services import db_service
from backend.utils.config import settings
from backend.utils.http_server import ApiError, Router, run_server

logger = logging.getLogger("skepsis.main")

router = Router()

# Cap on the number of papers analyzed per search. Each analyzed paper can
# trigger an LLM call, so a high cap directly increases request latency.
_MAX_PAPERS_PER_SEARCH = 10
# Overall wall-clock budget for one search request.
_PIPELINE_TIMEOUT_SECONDS = 120.0
# Sanity bounds for user-provided years.
_MIN_YEAR = 1900
_MAX_YEAR = 2100


def _parse_year(value: Any, field_name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        year = int(value)
    except (TypeError, ValueError):
        raise ApiError(400, f"\"{field_name}\" must be an integer year.")
    if year < _MIN_YEAR or year > _MAX_YEAR:
        raise ApiError(400, f"\"{field_name}\" must be between {_MIN_YEAR} and {_MAX_YEAR}.")
    return year


async def _run_pipeline(
    topic: str,
    year_from: Optional[int],
    year_to: Optional[int],
    max_results: int,
) -> SearchResponse:
    query = normalize_query(topic)
    if not query:
        raise ApiError(400, "Topic must not be empty.")

    logger.info("Search: topic=%r years=%s..%s max=%d",
                topic, year_from, year_to, max_results)

    # 1. Discovery — fan out to arXiv, Semantic Scholar, OpenAlex, Crossref
    candidates = await discover_papers(query)
    if not candidates:
        raise ApiError(
            404,
            "No results from any research source. The sources may be unreachable "
            "from this network, or the topic may need rephrasing.",
        )

    # 2. Verification / deduplication
    deduped = deduplicate_papers(candidates)

    # 3. Year filter, then relevance + recency ranking -> top N
    if year_from or year_to:
        pool = filter_by_year(deduped, year_from, year_to)
        if not pool:
            raise ApiError(
                404,
                "No papers matched this topic and year range. "
                "Try widening the date range.",
            )
    else:
        pool = deduped

    top_papers = rank_papers(query, pool, top_n=max_results)
    if not top_papers:
        raise ApiError(404, "No papers matched this topic.")

    # 4. Paper Analysis Agent — structured extraction per paper (parallel).
    # We use return_exceptions=True so one bad paper can't kill the whole
    # request; failed analyses fall back to a stub.
    raw_analyses = await asyncio.gather(
        *(analyze_paper(p) for p in top_papers),
        return_exceptions=True,
    )
    analyzed_papers: List[AnalyzedPaper] = []
    for p, a in zip(top_papers, raw_analyses):
        if isinstance(a, Exception):
            logger.warning("Analysis failed for paper %s: %s", p.id, a)
            a = PaperAnalysis(paper_id=p.id, source_of_analysis="error")
        analyzed_papers.append(AnalyzedPaper(paper=p, analysis=a))

    # 5. Comparison Agent
    comparison = await compare_papers(analyzed_papers)

    # 6. Research Gap Agent
    comparison.research_gaps = await identify_research_gaps(analyzed_papers)

    search_id = uuid.uuid4().hex[:12]
    # SQLite writes are blocking; run them on a worker thread so we don't
    # stall the shared asyncio loop.
    await asyncio.to_thread(
        db_service.save_search,
        search_id, topic, year_from, year_to, len(deduped),
        analyzed_papers, comparison,
    )

    return SearchResponse(
        search_id=search_id,
        topic=topic,
        total_candidates=len(deduped),
        papers=analyzed_papers,
        comparison=comparison,
    )


@router.post("/api/search")
async def search(path_params: Dict[str, str], query_params: Dict[str, str],
                  body: Optional[dict]) -> Any:
    if not body or not str(body.get("topic", "")).strip():
        raise ApiError(400, "\"topic\" is required in the request body.")

    topic = str(body["topic"]).strip()
    year_from = _parse_year(body.get("year_from"), "year_from")
    year_to = _parse_year(body.get("year_to"), "year_to")
    if year_from and year_to and year_from > year_to:
        raise ApiError(400, "\"year_from\" must not be greater than \"year_to\".")

    raw_max = body.get("max_results") or settings.top_n_papers
    try:
        max_results = max(1, min(int(raw_max), _MAX_PAPERS_PER_SEARCH))
    except (TypeError, ValueError):
        max_results = settings.top_n_papers

    try:
        response = await asyncio.wait_for(
            _run_pipeline(topic, year_from, year_to, max_results),
            timeout=_PIPELINE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise ApiError(
            504,
            f"Search timed out after {int(_PIPELINE_TIMEOUT_SECONDS)}s. "
            "Try a narrower topic or fewer results.",
        )
    return response


@router.get("/api/search/{search_id}")
async def get_search(path_params: Dict[str, str], query_params: Dict[str, str],
                      body: Optional[dict]) -> Any:
    record = await asyncio.to_thread(db_service.get_search, path_params["search_id"])
    if not record:
        raise ApiError(404, "Search not found.")
    return record


@router.get("/api/papers/{paper_id}")
async def get_paper(path_params: Dict[str, str], query_params: Dict[str, str],
                     body: Optional[dict]) -> Any:
    record = await asyncio.to_thread(db_service.get_paper, path_params["paper_id"])
    if not record:
        raise ApiError(404, "Paper not found.")
    return record


@router.post("/api/compare")
async def compare(path_params: Dict[str, str], query_params: Dict[str, str],
                   body: Optional[dict]) -> Any:
    raw_ids = (body or {}).get("paper_ids") or []
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ApiError(400, "\"paper_ids\" must be a non-empty list.")

    # Coerce to strings and cap length to avoid runaway work.
    paper_ids = [str(pid) for pid in raw_ids[:20] if isinstance(pid, (str, int))]

    analyzed_papers: List[AnalyzedPaper] = []
    for pid in paper_ids:
        record = await asyncio.to_thread(db_service.get_paper, pid)
        if not record:
            raise ApiError(404, f"Paper not found: {pid}")
        analyzed_papers.append(record)

    comparison = await compare_papers(analyzed_papers)
    comparison.research_gaps = await identify_research_gaps(analyzed_papers)
    return comparison


@router.get("/api/health")
async def health(path_params: Dict[str, str], query_params: Dict[str, str],
                  body: Optional[dict]) -> Any:
    return {
        "status": "ok",
        "llm_enabled": settings.has_llm,
        "sources": ["arXiv", "Semantic Scholar", "OpenAlex", "Crossref"],
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    db_service.init_db()
    logger.info(
        "Skepsis backend ready. LLM enrichment %s.",
        "enabled" if settings.has_llm else "disabled (heuristic mode only)",
    )
    run_server(router, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()