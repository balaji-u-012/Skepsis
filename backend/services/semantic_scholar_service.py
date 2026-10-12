from __future__ import annotations

import asyncio
import logging
from typing import List

import httpx

from backend.models.schemas import Paper
from backend.utils.config import settings

logger = logging.getLogger("skepsis.semantic_scholar")

S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,authors,year,publicationDate,venue,abstract,externalIds,url,citationCount"

# S2 returns 429 for rate limits and 503/504 when the shared pool is
# saturated. All three are transient; a single retry usually gets through.
_RETRYABLE_STATUS = {429, 503, 504}
_MAX_RETRIES = 1
_BACKOFF_SECONDS = 3.0


async def search_semantic_scholar(query: str, max_results: int = 20) -> List[Paper]:
    """Query the Semantic Scholar Graph API.

    Works without a key (lower rate limit). If an API key is configured
    via settings.semantic_scholar_api_key, it is sent as x-api-key for a
    much higher rate limit.
    """
    params = {
        "query": query,
        "limit": min(max_results, 100),  # S2 hard cap is 100
        "fields": FIELDS,
    }

    headers = {"User-Agent": settings.user_agent}
    api_key = getattr(settings, "semantic_scholar_api_key", None)
    if api_key:
        headers["x-api-key"] = api_key

    papers: List[Paper] = []
    data: dict | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
                resp = await client.get(S2_API_URL, params=params, headers=headers)

            if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                logger.warning(
                    "Semantic Scholar returned %d; retrying in %.1fs (attempt %d/%d)",
                    resp.status_code, _BACKOFF_SECONDS, attempt + 1, _MAX_RETRIES,
                )
                await asyncio.sleep(_BACKOFF_SECONDS)
                continue

            if resp.status_code == 429:
                logger.warning("Semantic Scholar rate-limited; giving up on this source.")
                return papers

            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:  # noqa: BLE001
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "Semantic Scholar call failed (%s); retrying in %.1fs",
                    exc, _BACKOFF_SECONDS,
                )
                await asyncio.sleep(_BACKOFF_SECONDS)
                continue
            logger.warning("Semantic Scholar search failed: %s", exc)
            return papers

    if not isinstance(data, dict):
        logger.warning("Semantic Scholar returned a non-dict payload.")
        return papers

    for item in data.get("data", []) or []:
        title = (item.get("title") or "").strip()
        if not title:
            # Empty titles cause false-positive dedup downstream.
            continue

        authors = [
            a.get("name", "")
            for a in (item.get("authors") or [])
            if a.get("name")
        ]

        external_ids = item.get("externalIds") or {}
        doi = external_ids.get("DOI")

        paper_id = (
            item.get("paperId")
            or doi
            or title
        )
        if not paper_id:
            # Extremely unlikely, but avoid colliding "s2:" IDs.
            continue

        papers.append(Paper(
            id=f"s2:{paper_id}",
            title=title,
            authors=authors,
            year=item.get("year"),
            published_date=item.get("publicationDate"),
            source="Semantic Scholar",
            doi=doi,
            url=item.get("url"),
            abstract=(item.get("abstract") or "").strip() or None,
            venue=item.get("venue") or None,
            citation_count=item.get("citationCount"),
        ))

    logger.info("Semantic Scholar discovery returned %d paper(s)", len(papers))
    return papers