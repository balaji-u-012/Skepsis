from __future__ import annotations

import logging
from typing import List
from urllib.parse import quote  # kept for API parity; may be used by other services

import feedparser
import httpx

from backend.models.schemas import Paper
from backend.utils.config import settings

logger = logging.getLogger("skepsis.arxiv")

ARXIV_API_URL = "https://export.arxiv.org/api/query"


def _clean_doi_href(href: str) -> str | None:
    """Extract a bare DOI from an arXiv DOI link like
    'http://dx.doi.org/10.1234/abc' → '10.1234/abc'.
    Returns None if it can't find a DOI."""
    if not href:
        return None
    for marker in ("dx.doi.org/", "doi.org/"):
        if marker in href:
            return href.split(marker, 1)[-1].strip() or None
    return href.strip() or None


async def search_arxiv(query: str, max_results: int = 20) -> List[Paper]:
    """Query arXiv's public Atom feed API. No API key required."""
    # arXiv requires the field prefix to apply to the whole phrase; wrapping
    # the query in double quotes makes multi-word queries behave correctly.
    safe_query = query.replace('"', '').strip()
    params = {
    "search_query": f'all:"{safe_query}"',
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    papers: List[Paper] = []
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            resp = await client.get(
                ARXIV_API_URL,
                params=params,
                headers={"User-Agent": settings.user_agent},
            )
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("arXiv search failed: %s", exc)
        return papers

    if getattr(feed, "bozo", 0):
        # feedparser never raises; it flags malformed feeds with bozo=1.
        logger.warning("arXiv returned a malformed feed: %s",
                       getattr(feed, "bozo_exception", "unknown"))

    for entry in getattr(feed, "entries", []):
        arxiv_id = entry.get("id", "").rsplit("/", 1)[-1]
        authors = [
            a.get("name", "").strip()
            for a in entry.get("authors", [])
            if a.get("name")
        ]

        year = None
        published = entry.get("published")
        if published and len(published) >= 4:
            try:
                year = int(published[:4])
            except ValueError:
                year = None

        # DOI: prefer the arXiv-native doi element, fall back to the link list.
        doi = entry.get("arxiv_doi")
        if not doi:
            for link in entry.get("links", []):
                if link.get("title") == "doi":
                    doi = _clean_doi_href(link.get("href", ""))
                    if doi:
                        break

        papers.append(Paper(
            id=f"arxiv:{arxiv_id}",
            title=" ".join(entry.get("title", "").split()),
            authors=authors,
            year=year,
            published_date=published,
            source="arXiv",
            doi=doi,
            url=entry.get("link"),
            abstract=" ".join(entry.get("summary", "").split()) or None,
            venue=entry.get("arxiv_journal_ref"),
            citation_count=None,
        ))

    logger.info("arXiv discovery returned %d paper(s)", len(papers))
    return papers