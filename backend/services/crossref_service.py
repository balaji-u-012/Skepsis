from __future__ import annotations

import html
import logging
import re
from typing import List, Optional

import httpx

from backend.models.schemas import Paper
from backend.utils.config import settings

logger = logging.getLogger("skepsis.crossref")

CROSSREF_API_URL = "https://api.crossref.org/works"

_JATS_TAG_RE = re.compile(r"<[^>]+>")


def _format_authors(author_list: Optional[list]) -> List[str]:
    if not author_list:
        return []
    names = []
    for a in author_list:
        given = a.get("given", "")
        family = a.get("family", "")
        full = " ".join(p for p in (given, family) if p).strip()
        if full:
            names.append(full)
    return names


def _clean_abstract(raw: Optional[str]) -> Optional[str]:
    """Crossref abstracts come as JATS XML fragments. Strip tags and
    decode HTML entities so downstream consumers get plain prose."""
    if not raw:
        return None
    text = _JATS_TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    text = " ".join(text.split())
    return text or None


def _extract_year(item: dict) -> tuple[Optional[int], Optional[str]]:
    """Crossref exposes several date fields: `issued` (the canonical
    publication date), `published-print`, `published-online`, `created`.
    Prefer `issued`, fall back through the others."""
    for key in ("issued", "published-print", "published-online", "created"):
        date_parts = (item.get(key) or {}).get("date-parts") or []
        if date_parts and date_parts[0]:
            parts = date_parts[0]
            year = parts[0] if parts and isinstance(parts[0], int) else None
            published_date = "-".join(str(p) for p in parts if p is not None) or None
            if year:
                return year, published_date
    return None, None


async def search_crossref(query: str, max_results: int = 20) -> List[Paper]:
    """Query the Crossref Works API. No key required, but a `mailto`
    (via settings.contact_email) is strongly recommended — Crossref
    deprioritizes requests without one."""
    params = {
        "query": query,
        "rows": max_results,
        "sort": "published",
        "order": "desc",
        "filter": "type:journal-article",
        # `issued` is the canonical publication-date field in Crossref.
        # `published` is not a valid select key and is silently ignored.
        "select": "DOI,title,author,issued,container-title,URL,abstract,is-referenced-by-count",
    }

    if settings.contact_email:
        params["mailto"] = settings.contact_email
    else:
        logger.warning(
            "Crossref `contact_email` is not set; requests will be slow and "
            "rate-limited. Set CONTACT_EMAIL in your .env to join the polite pool."
        )

    papers: List[Paper] = []
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            resp = await client.get(
                CROSSREF_API_URL,
                params=params,
                headers={"User-Agent": settings.user_agent},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Crossref search failed: %s", exc)
        return papers

    items = data.get("message", {}).get("items", []) or []
    for item in items:
        titles = item.get("title") or []
        title = titles[0] if titles else ""
        if not title:
            continue

        year, published_date = _extract_year(item)

        venues = item.get("container-title") or []
        abstract = _clean_abstract(item.get("abstract"))

        papers.append(Paper(
            id=f"crossref:{item.get('DOI') or title}",
            title=" ".join(title.split()),
            authors=_format_authors(item.get("author")),
            year=year,
            published_date=published_date,
            source="Crossref",
            doi=item.get("DOI"),
            url=item.get("URL"),
            abstract=abstract,
            venue=venues[0] if venues else None,
            citation_count=item.get("is-referenced-by-count"),
        ))

    logger.info("Crossref discovery returned %d paper(s)", len(papers))
    return papers