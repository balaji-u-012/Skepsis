from __future__ import annotations

import logging
from typing import List, Optional

import httpx

from backend.models.schemas import Paper
from backend.utils.config import settings

logger = logging.getLogger("skepsis.openalex")

OPENALEX_API_URL = "https://api.openalex.org/works"

# Only the fields we actually consume. Requesting this explicitly is the
# ONLY way to get `abstract_inverted_index` back — OpenAlex omits it from
# the default response, which is why every abstract was coming back None.
_SELECT_FIELDS = ",".join([
    "id",
    "title",
    "display_name",
    "authorships",
    "publication_year",
    "publication_date",
    "doi",
    "primary_location",
    "cited_by_count",
    "abstract_inverted_index",
])


def _reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    """OpenAlex returns abstracts as an inverted index {word: [positions]}.

    Positions are integers, so we sort numerically. We coerce to int
    defensively in case a source ever emits string positions.
    """
    if not inverted_index:
        return None
    position_word: dict[int, str] = {}
    for word, positions in inverted_index.items():
        if not positions:
            continue
        for pos in positions:
            try:
                position_word[int(pos)] = word
            except (TypeError, ValueError):
                continue
    if not position_word:
        return None
    ordered = [position_word[i] for i in sorted(position_word)]
    text = " ".join(ordered).strip()
    return text or None


async def search_openalex(
    query: str,
    max_results: int = 20,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> List[Paper]:
    """Query OpenAlex Works API. No key required; a mailto contact is polite.

    If year_from / year_to are provided, the filter is applied at the API
    level so we don't waste the result budget on out-of-range papers.
    """
    params = {
        "search": query,
        "per-page": min(max_results, 200),  # OpenAlex hard cap
        "sort": "publication_date:desc",
        "select": _SELECT_FIELDS,
    }

    # Apply the date range server-side when we can. OpenAlex uses the
    # filter syntax `from_publication_date:YYYY-MM-DD,to_publication_date:...`.
    filters = []
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if year_to:
        filters.append(f"to_publication_date:{year_to}-12-31")
    if filters:
        params["filter"] = ",".join(filters)

    if settings.contact_email:
        params["mailto"] = settings.contact_email

    papers: List[Paper] = []
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            resp = await client.get(
                OPENALEX_API_URL,
                params=params,
                headers={"User-Agent": settings.user_agent},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenAlex search failed: %s", exc)
        return papers

    for item in data.get("results", []) or []:
        raw_id = item.get("id") or ""
        if not raw_id:
            continue

        title = (item.get("title") or item.get("display_name") or "").strip()
        if not title:
            # Empty titles cause false-positive dedup downstream.
            continue

        authorships = item.get("authorships") or []
        authors = [
            (a.get("author") or {}).get("display_name", "")
            for a in authorships
            if (a.get("author") or {}).get("display_name")
        ]

        doi = item.get("doi")
        if doi:
            doi = doi.replace("https://doi.org/", "").strip() or None

        primary_location = item.get("primary_location") or {}
        source_info = primary_location.get("source") or {}

        papers.append(Paper(
            id=f"openalex:{raw_id.rsplit('/', 1)[-1]}",
            title=title,
            authors=authors,
            year=item.get("publication_year"),
            published_date=item.get("publication_date"),
            source="OpenAlex",
            doi=doi,
            url=primary_location.get("landing_page_url") or raw_id,
            abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
            venue=source_info.get("display_name"),
            citation_count=item.get("cited_by_count"),
        ))

    logger.info("OpenAlex discovery returned %d paper(s)", len(papers))
    return papers