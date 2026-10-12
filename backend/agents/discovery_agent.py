from __future__ import annotations

import asyncio
import logging
from typing import List

from backend.models.schemas import Paper
from backend.services.arxiv_service import search_arxiv
from backend.services.crossref_service import search_crossref
from backend.services.openalex_service import search_openalex
from backend.services.semantic_scholar_service import search_semantic_scholar
from backend.utils.config import settings

logger = logging.getLogger("skepsis.discovery")

# Overall cap for the whole fan-out. Individual services are expected to set
# their own timeouts too, but this guarantees a hung source can't stall the
# entire request.
_DISCOVERY_TIMEOUT_SECONDS = 30


async def discover_papers(query: str) -> List[Paper]:
    """Runs all four source searches concurrently and flattens the results.

    A failure or timeout in any one source is logged and simply yields fewer
    candidates from that source — it never aborts the whole search.
    """
    limit = settings.max_results_per_source

    # Keep the (name, task) pairs together so they can never drift out of
    # alignment if someone adds/removes a source later.
    sources: list[tuple[str, "asyncio.Task[List[Paper]]"]] = [
        ("arXiv", asyncio.create_task(search_arxiv(query, limit))),
        ("Semantic Scholar", asyncio.create_task(search_semantic_scholar(query, limit))),
        ("OpenAlex", asyncio.create_task(search_openalex(query, limit))),
        ("Crossref", asyncio.create_task(search_crossref(query, limit))),
    ]

    names = [name for name, _ in sources]
    tasks = [task for _, task in sources]

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=_DISCOVERY_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Discovery fan-out exceeded %ss; cancelling remaining sources",
            _DISCOVERY_TIMEOUT_SECONDS,
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        # Collect whatever finished in time.
        results = [task.result() if task.done() and not task.cancelled() else None
                   for task in tasks]

    papers: List[Paper] = []
    for name, result in zip(names, results):
        if result is None:
            logger.warning("%s discovery did not finish before timeout", name)
            continue
        if isinstance(result, Exception):
            logger.warning("%s discovery raised: %s", name, result)
            continue
        logger.info("%s discovery returned %d paper(s)", name, len(result))
        papers.extend(result)

    logger.info(
        "Discovery complete: %d total candidate(s) from %d source(s)",
        len(papers),
        sum(1 for r in results if r and not isinstance(r, Exception)),
    )
    return papers