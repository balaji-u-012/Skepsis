from __future__ import annotations

import io
import logging
import re
from typing import Optional

import httpx

from backend.models.schemas import Paper
from backend.utils.config import settings

logger = logging.getLogger("skepsis.fulltext")

_MAX_CHARS = 10_000  # enough for most papers, safe for LLM context


def _truncate(text: str, limit: int = _MAX_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


async def _fetch_pdf_text(url: str, client: httpx.AsyncClient) -> Optional[str]:
    """Download a PDF and extract its text using pypdf."""
    try:
        resp = await client.get(url, follow_redirects=True, timeout=30.0)
        if resp.status_code != 200:
            return None
        # Guard against non-PDF responses
        if not resp.content.startswith(b"%PDF"):
            return None

        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(resp.content))
        pages = []
        for page in reader.pages[:40]:  # first 40 pages is enough
            pages.append(page.extract_text() or "")
        text = "\n".join(pages)
        return _truncate(text) if text.strip() else None
    except Exception as exc:
        logger.warning("PDF fetch failed for %s: %s", url, exc)
        return None


async def _fetch_html_text(url: str, client: httpx.AsyncClient) -> Optional[str]:
    """Fallback: scrape text from an HTML landing page."""
    try:
        resp = await client.get(url, follow_redirects=True, timeout=20.0)
        if resp.status_code != 200:
            return None
        text = re.sub(r"<script.*?</script>", " ", resp.text, flags=re.S | re.I)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return _truncate(text) if len(text) > 500 else None
    except Exception as exc:
        logger.warning("HTML fetch failed for %s: %s", url, exc)
        return None


async def fetch_fulltext(paper: Paper) -> Optional[str]:
    """Try to get full text. Order: arXiv PDF -> Unpaywall (DOI) -> landing page."""
    headers = {"User-Agent": settings.user_agent}

    async with httpx.AsyncClient(headers=headers) as client:
        # 1. arXiv (guaranteed free)
        if paper.id.startswith("arxiv:"):
            arxiv_id = paper.id.split(":", 1)[1]
            # strip version suffix (v1, v2, ...)
            if len(arxiv_id) > 2 and arxiv_id[-2] == "v" and arxiv_id[-1].isdigit():
                arxiv_id = arxiv_id[:-2]
            text = await _fetch_pdf_text(f"https://arxiv.org/pdf/{arxiv_id}.pdf", client)
            if text:
                return text

        # 2. Unpaywall for DOIs
        if paper.doi:
            try:
                resp = await client.get(
                    f"https://api.unpaywall.org/v2/{paper.doi}",
                    params={"email": settings.contact_email or "skepsis@example.com"},
                    timeout=15.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    best = data.get("best_oa_location") or {}
                    url = best.get("url_for_pdf") or best.get("url")
                    if url:
                        text = await _fetch_pdf_text(url, client)
                        if text:
                            return text
            except Exception:
                pass

        # 3. Landing page fallback
        if paper.url:
            return await _fetch_html_text(paper.url, client)

    return None