from __future__ import annotations

import json
import logging
import re
from typing import List

from backend.models.schemas import NOT_REPORTED, AnalyzedPaper
from backend.services.llm_service import call_llm

logger = logging.getLogger("skepsis.gap")

_MAX_EXCERPT_CHARS = 200

_LLM_SYSTEM_PROMPT = """You identify potential research gaps by looking ONLY
at the structured fields of the given papers. Never invent a gap that isn't
supported by the text. If the provided text is too sparse to responsibly
identify gaps, say that plainly instead of guessing.

Respond with ONLY a JSON array of short strings (no prose, no markdown
fences), 2-5 items."""


def _truncate(text: str, limit: int = _MAX_EXCERPT_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _heuristic_gaps(analyzed_papers: List[AnalyzedPaper]) -> List[str]:
    gaps: List[str] = []

    limitation_texts = [
        (ap.paper.title, ap.analysis.limitations)
        for ap in analyzed_papers
        if ap.analysis.limitations != NOT_REPORTED
    ]
    novelty_texts = [
        (ap.paper.title, ap.analysis.novelty)
        for ap in analyzed_papers
        if ap.analysis.novelty != NOT_REPORTED
    ]

    if limitation_texts:
        gaps.append(
            "Stated limitations that point to open problems: " +
            "; ".join(f'"{title}": {_truncate(text)}' for title, text in limitation_texts[:3])
        )
    if novelty_texts:
        gaps.append(
            "Novelty claims that suggest directions not yet fully explored: " +
            "; ".join(f'"{title}": {_truncate(text)}' for title, text in novelty_texts[:3])
        )

    if not gaps:
        gaps.append(
            "The retrieved papers do not contain enough explicit limitation or "
            "novelty detail to responsibly identify research gaps."
        )
    return gaps


def _extract_json_block(raw: str) -> str | None:
    if not raw:
        return None
    cleaned = raw.strip()
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    if cleaned[:1] in "[{":
        return cleaned
    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if not match:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    return match.group(0) if match else None


def _coerce_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


async def _llm_gaps(analyzed_papers: List[AnalyzedPaper]) -> List[str] | None:
    payload = []
    for ap in analyzed_papers:
        a = ap.analysis
        entry = {"title": ap.paper.title}
        for field in ("problem", "methodology", "algorithm", "dataset",
                      "results", "limitations", "novelty"):
            value = getattr(a, field)
            if value and value != NOT_REPORTED:
                entry[field] = value
        payload.append(entry)

    user_prompt = "Papers:\n" + json.dumps(payload, indent=2)
    raw = await call_llm(_LLM_SYSTEM_PROMPT, user_prompt, max_tokens=1024)
    if not raw:
        return None

    cleaned = _extract_json_block(raw)
    if not cleaned:
        return None

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Could not parse LLM gap JSON: %s", exc)
        return None

    if isinstance(data, dict):
        data = data.get("gaps") or data.get("research_gaps") or []

    gaps = _coerce_string_list(data)
    return gaps or None


async def identify_research_gaps(analyzed_papers: List[AnalyzedPaper]) -> List[str]:
    if not analyzed_papers:
        return []
    llm_result = await _llm_gaps(analyzed_papers)
    if llm_result:
        return llm_result
    return _heuristic_gaps(analyzed_papers)