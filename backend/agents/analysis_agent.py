from __future__ import annotations

import json
import logging
import re
from typing import Optional

from backend.models.schemas import NOT_REPORTED, Paper, PaperAnalysis
from backend.services.fulltext_service import fetch_fulltext
from backend.services.llm_service import call_llm

logger = logging.getLogger("skepsis.analysis")


_LLM_SYSTEM_PROMPT = """You are analyzing a research paper's full text. Extract
the following 6 fields in JSON. Use ONLY what is stated or clearly implied in
the paper. Write "Not reported" only if the field is genuinely impossible to
determine.

Fields:
1. "abstract"     — 2-4 sentence plain-English summary of the paper.
2. "problem"      — the specific research problem the paper addresses.
3. "methodology"  — how the authors approached the problem.
4. "results"      — key quantitative or qualitative outcomes.
5. "limitations"  — limitations of the work. Look for: what the authors
                    acknowledge as constraints, what scenarios they did NOT
                    test, what assumptions they made, what their scope was
                    limited to, or what remains unresolved. If the authors
                    don't state limitations explicitly, infer reasonable ones
                    from what the paper does not cover. Avoid generic filler
                    like "further research is needed" — be specific.
6. "novelty"      — what is genuinely NEW compared to prior work.

Rules:
- Every field must be 1-3 sentences.
- Do not invent facts not present in the text.
- Respond with ONLY a JSON object, no prose, no markdown fences.

Example:
{"abstract": "...", "problem": "...", "methodology": "...",
 "results": "...", "limitations": "...", "novelty": "..."}"""
def _extract_json_block(raw: str) -> Optional[str]:
    if not raw:
        return None
    cleaned = raw.strip()
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(0)
    return cleaned or None


def _clean(value: Optional[str]) -> str:
    if not value or not isinstance(value, str):
        return NOT_REPORTED
    v = value.strip()
    return v if v else NOT_REPORTED


async def _llm_analysis(paper: Paper) -> Optional[PaperAnalysis]:
    # Fetch full text; fall back to abstract
    fulltext = await fetch_fulltext(paper)
    source_text = fulltext or paper.abstract

    if not source_text:
        return None

    user_prompt = (
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(paper.authors) if paper.authors else NOT_REPORTED}\n"
        f"Year: {paper.year or NOT_REPORTED}\n"
        f"Venue: {paper.venue or NOT_REPORTED}\n\n"
        f"Paper text:\n{source_text}\n\n"
        "Extract the 8 fields as JSON."
    )

    raw = await call_llm(_LLM_SYSTEM_PROMPT, user_prompt, max_tokens=2048)
    if not raw:
        return None

    cleaned = _extract_json_block(raw)
    if not cleaned:
        return None

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Could not parse LLM analysis JSON for %s: %s", paper.id, exc)
        return None

    if not isinstance(data, dict):
        return None

    return PaperAnalysis(
        paper_id=paper.id,
        abstract=_clean(data.get("abstract")),
        problem=_clean(data.get("problem")),
        methodology=_clean(data.get("methodology")),
        results=_clean(data.get("results")),
        limitations=_clean(data.get("limitations")),
        novelty=_clean(data.get("novelty")),
        source_of_analysis="llm",
    )


def _heuristic_analysis(paper: Paper) -> PaperAnalysis:
    abstract = _clean(paper.abstract)
    return PaperAnalysis(
        paper_id=paper.id,
        abstract=abstract,
        source_of_analysis="heuristic",
    )


async def analyze_paper(paper: Paper) -> PaperAnalysis:
    llm_result = await _llm_analysis(paper)
    if llm_result:
        return llm_result
    return _heuristic_analysis(paper)