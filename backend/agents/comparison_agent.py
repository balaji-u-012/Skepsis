from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from typing import List

from backend.models.schemas import NOT_REPORTED, AnalyzedPaper, ComparisonResult, ComparisonRow
from backend.services.llm_service import call_llm

logger = logging.getLogger("skepsis.comparison")

# Words that are too common to signal a meaningful overlap between papers.
_STOPWORDS = {
    "that", "this", "these", "those", "with", "from", "into", "their", "there",
    "which", "while", "where", "when", "have", "been", "were", "will", "would",
    "could", "should", "than", "then", "them", "they", "also", "such", "more",
    "most", "some", "each", "other", "only", "over", "under", "about", "using",
    "used", "uses", "based", "paper", "study", "work", "results", "approach",
    "method", "methods", "data", "propose", "proposed", "present", "presented",
}


def _row_from_analysis(ap: AnalyzedPaper) -> ComparisonRow:
    a = ap.analysis
    return ComparisonRow(
        paper_id=ap.paper.id,
        title=ap.paper.title,
        problem=a.problem,
        methodology=a.methodology,
        results=a.results,
        limitations=a.limitations,
        novelty=a.novelty,
    )


def _heuristic_similarities_differences(rows: List[ComparisonRow]) -> tuple[list[str], list[str]]:
    similarities: List[str] = []
    differences: List[str] = []

    def _reported_values(field: str) -> dict:
        return {r.paper_id: getattr(r, field) for r in rows if getattr(r, field) != NOT_REPORTED}

    def _meaningful_tokens(text: str) -> set[str]:
        tokens = set(re.findall(r"[a-zA-Z]{4,}", text.lower()))
        return tokens - _STOPWORDS

    # Compare algorithm and dataset overlap across papers
    for field, label in (("algorithm", "algorithm/model"), ("dataset", "dataset")):
        values = _reported_values(field)
        if len(values) < 2:
            continue
        tokenized = {pid: _meaningful_tokens(v) for pid, v in values.items()}
        pids = list(tokenized)
        shared_any = False
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                if len(tokenized[pids[i]] & tokenized[pids[j]]) >= 2:
                    shared_any = True
                    break
            if shared_any:
                break
        if shared_any:
            similarities.append(f"Several papers report overlapping terms in their {label}.")
        else:
            differences.append(f"Papers report distinct approaches to {label}, with no shared terms detected.")

    # Note any missing algorithm coverage
    reported_algorithms = _reported_values("algorithm")
    if len(reported_algorithms) < len(rows):
        differences.append(
            "One or more papers do not report enough detail to identify the "
            "algorithm/model used (marked Not reported)."
        )

    return similarities, differences


_LLM_SYSTEM_PROMPT = """You compare a set of research papers strictly using
the structured fields provided to you. Never invent facts not present in the
input. If there isn't enough information to state a similarity or
difference, say so plainly rather than guessing.

Focus your comparison on these fields: problem, methodology, algorithm,
dataset, results, limitations, and novelty.

Respond with ONLY a JSON object (no prose, no markdown fences) with two
keys: "similarities" (array of short strings) and "differences" (array of
short strings). 3-5 items each is ideal."""


def _extract_json_block(raw: str) -> str | None:
    """Robustly pull the first JSON object out of an LLM response."""
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


def _coerce_string_list(value) -> list[str]:
    """Turn whatever the LLM returned into a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


async def _llm_similarities_differences(rows: List[ComparisonRow]) -> tuple[list[str], list[str]] | None:
    payload = [asdict(row) for row in rows]
    user_prompt = "Papers:\n" + json.dumps(payload, indent=2)
    raw = await call_llm(_LLM_SYSTEM_PROMPT, user_prompt, max_tokens=2048)
    if not raw:
        return None

    cleaned = _extract_json_block(raw)
    if not cleaned:
        logger.warning("LLM returned no JSON object for comparison")
        return None

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Could not parse LLM comparison JSON: %s", exc)
        return None

    if not isinstance(data, dict):
        logger.warning("LLM comparison response was not a JSON object")
        return None

    return (
        _coerce_string_list(data.get("similarities")),
        _coerce_string_list(data.get("differences")),
    )


async def compare_papers(analyzed_papers: List[AnalyzedPaper]) -> ComparisonResult:
    rows = [_row_from_analysis(ap) for ap in analyzed_papers]

    llm_result = await _llm_similarities_differences(rows) if analyzed_papers else None
    if llm_result:
        similarities, differences = llm_result
        source = "llm"
    else:
        similarities, differences = _heuristic_similarities_differences(rows)
        source = "heuristic"

    return ComparisonResult(
        rows=rows,
        similarities=similarities,
        differences=differences,
        research_gaps=[],  # filled in by the gap agent
        source_of_analysis=source,
    )