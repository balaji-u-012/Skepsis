from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from backend.utils.config import settings

logger = logging.getLogger("skepsis.llm")

# Gemini's OpenAI-compatible endpoint. This lets us keep the standard
# OpenAI chat-completions format (system + user messages) while using
# Google's free-tier Gemini models.
GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

# Transient status codes worth retrying.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 2
_BACKOFF_BASE = 1.5

_logged_no_llm = False


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1000,
) -> Optional[str]:
    """Calls Google Gemini via its OpenAI-compatible endpoint.

    Returns None (never raises) if no key is configured or the call fails,
    so every caller must have a non-LLM fallback.
    """
    global _logged_no_llm

    if not settings.has_llm:
        if not _logged_no_llm:
            logger.info(
                "No Gemini API key configured; LLM features disabled "
                "and all analysis will run in heuristic mode."
            )
            _logged_no_llm = True
        return None

    headers = {
        "Authorization": f"Bearer {settings.gemini_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": settings.gemini_model,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    timeout = getattr(settings, "llm_timeout_seconds", 60.0)
    data: Optional[dict] = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(GEMINI_OPENAI_URL, headers=headers, json=body)

            if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    "Gemini returned %d; retrying in %.1fs (attempt %d/%d)",
                    resp.status_code, wait, attempt + 1, _MAX_RETRIES,
                )
                await asyncio.sleep(wait)
                continue

            if resp.status_code >= 400:
                logger.warning("Gemini HTTP %d: %s", resp.status_code, resp.text[:500])
                return None

            data = resp.json()
            break
        except Exception as exc:
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_BACKOFF_BASE ** attempt)
                continue
            logger.warning("LLM call failed: %s", exc)
            return None

    if not isinstance(data, dict):
        return None

    choices = data.get("choices") or []
    if not choices:
        logger.warning("Gemini response had no choices.")
        return None

    choice = choices[0]
    message = choice.get("message") or {}
    text = (message.get("content") or "").strip()
    return text or None