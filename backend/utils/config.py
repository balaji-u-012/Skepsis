"""
Central configuration. Reads backend/.env (or repo-root .env) if present.
No hard dependency on any value being set — every optional feature checks
`settings.has_llm` / `settings.contact_email` before using it.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is in requirements, but degrade gracefully
    load_dotenv = None

logger = logging.getLogger("skepsis.config")

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


def _load_env_once() -> None:
    """Load the first .env file found (backend/.env, then repo-root .env)."""
    if not load_dotenv:
        return
    for candidate in (BACKEND_DIR / ".env", PROJECT_ROOT / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            logger.info("Loaded environment from %s", candidate)
            return


_load_env_once()


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a valid float; using default %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not a valid int; using default %s", name, raw, default)
        return default


class Settings:
    def __init__(self) -> None:
        # Google Gemini (free tier)
        self.gemini_api_key: str = _env_str("GEMINI_API_KEY")
        self.gemini_model: str = _env_str("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self.llm_timeout_seconds: float = _env_float("LLM_TIMEOUT_SECONDS", 60.0)

        # External API contact / etiquette
        self.contact_email: str = _env_str("CONTACT_EMAIL")

        # Optional Semantic Scholar API key
        self.semantic_scholar_api_key: str = _env_str("SEMANTIC_SCHOLAR_API_KEY")

        # Storage
        self.db_path: str = _env_str("DB_PATH", "../data/skepsis.db")

        # HTTP behaviour towards external APIs
        self.request_timeout_seconds: float = _env_float("REQUEST_TIMEOUT_SECONDS", 15.0)
        self.max_results_per_source: int = _env_int("MAX_RESULTS_PER_SOURCE", 20)
        self.top_n_papers: int = _env_int("TOP_N_PAPERS", 5)

    @property
    def has_llm(self) -> bool:
        return bool(self.gemini_api_key) and bool(self.gemini_model)

    @property
    def resolved_db_path(self) -> Path:
        path = Path(self.db_path)
        if not path.is_absolute():
            path = (BACKEND_DIR / path).resolve()
        return path

    @property
    def user_agent(self) -> str:
        contact = self.contact_email or "skepsis-app@example.com"
        return f"Skepsis/1.0 (mailto:{contact})"


settings = Settings()