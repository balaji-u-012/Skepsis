from __future__ import annotations

import re

# Very small stopword list — we only strip words that add no signal to a
# search query and that some APIs score as noise.
_QUERY_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on",
    "with", "by", "at", "from", "is", "are", "be", "as",
}

# Characters that break at least one discovery API's query parser:
#   " '  → break S2 / general JSON embedding
#   &     → Crossref treats as a param separator
#   ? #   → URL query/fragment confusion
#   :     → arXiv field-prefix operator (ti:, au:, abs:)
#   ( )   → boolean grouping in S2 / arXiv
#   < >   → XSS-ish / XML parsers
_UNSAFE_QUERY_CHARS = re.compile(r'["\'&?#:()<>]')

# Hard cap so a pasted paragraph can't be submitted as a query.
_MAX_QUERY_CHARS = 300

# Short single-word queries are usually too broad; wrapping them in quotes
# asks the discovery APIs for exact-phrase matches instead of keyword soup.
_SHORT_QUERY_CHARS = 6


def normalize_query(raw_topic: str) -> str:
    """Cleans the user's input into a compact query string suitable for the
    discovery APIs. Kept intentionally simple and deterministic (no LLM
    dependency) so the pipeline's first stage never fails for lack of a key.
    """
    if not raw_topic:
        return ""

    cleaned = re.sub(r"\s+", " ", raw_topic).strip()
    cleaned = _UNSAFE_QUERY_CHARS.sub(" ", cleaned)

    # Collapse any whitespace the substitution may have introduced.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Drop pure stopwords only when there's still something left to search on.
    tokens = [t for t in cleaned.split(" ") if t]
    meaningful = [t for t in tokens if t.lower() not in _QUERY_STOPWORDS]
    if meaningful:
        cleaned = " ".join(meaningful)

    # Truncate on a word boundary.
    if len(cleaned) > _MAX_QUERY_CHARS:
        cleaned = cleaned[:_MAX_QUERY_CHARS].rsplit(" ", 1)[0]

    # A single short, unquoted word is usually too broad — quote it so
    # phrase-aware APIs (arXiv, S2) tighten the match.
    if (
        " " not in cleaned
        and len(cleaned) <= _SHORT_QUERY_CHARS
        and not cleaned.startswith('"')
    ):
        cleaned = f'"{cleaned}"'

    return cleaned