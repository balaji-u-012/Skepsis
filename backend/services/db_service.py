from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, List, Optional

from backend.models.schemas import AnalyzedPaper, ComparisonResult, Paper, PaperAnalysis
from backend.utils.config import settings

# Composite primary key on (search_id, paper_id) so the same paper can
# legitimately appear in multiple searches without one overwriting the other.
SCHEMA = """
CREATE TABLE IF NOT EXISTS searches (
    search_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    year_from INTEGER,
    year_to INTEGER,
    total_candidates INTEGER,
    comparison_json TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS papers (
    search_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    paper_json TEXT NOT NULL,
    analysis_json TEXT NOT NULL,
    PRIMARY KEY (search_id, paper_id),
    FOREIGN KEY (search_id) REFERENCES searches (search_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_papers_search_id ON papers(search_id);
CREATE INDEX IF NOT EXISTS idx_searches_created_at ON searches(created_at DESC);
"""


def _ensure_db_dir() -> None:
    """Make sure the parent directory for the SQLite file exists, otherwise
    sqlite3.connect will silently create an unusable DB (or fail later)."""
    db_path = Path(settings.resolved_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    _ensure_db_dir()
    conn = sqlite3.connect(settings.resolved_db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    # WAL mode lets readers continue while a writer is active, avoiding
    # the classic "database is locked" error under concurrent requests.
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
    except sqlite3.DatabaseError:
        # Some environments (e.g. read-only FS) reject these pragmas; the
        # connection still works without them.
        pass
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def save_search(
    search_id: str,
    topic: str,
    year_from: Optional[int],
    year_to: Optional[int],
    total_candidates: int,
    analyzed_papers: List[AnalyzedPaper],
    comparison: ComparisonResult,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO searches
               (search_id, topic, year_from, year_to, total_candidates, comparison_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                search_id,
                topic,
                year_from,
                year_to,
                total_candidates,
                json.dumps(asdict(comparison)),
            ),
        )
        for ap in analyzed_papers:
            conn.execute(
                """INSERT OR REPLACE INTO papers
                   (search_id, paper_id, paper_json, analysis_json)
                   VALUES (?, ?, ?, ?)""",
                (
                    search_id,
                    ap.paper.id,
                    json.dumps(asdict(ap.paper)),
                    json.dumps(asdict(ap.analysis)),
                ),
            )


def get_paper(paper_id: str, search_id: Optional[str] = None) -> Optional[AnalyzedPaper]:
    """Fetch a single analyzed paper. If `search_id` is given, restrict to
    that search; otherwise return the most recently stored record for the
    paper across all searches."""
    query = "SELECT paper_json, analysis_json FROM papers WHERE paper_id = ?"
    params: tuple = (paper_id,)
    if search_id is not None:
        query += " AND search_id = ?"
        params = (paper_id, search_id)
    query += " ORDER BY rowid DESC LIMIT 1"

    with get_connection() as conn:
        row = conn.execute(query, params).fetchone()
    if not row:
        return None
    return AnalyzedPaper(
        paper=Paper.from_dict(json.loads(row["paper_json"])),
        analysis=PaperAnalysis.from_dict(json.loads(row["analysis_json"])),
    )


def get_search(search_id: str) -> Optional[dict]:
    with get_connection() as conn:
        search_row = conn.execute(
            "SELECT * FROM searches WHERE search_id = ?", (search_id,)
        ).fetchone()
        if not search_row:
            return None
        paper_rows = conn.execute(
            "SELECT paper_json, analysis_json FROM papers WHERE search_id = ?",
            (search_id,),
        ).fetchall()

    papers = [
        AnalyzedPaper(
            paper=Paper.from_dict(json.loads(r["paper_json"])),
            analysis=PaperAnalysis.from_dict(json.loads(r["analysis_json"])),
        )
        for r in paper_rows
    ]
    return {
        "search_id": search_row["search_id"],
        "topic": search_row["topic"],
        "year_from": search_row["year_from"],
        "year_to": search_row["year_to"],
        "total_candidates": search_row["total_candidates"],
        "created_at": search_row["created_at"],
        "papers": papers,
        "comparison": ComparisonResult.from_dict(json.loads(search_row["comparison_json"])),
    }