from __future__ import annotations

from dataclasses import replace
from typing import List

from backend.models.schemas import Paper
from backend.utils.text_utils import author_overlap, title_similarity

TITLE_SIMILARITY_THRESHOLD = 0.90
TITLE_SIMILARITY_WITH_AUTHOR_OVERLAP_THRESHOLD = 0.75

# Preference order used when two records refer to the same paper and we must
# keep one as the "primary" record (richer abstracts / DOIs tend to win).
# NOTE: keys are case-sensitive and must match what each service sets on
# Paper.source exactly.
SOURCE_PRIORITY = {
    "Semantic Scholar": 0,
    "arXiv": 1,
    "OpenAlex": 2,
    "Crossref": 3,
}


def _normalize_doi(doi: str | None) -> str | None:
    """Normalize a DOI so different representations compare equal:
       'https://doi.org/10.1/x'  →  '10.1/x'
       'DOI:10.1/x'              →  '10.1/x'
       '10.1/X'                  →  '10.1/x'
    """
    if not doi:
        return None
    d = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:", "https://dx.doi.org/"):
        if d.startswith(prefix):
            d = d[len(prefix):]
            break
    return d or None


def _is_duplicate(a: Paper, b: Paper) -> bool:
    doi_a = _normalize_doi(a.doi)
    doi_b = _normalize_doi(b.doi)
    if doi_a and doi_b and doi_a == doi_b:
        return True

    sim = title_similarity(a.title, b.title)
    if sim >= TITLE_SIMILARITY_THRESHOLD:
        return True
    if (
        sim >= TITLE_SIMILARITY_WITH_AUTHOR_OVERLAP_THRESHOLD
        and author_overlap(a.authors, b.authors) >= 0.5
    ):
        return True
    return False


def _merge_pair(primary: Paper, secondary: Paper) -> Paper:
    """Fill any missing fields on `primary` using `secondary`, never
    inventing data. Builds the merged object in a single replace() call so
    it works whether Paper is frozen or not."""
    return replace(
        primary,
        abstract=primary.abstract or secondary.abstract,
        doi=primary.doi or secondary.doi,
        year=primary.year or secondary.year,
        venue=primary.venue or secondary.venue,
        authors=primary.authors or secondary.authors,
        citation_count=(
            primary.citation_count
            if primary.citation_count is not None
            else secondary.citation_count
        ),
    )


def deduplicate_papers(papers: List[Paper]) -> List[Paper]:
    """Clusters near-duplicate records (same paper indexed by multiple
    sources) into a single merged Paper, keeping the record from the
    highest-priority source as primary and back-filling gaps from others.

    Uses full single-linkage clustering: a new paper is checked against
    EVERY member of an existing cluster, not just the cluster head, so
    chains like A~B~C correctly collapse even when A and C are only
    indirectly related.
    """
    clusters: List[List[Paper]] = []

    for paper in papers:
        placed = False
        for cluster in clusters:
            if any(_is_duplicate(member, paper) for member in cluster):
                cluster.append(paper)
                placed = True
                break
        if not placed:
            clusters.append([paper])

    merged_papers: List[Paper] = []
    for cluster in clusters:
        cluster_sorted = sorted(
            cluster, key=lambda p: SOURCE_PRIORITY.get(p.source, 99)
        )
        primary = cluster_sorted[0]
        for other in cluster_sorted[1:]:
            primary = _merge_pair(primary, other)
        merged_papers.append(primary)

    return merged_papers