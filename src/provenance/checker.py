"""
Public provenance-checking API. Batches lookups, checks cache first,
falls through to live Crossref for misses/stale entries.
"""
from __future__ import annotations

import asyncio
from src.config import settings
from src.provenance.cache import get_cached, upsert_status
from src.provenance.crossref_client import lookup_doi


async def check_many(identifiers: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    """
    identifiers: list of (doi, arxiv_id) tuples, either may be "".
    Returns {(doi, arxiv_id): {"status": ..., "reason": ..., "notice_url": ...}}
    """
    out: dict[tuple[str, str], dict] = {}
    to_fetch: list[tuple[str, str]] = []

    for doi, arxiv_id in identifiers:
        if not doi and not arxiv_id:
            out[(doi, arxiv_id)] = {"status": "clean", "reason": "", "notice_url": ""}
            continue
        cached = get_cached(doi=doi, arxiv_id=arxiv_id)
        if cached is not None:
            out[(doi, arxiv_id)] = cached
        else:
            to_fetch.append((doi, arxiv_id))

    if to_fetch:
        sem = asyncio.Semaphore(settings.provenance_max_concurrent_lookups)

        async def _one(doi: str, arxiv_id: str):
            if not doi:
                # No DOI to check live against Crossref — arXiv preprints
                # rarely retract via Crossref; mark clean but unverified.
                # (Phase 23's bulk sweep re-checks these against arXiv's
                # own withdrawal notices separately — see 23.3.)
                return (doi, arxiv_id), {"status": "clean", "reason": "", "notice_url": ""}
            async with sem:
                result = await lookup_doi(doi)
            upsert_status(doi=doi, arxiv_id=arxiv_id, source="crossref", **{
                k: v for k, v in result.items() if k != "source"
            })
            return (doi, arxiv_id), result

        results = await asyncio.gather(*[_one(d, a) for d, a in to_fetch])
        for key, val in results:
            out[key] = val

    return out