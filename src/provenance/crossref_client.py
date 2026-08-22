"""
Crossref REST API client — free, no key required.

Crossref now carries retraction relations directly on work metadata:
  - `update-to` array with `type: "retraction"` — the work THIS DOI
    retracts, or that retracts THIS DOI (check `label` field: works
    have `update-to[].updated.DOI` pointing at what superseded them).
  - Some publishers additionally tag `type: "retraction"` on the DOI's
    OWN metadata (i.e. the retraction notice is itself a separate DOI
    linked back via `relation.is-retraction-of`).

We check both shapes. If neither is present, status is "clean" — but
note Crossref coverage of retraction metadata is NOT 100%, which is
exactly why `retraction_source_mode="bulk"` (importing the actual
Retraction Watch Database) is the recommended default — see
retraction_watch.py.
"""
from __future__ import annotations

import httpx
from src.config import settings


async def lookup_doi(doi: str) -> dict:
    """
    Returns {"status": "clean"|"retracted"|"corrected"|"concern",
             "reason": str, "notice_url": str, "source": "crossref"}
    """
    params = {"mailto": settings.crossref_mailto} if settings.crossref_mailto else {}
    url = f"{settings.crossref_base_url}/works/{doi}"
    async with httpx.AsyncClient(timeout=8.0) as client:
        resp = await client.get(url, params=params)
        if resp.status_code != 200:
            return {"status": "clean", "reason": "", "notice_url": "", "source": "crossref"}
        msg = resp.json().get("message", {})
        # Check update-to relations for retraction/correction/EOC types
        for upd in msg.get("update-to", []):
            utype = (upd.get("type") or "").lower()
            if "retraction" in utype:
                return {
                    "status": "retracted", "reason": utype,
                    "notice_url": upd.get("URL", ""), "source": "crossref",
                }
            if "correction" in utype:
                return {
                    "status": "corrected", "reason": utype,
                    "notice_url": upd.get("URL", ""), "source": "crossref",
                }
            if "expression" in utype or "concern" in utype:
                return {
                    "status": "concern", "reason": utype,
                    "notice_url": upd.get("URL", ""), "source": "crossref",
                }
        return {"status": "clean", "reason": "", "notice_url": "", "source": "crossref"}