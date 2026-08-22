"""
Bulk-imports the Retraction Watch Database CSV into retraction_status_cache.

Run via: researgent provenance import-retraction-watch --csv path/to/file.csv

Expected columns (Retraction Watch Database schema — verify against the
current export, column names have shifted across versions):
  OriginalPaperDOI, RetractionDOI, RetractionNature, Reason, URLS

`RetractionNature` maps to our status vocabulary:
  "Retraction"              -> "retracted"
  "Correction"               -> "corrected"
  "Expression of concern"    -> "concern"
  anything else / blank      -> skip row
"""
from __future__ import annotations

import csv
from src.provenance.cache import upsert_status

_NATURE_MAP = {
    "retraction": "retracted",
    "correction": "corrected",
    "expression of concern": "concern",
}


def import_csv(path: str) -> dict:
    imported, skipped = 0, 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nature = (row.get("RetractionNature") or "").strip().lower()
            status = _NATURE_MAP.get(nature)
            doi = (row.get("OriginalPaperDOI") or "").strip()
            if not status or not doi:
                skipped += 1
                continue
            upsert_status(
                doi=doi,
                status=status,
                reason=(row.get("Reason") or "").strip(),
                notice_url=(row.get("URLS") or "").strip(),
                source="retraction_watch_bulk",
            )
            imported += 1
    return {"imported": imported, "skipped": skipped}