"""Ingestion pipeline: PDF or Obsidian vault -> chunks -> embeddings -> vector store."""

from src.ingest.pipeline import (
    ingest_all_domains,
    ingest_directory,
    ingest_file,
    ingest_vault,
)

__all__ = [
    "ingest_all_domains",
    "ingest_directory",
    "ingest_file",
    "ingest_vault",
]
