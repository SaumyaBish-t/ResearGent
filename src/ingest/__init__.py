"""Ingestion pipeline: PDF or Obsidian vault -> chunks -> embeddings -> vector store."""

from src.ingest.pipeline import ingest_directory, ingest_file, ingest_vault

__all__ = ["ingest_directory", "ingest_file", "ingest_vault"]
