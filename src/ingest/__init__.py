"""Ingestion pipeline: PDF -> page text -> chunks -> embeddings -> vector store."""

from src.ingest.pipeline import ingest_directory, ingest_file

__all__ = ["ingest_directory", "ingest_file"]
