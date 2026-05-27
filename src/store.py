"""
Vector store — thin wrapper around a persistent ChromaDB client.

Why a wrapper at all?
  - Single place that knows where the DB lives on disk.
  - Collection naming convention encodes (provider, embed-model) so switching
    providers doesn't silently corrupt a collection with mismatched embedding
    dimensions. A new (provider, model) -> a new collection.
  - One place to add telemetry / migrations / backups later.

Chroma's embedding behavior
---------------------------
We pass `embedding_function=None`. That means Chroma does NOT embed on insert
or query — we provide the vectors ourselves via `src.llm.embed`. This keeps
embedding logic in one place (the provider abstraction) and avoids Chroma's
auto-download of a sentence-transformers default model.
"""

from __future__ import annotations

import re
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from src.config import ModelTier, settings as app_settings

DB_PATH = Path("data") / "chroma_db"
PAPERS_PREFIX = "papers"


def _sanitize(s: str) -> str:
    """Chroma collection names: [a-zA-Z0-9._-], length 3-63."""
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s)
    return s.strip("-._")[:50] or "default"


def collection_name_for_current_embedder() -> str:
    """
    Encode (embed-provider, embed-model) into the collection name.

    Example: papers__nvidia__nvidia-nv-embed-v1
             papers__ollama__nomic-embed-text
    """
    provider = app_settings.resolve_provider(ModelTier.EMBED)
    if provider == "nvidia":
        model = app_settings.nvidia_model_embed
    elif provider == "ollama":
        model = app_settings.ollama_model_embed
    else:
        raise RuntimeError(f"Provider '{provider}' has no embedding model.")
    return f"{PAPERS_PREFIX}__{provider}__{_sanitize(model)}"


_client: chromadb.PersistentClient | None = None


def get_client() -> chromadb.PersistentClient:
    """Process-wide singleton client."""
    global _client
    if _client is None:
        DB_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(DB_PATH),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_or_create_papers_collection() -> Collection:
    """Get the collection that matches the *currently configured* embedder."""
    return get_client().get_or_create_collection(
        name=collection_name_for_current_embedder(),
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,  # we embed externally
    )


def list_collections() -> list[dict]:
    """Diagnostic — what's in the store?"""
    out = []
    for c in get_client().list_collections():
        col = get_client().get_collection(c.name, embedding_function=None)
        out.append({"name": c.name, "count": col.count()})
    return out


def reset_papers_collection() -> str:
    """Delete the current embedder's collection. Use with care."""
    name = collection_name_for_current_embedder()
    try:
        get_client().delete_collection(name)
    except Exception:
        pass
    return name
