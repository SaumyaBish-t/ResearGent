"""
Tiny persistent vector store — numpy-backed cosine-similarity search.

Why we rolled our own instead of using ChromaDB
-----------------------------------------------
ChromaDB on Windows CPUs has hellish cold-start behavior — onnxruntime DLL
loading, hnswlib initialization, ~60-120s on first run. For our scale
(hundreds of chunks per corpus, maybe ~10k at the high end), a brute-force
cosine search on a normalized numpy matrix is:

  - Sub-100ms cold start (just numpy + pickle)
  - Sub-millisecond per-query for our scale
  - Zero ML dependencies (no onnxruntime, no sentence-transformers)
  - No HuggingFace model auto-downloads on first import
  - One pickle file per collection — trivially backed up / inspected / deleted

Public surface is intentionally a ChromaDB-shaped subset so the rest of the
codebase didn't have to change:
  - col.count() -> int
  - col.add(ids, embeddings, documents, metadatas) -> None
  - col.delete(ids=...) -> None
  - col.get(where=..., include=...) -> {"ids": [...], "documents": [...], "metadatas": [...]}
  - col.query(query_embeddings, n_results, include=...) -> chroma-shaped dict

If we ever need ANN / billion-scale, swapping back to a real vector DB is a
two-file change (this module + the embed_function=None argument).
"""

from __future__ import annotations

import pickle
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import ModelTier, settings as app_settings

DB_PATH = Path("data") / "store"
PAPERS_PREFIX = "papers"


def _sanitize(s: str) -> str:
    """Filesystem-safe collection name."""
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s)
    return s.strip("-._")[:60] or "default"


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
    elif provider == "openrouter":
        model = app_settings.openrouter_model_embed
    else:
        raise RuntimeError(f"Provider '{provider}' has no embedding model.")
    return f"{PAPERS_PREFIX}__{provider}__{_sanitize(model)}"


# ---------------------------------------------------------------------------
# Persistence + in-memory shape
# ---------------------------------------------------------------------------


@dataclass
class _Payload:
    """What we pickle to disk per collection."""

    # Parallel arrays — index i refers to the same chunk across all three.
    ids: list[str] = field(default_factory=list)
    embeddings: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    documents: list[str] = field(default_factory=list)
    metadatas: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Collection — the ChromaDB-shaped object
# ---------------------------------------------------------------------------


class Collection:
    """
    Persistent vector collection.

    Mutations are flushed to disk after every `add` / `delete`. For our write
    cadence (batch ingest, then mostly read) this is fine; if we ever push
    millions of chunks we'd batch writes.

    Thread-safety: a single lock guards in-memory state. Process-level
    concurrency on the same collection isn't supported (one ResearGent
    process per machine — typical for a desktop tool).
    """

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path
        self._lock = threading.Lock()
        self._payload = self._load_or_init()

    # ---- persistence ----

    def _load_or_init(self) -> _Payload:
        if not self.path.exists():
            return _Payload()
        try:
            with self.path.open("rb") as f:
                p = pickle.load(f)
            if not isinstance(p, _Payload):
                return _Payload()
            return p
        except Exception:
            return _Payload()

    def _flush(self) -> None:
        # Atomic write — tmp file then rename, so a crash mid-write leaves
        # the previous good payload intact.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(self._payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.path)

    # ---- public, ChromaDB-shaped API ----

    def count(self) -> int:
        return len(self._payload.ids)

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError("ids/embeddings/documents/metadatas length mismatch")
        if not ids:
            return

        new_emb = np.asarray(embeddings, dtype=np.float32)
        with self._lock:
            p = self._payload
            if p.embeddings.size == 0:
                p.embeddings = new_emb
            else:
                if new_emb.shape[1] != p.embeddings.shape[1]:
                    raise ValueError(
                        f"Embedding dim mismatch: existing={p.embeddings.shape[1]}, "
                        f"new={new_emb.shape[1]}. Did the embedder change? "
                        "Use `researgent store reset` to rebuild."
                    )
                p.embeddings = np.vstack([p.embeddings, new_emb])
            p.ids.extend(ids)
            p.documents.extend(documents)
            p.metadatas.extend(metadatas)
            self._flush()

    def delete(self, ids: list[str] | None = None) -> None:
        if not ids:
            return
        target = set(ids)
        with self._lock:
            p = self._payload
            keep_idx = [i for i, _id in enumerate(p.ids) if _id not in target]
            if len(keep_idx) == len(p.ids):
                return  # nothing matched
            p.ids = [p.ids[i] for i in keep_idx]
            p.documents = [p.documents[i] for i in keep_idx]
            p.metadatas = [p.metadatas[i] for i in keep_idx]
            if p.embeddings.size:
                p.embeddings = p.embeddings[keep_idx]
            self._flush()

    @staticmethod
    def _match(meta: dict, where: dict) -> bool:
        """
        Tiny `where` evaluator covering the subset of Chroma operators we use.

        Supported:
          {"key": "value"}                         -> equality
          {"key": {"$in":  [v1, v2, ...]}}         -> membership
          {"key": {"$eq":  v}}                     -> explicit equality
          {"key": {"$ne":  v}}                     -> inequality

        Anything else raises so a typo doesn't silently return everything.
        """
        for k, expected in where.items():
            actual = meta.get(k)
            if isinstance(expected, dict):
                if "$in" in expected:
                    if actual not in expected["$in"]:
                        return False
                elif "$eq" in expected:
                    if actual != expected["$eq"]:
                        return False
                elif "$ne" in expected:
                    if actual == expected["$ne"]:
                        return False
                else:
                    raise ValueError(
                        f"Unsupported where operator(s) for key {k!r}: {expected!r}"
                    )
            else:
                if actual != expected:
                    return False
        return True

    def get_by_ids(self, ids: list[str]) -> dict[str, list]:
        """
        O(len(ids)) indexed lookup by chunk id — for pointer-based hydration.

        Returns the same shape as `.get(where=...)` but skips the metadata
        scan. Used by `agent.artifacts.hydrate()` to rehydrate refs into
        chunks at the start of each node, so we never have to scan the
        full collection just to look up 50 known ids.
        """
        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}
        with self._lock:
            p = self._payload
            pos = {cid: i for i, cid in enumerate(p.ids)}
            idxs = [pos[i] for i in ids if i in pos]
            return {
                "ids": [p.ids[i] for i in idxs],
                "documents": [p.documents[i] for i in idxs],
                "metadatas": [p.metadatas[i] for i in idxs],
            }

    def get(
        self,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict[str, list]:
        """ChromaDB-shaped get. `where` supports equality + `$in`/`$eq`/`$ne`."""
        include = include if include is not None else ["documents", "metadatas"]
        with self._lock:
            p = self._payload
            if where:
                idxs = [
                    i for i, m in enumerate(p.metadatas) if self._match(m, where)
                ]
            else:
                idxs = list(range(len(p.ids)))

            out: dict[str, list] = {"ids": [p.ids[i] for i in idxs]}
            if "documents" in include:
                out["documents"] = [p.documents[i] for i in idxs]
            if "metadatas" in include:
                out["metadatas"] = [p.metadatas[i] for i in idxs]
            return out

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        include: list[str] | None = None,
        where: dict | None = None,
    ) -> dict[str, list[list]]:
        """
        Top-k cosine similarity. Returns ChromaDB-shaped batched dict where
        outer list is per-query.

        Distances returned are cosine DISTANCE (1 - cos_sim), matching what
        ChromaDB returned for `hnsw:space=cosine` — so downstream code that
        does `score = 1 - distance` keeps working unchanged.
        """
        include = include if include is not None else ["documents", "metadatas", "distances"]
        with self._lock:
            p = self._payload
            if p.embeddings.size == 0 or not p.ids:
                # No data — return empty per-query lists.
                empty = [[] for _ in query_embeddings]
                out: dict[str, list[list]] = {"ids": empty}
                if "documents" in include: out["documents"] = empty
                if "metadatas" in include: out["metadatas"] = empty
                if "distances" in include: out["distances"] = empty
                return out

            # Restrict candidate pool by `where` BEFORE similarity sort, so
            # filters like {"doc_id": {"$in": [...]}} cost O(filtered) rather
            # than O(corpus). Big win when filtering to a handful of docs in
            # a corpus of thousands of chunks.
            if where:
                allowed = np.array(
                    [self._match(m, where) for m in p.metadatas], dtype=bool
                )
                if not allowed.any():
                    empty = [[] for _ in query_embeddings]
                    out: dict[str, list[list]] = {"ids": empty}
                    if "documents" in include: out["documents"] = empty
                    if "metadatas" in include: out["metadatas"] = empty
                    if "distances" in include: out["distances"] = empty
                    return out
                allowed_idx = np.nonzero(allowed)[0]
                corpus = p.embeddings[allowed_idx]
                id_map = allowed_idx
            else:
                corpus = p.embeddings
                id_map = np.arange(len(p.ids))

            # Normalize once. Could be cached, but for our scale recomputing
            # is sub-millisecond and avoids stale-cache bugs.
            corpus_n = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12)

            ids_out, docs_out, metas_out, dists_out = [], [], [], []
            for q in query_embeddings:
                qv = np.asarray(q, dtype=np.float32)
                qv = qv / (np.linalg.norm(qv) + 1e-12)
                sims = corpus_n @ qv  # cosine similarity for each row
                # argpartition then sort is faster than full argsort for top-k
                k = min(n_results, len(sims))
                if k <= 0:
                    ids_out.append([]); docs_out.append([]); metas_out.append([]); dists_out.append([])
                    continue
                top_unordered = np.argpartition(-sims, k - 1)[:k]
                top = top_unordered[np.argsort(-sims[top_unordered])]
                # `top` indexes the FILTERED corpus; map back to original
                # row indices for id/doc/metadata lookup.
                orig = [int(id_map[i]) for i in top]
                ids_out.append([p.ids[i] for i in orig])
                docs_out.append([p.documents[i] for i in orig])
                metas_out.append([p.metadatas[i] for i in orig])
                dists_out.append([float(1.0 - sims[top[j]]) for j in range(len(top))])

            out = {"ids": ids_out}
            if "documents" in include: out["documents"] = docs_out
            if "metadatas" in include: out["metadatas"] = metas_out
            if "distances" in include: out["distances"] = dists_out
            return out


# ---------------------------------------------------------------------------
# Client surface — keeps the ChromaDB-shaped public API
# ---------------------------------------------------------------------------


_collections: dict[str, Collection] = {}


def _path_for(name: str) -> Path:
    DB_PATH.mkdir(parents=True, exist_ok=True)
    return DB_PATH / f"{_sanitize(name)}.pkl"


def get_or_create_papers_collection() -> Collection:
    """Get the collection that matches the *currently configured* embedder."""
    name = collection_name_for_current_embedder()
    if name not in _collections:
        _collections[name] = Collection(name=name, path=_path_for(name))
    return _collections[name]


def list_collections() -> list[dict]:
    """All persisted collections + chunk counts. Diagnostic."""
    DB_PATH.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(DB_PATH.glob("*.pkl")):
        name = p.stem
        col = _collections.get(name) or Collection(name=name, path=p)
        out.append({"name": name, "count": col.count()})
    return out


def reset_papers_collection() -> str:
    """Drop the current embedder's collection (in-memory + on-disk)."""
    name = collection_name_for_current_embedder()
    _collections.pop(name, None)
    p = _path_for(name)
    if p.exists():
        p.unlink()
    return name
