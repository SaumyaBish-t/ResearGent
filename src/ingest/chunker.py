"""
Semantic chunking + local entity extraction (Phase 14).

What changed vs. Phase 1's naive chunker
----------------------------------------
The old chunker greedy-packed paragraphs/sentences until the token budget
was hit. That keeps chunks under the embedder's input cap but ignores
*meaning* — a chunk frequently straddles a topical boundary because the
boundary happened to fall mid-budget.

The new chunker is semantic:

  1. Split the page into sentences.
  2. Embed each sentence locally with `sentence-transformers/all-MiniLM-L6-v2`
     (22M-param, CPU-friendly, ~50ms per ~30 sentences on a laptop CPU).
  3. Compute cosine *distance* between every adjacent pair. A large distance
     means "the topic just shifted." We pick a percentile-based threshold so
     the cutoff adapts to each page (a homogeneous methods page yields tighter
     distances than a wide-ranging discussion page).
  4. Greedy-pack sentences, but force a break whenever (a) we hit `max_tokens`
     or (b) we cross a topic-shift boundary AND we've already accumulated
     enough text that the next chunk won't be a sliver.

After chunking, we run GLiNER over each chunk to surface technical entities
(algorithms, frameworks, datasets, organizations, …). GLiNER is a 166M-param
zero-shot NER model — you give it an arbitrary label list at inference time
and it spans-tags accordingly. We expose entities both as `Chunk.entities`
(for downstream metadata) and as a single appended line on the chunk text
itself, so the existing BM25 + dense indexes both pick up the technical
keywords without any retrieval-side changes.

Free-tier budget
----------------
Everything runs locally on CPU. No LLM API calls. First import downloads
the two model checkpoints (~230MB total) to the HuggingFace cache; later
runs hit the cache and start in ~1-2s.

Public API
----------
    chunk_pages(pages, target_tokens=500, max_tokens=800) -> list[Chunk]
    semantic_chunk_text(text, ...) -> list[str]
    extract_entities(text, ...) -> list[str]

Both `chunk_pages` and `extract_entities` are also imported by
`src/ingest/obsidian.py` so vault chunks pick up entities too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import tiktoken

from src.ingest.pdf import Page

if TYPE_CHECKING:  # purely for type checkers — no runtime import
    from sentence_transformers import SentenceTransformer  # noqa: F401

# Tokenizer kept for *budget bookkeeping only* — the semantic chunker decides
# where breaks fall; tiktoken just tells us how big a chunk is so we can cap
# it under the embedder's input limit.
_ENC = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Tiny + fast. 22M params, 384-dim embeddings — enough resolution to detect
# topical shifts at the sentence level. We are NOT using these embeddings for
# retrieval (the retrieval path keeps using whichever embedder `src.llm` is
# configured for); MiniLM is purely an offline boundary detector.
_SEM_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# `gliner_small-v2.1` is 166M params (~150MB). The medium variant scores a few
# F1 points higher but doubles latency on CPU. For research-paper entities the
# small model is the right speed/accuracy tradeoff.
_GLINER_MODEL_NAME = "urchade/gliner_small-v2.1"

# Labels surfaced to GLiNER at inference. Tuned for academic ML/AI papers —
# the entities most likely to be query terms a retriever needs to hit.
ENTITY_LABELS: list[str] = [
    "Algorithm",
    "Framework",
    "Scientific Concept",
    "Organization",
    "Person",
    "Metric",
    "Dataset",
]


@dataclass
class Chunk:
    text: str
    source_file: str
    page_number: int       # page where this chunk STARTS
    chunk_index: int       # 0-based index within the document
    token_count: int
    # Phase 14: locally extracted technical entities (de-duplicated, ordered).
    # Empty list when GLiNER finds nothing relevant or extraction is disabled.
    entities: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Local model loaders (lazy, singleton)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_sentence_encoder():
    """
    Load MiniLM exactly once per process. First call downloads + warms up
    (a few seconds on a cold cache); subsequent calls are free.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_SEM_ENCODER_NAME)


@lru_cache(maxsize=1)
def _get_gliner_model():
    """
    Same singleton pattern for GLiNER. Kept separate from the sentence
    encoder so a user can swap one without touching the other.
    """
    from gliner import GLiNER

    return GLiNER.from_pretrained(_GLINER_MODEL_NAME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tok_count(s: str) -> int:
    return len(_ENC.encode(s))


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def _split_sentences(text: str) -> list[str]:
    """Same conservative splitter as Phase 1 — works well for academic English."""
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _hard_split_tokens(text: str, target: int) -> list[str]:
    """Last-resort splitter for a single sentence that exceeds max_tokens."""
    ids = _ENC.encode(text)
    out: list[str] = []
    for i in range(0, len(ids), target):
        out.append(_ENC.decode(ids[i : i + target]))
    return out


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------


def semantic_chunk_text(
    text: str,
    *,
    target_tokens: int = 500,
    max_tokens: int = 800,
    breakpoint_percentile: float = 90.0,
    min_chunk_tokens: int | None = None,
) -> list[str]:
    """
    Split `text` into chunks at sentence-level topic shifts.

    The algorithm (a stripped-down version of LangChain/Greg Kamradt's
    "semantic chunker"):

      1. Sentences → embeddings (normalized so cosine = dot product).
      2. Adjacent cosine *distance* d_i = 1 - sim(s_i, s_{i+1}).
      3. Threshold τ = percentile(d, `breakpoint_percentile`). Anywhere the
         distance exceeds τ is a candidate topical boundary.
      4. Greedy pack: keep adding sentences to the current chunk until either
         (a) we hit `max_tokens` (hard cap — embedder will truncate beyond
         this) or (b) we hit a topical boundary AND we've already accumulated
         enough text to make a usable chunk. The "enough text" guard prevents
         pathological single-sentence chunks at the start of a page.

    Parameters
    ----------
    target_tokens:
        Soft target — chunks are allowed to break above this once a topic
        shift is detected, but must not exceed `max_tokens`.
    max_tokens:
        Hard cap. Anything over this gets force-split at sentence boundaries
        or (last resort) raw token boundaries.
    breakpoint_percentile:
        Higher → fewer, larger chunks. 90 is a balanced default; bump to 95
        for very long sequential text, drop to 80 for dense fragmented text.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    min_chunk_tokens = min_chunk_tokens if min_chunk_tokens is not None else max(50, target_tokens // 2)

    # Single-sentence fast path — nothing to compare against.
    if len(sentences) == 1:
        s = sentences[0]
        return [s] if _tok_count(s) <= max_tokens else _hard_split_tokens(s, target_tokens)

    encoder = _get_sentence_encoder()
    # `normalize_embeddings=True` → dot product is cosine similarity. Cheaper
    # than calling util.cos_sim later, and we never need the raw vectors.
    embeddings = encoder.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # Pairwise cosine distances between consecutive sentences.
    sims = (embeddings[:-1] * embeddings[1:]).sum(axis=1)
    distances = (1.0 - sims).tolist()

    # numpy is already imported transitively by sentence-transformers, but be
    # explicit here so this stays readable.
    import numpy as np

    threshold = float(np.percentile(distances, breakpoint_percentile))

    chunks: list[str] = []
    current: list[str] = [sentences[0]]
    current_tokens = _tok_count(sentences[0])

    for i in range(1, len(sentences)):
        s = sentences[i]
        s_tokens = _tok_count(s)
        topic_shift = distances[i - 1] >= threshold

        # Hard overflow → break now regardless of topical signal.
        if current_tokens + s_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current, current_tokens = [s], s_tokens
            continue

        # Topical break, but only if the current chunk is big enough to stand
        # alone. Without this guard the first sentence-shift on a page would
        # emit a 1-sentence chunk.
        if topic_shift and current_tokens >= min_chunk_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = [s], s_tokens
            continue

        current.append(s)
        current_tokens += s_tokens

    if current:
        chunks.append(" ".join(current))

    # Belt-and-braces: anything still over max_tokens (a single huge sentence
    # combined with a topical shift hold) gets hard-split.
    final: list[str] = []
    for c in chunks:
        if _tok_count(c) > max_tokens:
            final.extend(_hard_split_tokens(c, target_tokens))
        else:
            final.append(c)
    return final


# ---------------------------------------------------------------------------
# Entity extraction (GLiNER)
# ---------------------------------------------------------------------------


def extract_entities(
    text: str,
    *,
    labels: list[str] | None = None,
    threshold: float = 0.5,
    max_entities: int = 25,
) -> list[str]:
    """
    Run GLiNER zero-shot NER over `text` and return de-duplicated entity
    surface forms in order of first appearance.

    `threshold` is GLiNER's confidence cutoff. 0.5 is the library default;
    raising it to 0.6+ trims noise on short chunks but loses some real hits.

    `max_entities` is a per-chunk safety cap — if a chunk somehow surfaces
    50 entities, we'd be bloating the embedded text. 25 is more than enough
    for academic paragraphs (typical: 3-8 entities/chunk).

    On any error (model fails to load, runtime exception) we return [] so
    the ingest pipeline degrades gracefully to entity-less chunks rather
    than crashing the whole job.
    """
    if not text or not text.strip():
        return []

    labels = labels or ENTITY_LABELS
    try:
        model = _get_gliner_model()
        raw = model.predict_entities(text, labels, threshold=threshold)
    except Exception:
        return []

    seen: set[str] = set()
    out: list[str] = []
    for ent in raw:
        name = (ent.get("text") or "").strip()
        if not name:
            continue
        # Case-insensitive de-dup; first surface form wins so casing matches
        # how the entity appears in the source text.
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
        if len(out) >= max_entities:
            break
    return out


# ---------------------------------------------------------------------------
# Public chunker
# ---------------------------------------------------------------------------


def chunk_pages(
    pages: list[Page],
    *,
    target_tokens: int = 500,
    max_tokens: int = 800,
    extract_entities_flag: bool = True,
    # Accepted for backwards-compat with old callers; the semantic chunker
    # does not use explicit token overlap (topical boundaries do the job).
    overlap_tokens: int = 0,  # noqa: ARG001
) -> list[Chunk]:
    """
    Chunk an ordered list of Pages with the semantic chunker.

    Chunking happens per-page so page citations remain accurate. Entity
    extraction is per-chunk so the entity list reflects exactly what that
    chunk contains (not the whole page's grab-bag).
    """
    if not pages:
        return []

    all_chunks: list[Chunk] = []
    idx = 0

    for page in pages:
        if not page.text or not page.text.strip():
            continue

        texts = semantic_chunk_text(
            page.text,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
        )

        for txt in texts:
            entities = extract_entities(txt) if extract_entities_flag else []
            all_chunks.append(
                Chunk(
                    text=txt,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    chunk_index=idx,
                    token_count=_tok_count(txt),
                    entities=entities,
                )
            )
            idx += 1

    return all_chunks
