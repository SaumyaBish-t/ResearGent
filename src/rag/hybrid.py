"""
Hybrid RAG — top-k from `hybrid_retrieve` + same cited-generation prompt.

This is intentionally a near-clone of `naive_rag` so a side-by-side benchmark
(same prompt, same model, only the retriever differs) cleanly isolates the
retrieval gain. That's the whole point of Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import ModelTier
from src.llm import chat
from src.rag.naive import SYSTEM_PROMPT, USER_TEMPLATE
from src.retrieval import HybridChunk, hybrid_retrieve


@dataclass
class HybridRAGAnswer:
    question: str
    answer: str
    sources: list[HybridChunk] = field(default_factory=list)

    def formatted(self) -> str:
        if not self.sources:
            return self.answer
        lines = [self.answer.strip(), "", "Sources:"]
        for i, s in enumerate(self.sources, start=1):
            tag = f"[S{i}]"
            origin = f"signal={s.signal}, rrf={s.rrf_score:.4f}"
            if s.dense_rank:
                origin += f", dense#{s.dense_rank}"
            if s.bm25_rank:
                origin += f", bm25#{s.bm25_rank}"
            lines.append(f"  {tag} {s.citation}  ({origin})")
        return "\n".join(lines)


def _build_context_block(chunks: list[HybridChunk]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        header = f"[S{i}] {c.citation}"
        if c.doc_title:
            header += f"  -- {c.doc_title}"
        parts.append(f"{header}\n{c.text.strip()}")
    return "\n\n---\n\n".join(parts)


def hybrid_rag(question: str, *, k: int = 5) -> HybridRAGAnswer:
    """Hybrid retrieval + cited generation. Same prompt as naive_rag."""
    chunks = hybrid_retrieve(question, k=k)
    if not chunks:
        return HybridRAGAnswer(
            question=question,
            answer="I don't have any documents indexed yet. Run `researgent ingest` first.",
            sources=[],
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            question=question, context_block=_build_context_block(chunks)
        )},
    ]
    answer = chat(messages, tier=ModelTier.REASONING, temperature=0.1, max_tokens=900)
    return HybridRAGAnswer(question=question, answer=answer.strip(), sources=chunks)
