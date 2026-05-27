"""
Naive RAG pipeline — the textbook retrieve / stuff / generate flow.

This is intentionally the simplest thing that works end-to-end. We're going
to BEAT this in every later phase, so it doubles as a baseline for evaluation.

Prompt design notes
-------------------
We number sources [S1]...[Sk] in the context block and ask the model to cite
inline by tag. We then resolve those tags back to filenames+pages in the final
output — this makes citations *grounded* (the LLM can't hallucinate a page
that wasn't in its context).

If retrieval returns zero chunks (empty corpus, weak query), we short-circuit
with a clear "I don't know" instead of inviting hallucination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import ModelTier
from src.llm import chat
from src.retrieval import RetrievedChunk, naive_retrieve


SYSTEM_PROMPT = """You are a careful research assistant. Answer the user's question \
using ONLY the numbered sources provided below.

Rules:
- Cite every factual claim inline using the tags [S1], [S2], etc.
- If the sources do not contain the answer, say so plainly. Do NOT invent facts.
- Prefer concise, direct answers. Use bullet points for lists.
- Quote short phrases only when wording matters; otherwise paraphrase."""

USER_TEMPLATE = """Question: {question}

Sources:
{context_block}

Write the answer now. Remember: cite with [S1], [S2], etc."""


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)

    def formatted(self) -> str:
        """Render answer + a 'Sources' footer mapping tags to citations."""
        if not self.sources:
            return self.answer
        lines = [self.answer.strip(), "", "Sources:"]
        for i, s in enumerate(self.sources, start=1):
            lines.append(f"  [S{i}] {s.citation}  (score={s.score:.2f})")
        return "\n".join(lines)


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as numbered [S1]..[Sk] sections for the prompt."""
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        header = f"[S{i}] {c.citation}"
        if c.doc_title:
            header += f"  — {c.doc_title}"
        parts.append(f"{header}\n{c.text.strip()}")
    return "\n\n---\n\n".join(parts)


def naive_rag(question: str, *, k: int = 5) -> RAGAnswer:
    """Retrieve top-k chunks and generate a cited answer."""
    chunks = naive_retrieve(question, k=k)

    if not chunks:
        return RAGAnswer(
            question=question,
            answer="I don't have any documents indexed yet. Run `researgent ingest` first.",
            sources=[],
        )

    context_block = _build_context_block(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            question=question, context_block=context_block
        )},
    ]
    answer = chat(messages, tier=ModelTier.REASONING, temperature=0.1, max_tokens=900)
    return RAGAnswer(question=question, answer=answer.strip(), sources=chunks)
