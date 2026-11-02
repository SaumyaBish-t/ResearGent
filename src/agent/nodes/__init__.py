"""Agent graph nodes. Each is a pure function: state -> state-update dict."""

from . import (
    critic,
    generator,
    llm_reasoning,
    memory_keeper,
    novelty_check,
    originality_check,
    paper_discovery,
    planner,
    provenance_check,
    reflector,
    retriever,
    rewriter,
    review_gate,
    verifier,
    web_fallback,
)
