"""Agent graph nodes. Each is a pure function: state -> state-update dict."""

from . import (
    critic,
    generator,
    llm_reasoning,
    memory_keeper,
    originality_check,
    paper_discovery,
    planner,
    provenance_check,
    reflector,
    retriever,
    rewriter,
    verifier,
    web_fallback,
)
