"""Agent graph nodes. Each is a pure function: state -> state-update dict."""

from . import (
    critic,
    generator,
    llm_reasoning,
    memory_keeper,
    paper_discovery,
    planner,
    reflector,
    retriever,
    rewriter,
    verifier,
    web_fallback,
)
