"""
Tests for the Retriever consuming the user's own results (Phase 24).

The spec: "R (Results Ingestion) feeds the Retriever only."

When the user has ingested their own experimental results, the Retriever
must look for the RELATED PRIOR WORK — papers doing the same thing on the
same metric — because that is the only thing the Novelty Check can
meaningfully compare against.

Key property: the user's results must NOT become a retrieval QUERY
verbatim in a way that leaks the claim, and R must not feed any node
other than the Retriever.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_retriever_user_results.py -v
"""

import sys

sys.path.insert(0, ".")

from src.agent.nodes import retriever as retriever_node


USER_RESULTS = {
    "summary": "Sparse attention with learned pruning masks.",
    "approach": "Sparse attention with learned pruning masks.",
    "dataset": "WMT14",
    "metrics": [{"name": "BLEU-4", "value": 38.1, "numeric_value": 38.1}],
    "search_queries": [
        "Sparse attention with learned pruning masks",
        "Sparse attention with learned pruning masks WMT14",
        "BLEU-4 WMT14",
    ],
}


# ---------------------------------------------------------------------------
# The Retriever derives queries from the user's results.
# ---------------------------------------------------------------------------


def test_retriever_emits_queries_derived_from_user_results(monkeypatch):
    """The Retriever must search for the related prior work."""
    monkeypatch.setattr(retriever_node.settings, "enable_local_retrieval", True)

    seen_queries: list[str] = []

    def fake_retrieve(q, k=8, doc_ids=None, domains=None):
        seen_queries.append(q)
        return []

    monkeypatch.setattr(retriever_node, "hybrid_retrieve", fake_retrieve)
    monkeypatch.setattr(retriever_node, "persist_mixed", lambda _t, _c: {})
    monkeypatch.setattr(retriever_node.settings, "graph_expansion_enabled", False)

    retriever_node.retrieve({
        "question": "How does sparse attention work?",
        "sub_questions": ["How does sparse attention work?"],
        "user_results": USER_RESULTS,
    })

    assert seen_queries, "the retriever issued no queries"
    joined = " | ".join(seen_queries).lower()
    assert "sparse attention" in joined


def test_retriever_uses_the_dataset_in_its_query(monkeypatch):
    monkeypatch.setattr(retriever_node.settings, "enable_local_retrieval", True)

    seen_queries: list[str] = []

    def fake_retrieve(q, k=8, doc_ids=None, domains=None):
        seen_queries.append(q)
        return []

    monkeypatch.setattr(retriever_node, "hybrid_retrieve", fake_retrieve)
    monkeypatch.setattr(retriever_node, "persist_mixed", lambda _t, _c: {})
    monkeypatch.setattr(retriever_node.settings, "graph_expansion_enabled", False)

    retriever_node.retrieve({
        "question": "How does sparse attention work?",
        "sub_questions": ["How does sparse attention work?"],
        "user_results": USER_RESULTS,
    })

    joined = " | ".join(seen_queries).lower()
    assert "wmt14" in joined


def test_retriever_records_the_derived_queries_in_the_trace(monkeypatch):
    monkeypatch.setattr(retriever_node.settings, "enable_local_retrieval", True)

    def fake_retrieve(q, k=8, doc_ids=None, domains=None):
        return []

    monkeypatch.setattr(retriever_node, "hybrid_retrieve", fake_retrieve)
    monkeypatch.setattr(retriever_node, "persist_mixed", lambda _t, _c: {})
    monkeypatch.setattr(retriever_node.settings, "graph_expansion_enabled", False)

    out = retriever_node.retrieve({
        "question": "How does sparse attention work?",
        "sub_questions": ["How does sparse attention work?"],
        "user_results": USER_RESULTS,
    })

    trace = out.get("trace") or []
    assert any(t.get("user_results_queries") for t in trace)


# ---------------------------------------------------------------------------
# Without user results, behaviour is unchanged.
# ---------------------------------------------------------------------------


def test_retriever_without_user_results_uses_the_sub_question(monkeypatch):
    monkeypatch.setattr(retriever_node.settings, "enable_local_retrieval", True)

    seen_queries: list[str] = []

    def fake_retrieve(q, k=8, doc_ids=None, domains=None):
        seen_queries.append(q)
        return []

    monkeypatch.setattr(retriever_node, "hybrid_retrieve", fake_retrieve)
    monkeypatch.setattr(retriever_node, "persist_mixed", lambda _t, _c: {})
    monkeypatch.setattr(retriever_node.settings, "graph_expansion_enabled", False)

    retriever_node.retrieve({
        "question": "How does sparse attention work?",
        "sub_questions": ["How does sparse attention work?"],
    })

    assert seen_queries == ["How does sparse attention work?"]


def test_retriever_without_user_results_has_no_query_trace(monkeypatch):
    monkeypatch.setattr(retriever_node.settings, "enable_local_retrieval", True)

    def fake_retrieve(q, k=8, doc_ids=None, domains=None):
        return []

    monkeypatch.setattr(retriever_node, "hybrid_retrieve", fake_retrieve)
    monkeypatch.setattr(retriever_node, "persist_mixed", lambda _t, _c: {})
    monkeypatch.setattr(retriever_node.settings, "graph_expansion_enabled", False)

    out = retriever_node.retrieve({
        "question": "How does sparse attention work?",
        "sub_questions": ["How does sparse attention work?"],
    })

    trace = out.get("trace") or []
    assert not any(t.get("user_results_queries") for t in trace)


# ---------------------------------------------------------------------------
# The helper is pure and testable on its own.
# ---------------------------------------------------------------------------


def test_derived_queries_helper_returns_the_search_queries():
    out = retriever_node._user_results_queries(USER_RESULTS)
    assert out == USER_RESULTS["search_queries"]


def test_derived_queries_helper_falls_back_to_building_them():
    """A hand-built payload without `search_queries` still works."""
    out = retriever_node._user_results_queries({
        "approach": "sparse attention",
        "dataset": "WMT14",
        "metrics": [{"name": "BLEU-4", "value": 38.1}],
    })
    assert out
    assert any("sparse attention" in q.lower() for q in out)


def test_derived_queries_helper_returns_empty_without_results():
    assert retriever_node._user_results_queries(None) == []
    assert retriever_node._user_results_queries({}) == []
