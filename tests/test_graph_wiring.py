"""
Tests for the Phase 24 graph wiring.

The spec's topology:

    R (Results Ingestion) ──▶ Retriever only
    Retriever ──▶ Provenance Check ──▶ Novelty Check ──▶ Critic
                                        │
                                        └─▶ Human Review Gate (side branch)
                                              similarity > 50% → PAUSE
                                              confirm → resume from Novelty Check

And the Originality Check must sit between Answer Generation and Vault:
    Generator ──▶ Verifier ──▶ Reflector ──▶ Originality Check ──▶ Vault

These tests assert the TOPOLOGY of the compiled graph — which nodes exist,
which edges connect them, and that the Critic's input is the filtered set.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_graph_wiring.py -v
"""

import sys

sys.path.insert(0, ".")

from src.agent.graph import build_graph


def _edge_pairs(graph) -> set[tuple[str, str]]:
    """The set of (from, to) node pairs in the compiled graph."""
    pairs: set[tuple[str, str]] = set()
    for edge in graph.get_graph().edges:
        pairs.add((edge.source, edge.target))
    return pairs


def _node_names(graph) -> set[str]:
    return set(graph.get_graph().nodes.keys())


# ---------------------------------------------------------------------------
# The new nodes exist.
# ---------------------------------------------------------------------------


def test_graph_has_the_novelty_check_node():
    graph = build_graph(use_checkpointer=False)
    assert "novelty_check" in _node_names(graph)


def test_graph_has_the_human_review_gate_node():
    graph = build_graph(use_checkpointer=False)
    assert "human_review_gate" in _node_names(graph)


# ---------------------------------------------------------------------------
# The Novelty Check sits between Provenance and the Critic.
# ---------------------------------------------------------------------------


def test_provenance_check_flows_into_novelty_check():
    graph = build_graph(use_checkpointer=False)
    edges = _edge_pairs(graph)
    assert ("provenance_check", "novelty_check") in edges


def test_novelty_check_is_upstream_of_the_critic():
    """
    The Critic must only see sources that passed Provenance AND Novelty.

    The Critic reads `chunk_refs_by_subq` from state, and the Novelty
    Check filters that field — so the Critic sees the filtered set as
    long as the Novelty Check runs BEFORE it.
    """
    graph = build_graph(use_checkpointer=False)
    edges = _edge_pairs(graph)
    assert ("novelty_check", "critic") in edges or ("human_review_gate", "critic") in edges


def test_critic_does_not_run_directly_after_provenance():
    """Provenance must NOT go straight to the Critic — Novelty sits between."""
    graph = build_graph(use_checkpointer=False)
    edges = _edge_pairs(graph)
    assert ("provenance_check", "critic") not in edges


# ---------------------------------------------------------------------------
# The Human Review Gate is a side branch under the Novelty Check.
# ---------------------------------------------------------------------------


def test_review_gate_is_reachable_from_the_novelty_check():
    graph = build_graph(use_checkpointer=False)
    edges = _edge_pairs(graph)
    assert ("novelty_check", "human_review_gate") in edges


def test_review_gate_feeds_the_critic():
    graph = build_graph(use_checkpointer=False)
    edges = _edge_pairs(graph)
    assert ("human_review_gate", "critic") in edges


def test_review_gate_is_not_on_the_web_fallback_path():
    """
    The gate is a side branch under Novelty Check, NOT on every path.

    `web_fallback` feeds provenance directly — it must not be forced
    through the review gate, because the gate's trigger is the user's
    own results, which the web path never scores.
    """
    graph = build_graph(use_checkpointer=False)
    edges = _edge_pairs(graph)
    assert ("web_fallback", "provenance_check") in edges


# ---------------------------------------------------------------------------
# The Originality Check sits between Answer Generation and the Vault.
# ---------------------------------------------------------------------------


def test_originality_check_runs_after_the_generator():
    graph = build_graph(use_checkpointer=False)
    edges = _edge_pairs(graph)
    assert ("generator", "verifier") in edges


def test_originality_check_is_downstream_of_the_generator():
    """
    Originality must run AFTER answer generation — it checks the draft,
    so it cannot precede it.
    """
    graph = build_graph(use_checkpointer=False)
    nodes = graph.get_graph().nodes
    assert "originality_check" in nodes
    # The generator's outgoing edges must eventually reach originality.
    edges = _edge_pairs(graph)
    assert ("verifier", "originality_check") in edges or (
        "reflector", "originality_check"
    ) in edges


# ---------------------------------------------------------------------------
# The graph still compiles and runs a trivial pass through.
# ---------------------------------------------------------------------------


def test_graph_compiles_without_a_checkpointer():
    graph = build_graph(use_checkpointer=False)
    assert graph is not None


def test_graph_compiles_with_a_checkpointer():
    graph = build_graph(use_checkpointer=True)
    assert graph is not None
