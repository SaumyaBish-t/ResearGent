"""
Contract tests for the Phase 24 frontend wiring.

The frontend lives in TypeScript/Next.js and has no test runner configured,
so these are STATIC contract checks: they parse the frontend source files
and assert the backend↔frontend contract holds.

This is the same idea as `scripts/audit_contract_drift.py`, but as pytest,
so it runs in the same command as the rest of the suite and fails loudly in
CI.

What it asserts (the four touchpoints a new graph node needs):
  1. `NodeId` union contains the node.
  2. 3D `NODES[]` contains the node.
  3. `EDGES[]` contains every real transition.
  4. `summarizeNode()` renders a non-default summary for the node.

Plus the Phase 24-specific data contract:
  5. `NodeSummary` declares the novelty fields the backend emits.
  6. `FinalEvent` carries `novelty_report`.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_frontend_contract.py -v
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pytest


FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
TYPES = FRONTEND / "lib" / "types.ts"
GRAPH_CONFIG = FRONTEND / "lib" / "graph-config.ts"
STORE = FRONTEND / "lib" / "store.ts"


pytestmark = pytest.mark.skipif(
    not FRONTEND.exists(),
    reason="frontend/ directory not present",
)


@pytest.fixture(scope="module")
def types_src() -> str:
    return TYPES.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def graph_src() -> str:
    return GRAPH_CONFIG.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def store_src() -> str:
    return STORE.read_text(encoding="utf-8")


def _node_id_union(types_src: str) -> set[str]:
    """Extract the node ids from the `NodeId` union."""
    m = re.search(r"export type NodeId\s*=(.*?);", types_src, re.DOTALL)
    assert m, "could not find the NodeId union in types.ts"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def _node_ids_in_nodes_array(graph_src: str) -> set[str]:
    """Extract the ids declared in the 3D `NODES[]` array."""
    m = re.search(r"export const NODES[^=]*=\s*\[(.*?)\n\];", graph_src, re.DOTALL)
    assert m, "could not find the NODES array in graph-config.ts"
    return set(re.findall(r'\{\s*id:\s*"([^"]+)"', m.group(1)))


def _edge_pairs(graph_src: str) -> set[tuple[str, str]]:
    """Extract the (from, to) pairs declared in the 3D `EDGES[]` array."""
    m = re.search(r"export const EDGES[^=]*=\s*\[(.*?)\n\];", graph_src, re.DOTALL)
    assert m, "could not find the EDGES array in graph-config.ts"
    return {
        (a, b)
        for a, b in re.findall(
            r'\{\s*from:\s*"([^"]+)"\s*,\s*to:\s*"([^"]+)"', m.group(1)
        )
    }


# ---------------------------------------------------------------------------
# 1. NodeId union
# ---------------------------------------------------------------------------


def test_node_id_union_has_the_phase_24_nodes(types_src):
    ids = _node_id_union(types_src)
    assert "novelty_check" in ids
    assert "human_review_gate" in ids


def test_node_id_union_still_has_the_existing_nodes(types_src):
    """A refactor must not silently drop an existing node."""
    ids = _node_id_union(types_src)
    for node in (
        "planner", "retriever", "provenance_check", "critic", "rewriter",
        "paper_discovery", "web_fallback", "reflector", "generator",
        "verifier", "memory_keeper", "llm_reasoning", "no_answer", "vault",
    ):
        assert node in ids, f"{node} vanished from the NodeId union"


# ---------------------------------------------------------------------------
# 2. 3D NODES[]
# ---------------------------------------------------------------------------


def test_nodes_array_has_the_phase_24_nodes(graph_src):
    ids = _node_ids_in_nodes_array(graph_src)
    assert "novelty_check" in ids
    assert "human_review_gate" in ids


def test_phase_24_nodes_have_a_label_and_position(graph_src):
    """Every node needs a label + sublabel + 3D position or it can't render."""
    for node_id in ("novelty_check", "human_review_gate"):
        # Build the pattern outside the f-string: Python <3.12 forbids
        # backslashes inside f-string expression parts.
        pattern = r'\{\s*id:\s*"' + node_id + r'"[^}}]*\}'
        m = re.search(pattern, graph_src, re.DOTALL)
        assert m, f"{node_id} has no NODES[] entry"
        entry = m.group(0)
        assert "label:" in entry, f"{node_id} has no label"
        assert "sublabel:" in entry, f"{node_id} has no sublabel"
        assert "position:" in entry, f"{node_id} has no position"
        assert "accent:" in entry, f"{node_id} has no accent"


def test_nodes_array_keeps_the_story_nodes(graph_src):
    """The 7 curated story nodes must survive the array edit."""
    ids = _node_ids_in_nodes_array(graph_src)
    for node in (
        "planner", "retriever", "critic", "rewriter",
        "paper_discovery", "web_fallback", "vault",
    ):
        assert node in ids, f"{node} vanished from NODES[]"


# ---------------------------------------------------------------------------
# 3. 3D EDGES[]
# ---------------------------------------------------------------------------


def test_edges_have_the_phase_24_transitions(graph_src):
    edges = _edge_pairs(graph_src)
    for a, b in (
        ("provenance_check", "novelty_check"),
        ("novelty_check", "human_review_gate"),
        ("human_review_gate", "critic"),
    ):
        assert (a, b) in edges, f"missing edge {a} -> {b}"


def test_edges_have_the_repositioned_originality_transitions(graph_src):
    """Originality moved to sit between Answer Generation and Vault."""
    edges = _edge_pairs(graph_src)
    assert ("generator", "verifier") in edges
    assert ("verifier", "reflector") in edges
    assert ("reflector", "originality_check") in edges
    assert ("originality_check", "memory_keeper") in edges


def test_edges_no_longer_skip_provenance_to_the_critic(graph_src):
    """The Critic must not be reachable directly from Provenance."""
    edges = _edge_pairs(graph_src)
    assert ("provenance_check", "critic") not in edges


# ---------------------------------------------------------------------------
# 4. summarizeNode()
# ---------------------------------------------------------------------------


def test_store_renders_a_novelty_summary(store_src):
    """A node with no summarizeNode() case falls through to `default:`
    and renders its raw id — that is how a finished backend feature ships
    with no visible UI."""
    m = re.search(r"function summarizeNode\(.*?\n\}", store_src, re.DOTALL)
    assert m, "could not find summarizeNode() in store.ts"
    body = m.group(0)
    assert 'case "novelty_check"' in body


def test_store_renders_a_review_gate_summary(store_src):
    m = re.search(r"function summarizeNode\(.*?\n\}", store_src, re.DOTALL)
    body = m.group(0)
    assert 'case "human_review_gate"' in body


def test_store_handles_the_review_decision(store_src):
    """The approve/reject decision must reach the log panel."""
    m = re.search(r"function summarizeNode\(.*?\n\}", store_src, re.DOTALL)
    body = m.group(0)
    assert "decision" in body


# ---------------------------------------------------------------------------
# 5-6. Data contract
# ---------------------------------------------------------------------------


def test_node_summary_declares_the_novelty_fields(types_src):
    m = re.search(r"export interface NodeSummary\s*\{(.*?)\n\}", types_src, re.DOTALL)
    assert m, "could not find the NodeSummary interface"
    body = m.group(1)
    for field in (
        "verdict?",
        "is_novel?",
        "overall_similarity?",
        "approach_similarity?",
        "result_similarity?",
        "requires_review?",
        "matched_metrics?",
        "compared_papers?",
        "dropped_sources?",
    ):
        assert field in body, f"NodeSummary is missing {field}"


def test_node_summary_declares_the_review_gate_fields(types_src):
    m = re.search(r"export interface NodeSummary\s*\{(.*?)\n\}", types_src, re.DOTALL)
    body = m.group(1)
    assert "decision?" in body
    assert "similarity?" in body
    assert "threshold?" in body
    assert "approved?" in body


def test_final_event_carries_the_novelty_report(types_src):
    m = re.search(r"export interface FinalEvent\s*\{(.*?)\n\}", types_src, re.DOTALL)
    assert m, "could not find the FinalEvent interface"
    assert "novelty_report?" in m.group(1)


def test_types_declare_a_novelty_metric_match(types_src):
    """The matched-metric rows need a typed shape to render."""
    assert "NoveltyMetricMatch" in types_src or "matched_metrics?:" in types_src
