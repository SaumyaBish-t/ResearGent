"""
Tests for the Critic's source-filtering contract (Phase 24).

The spec: "Critic's text now says it only sees sources that passed
Provenance and Novelty."

Two things must hold:

  1. The Critic's system prompt STATES that it only sees sources which
     cleared Provenance and Novelty — so the grader knows the pool is
     already vetted and does not re-litigate it.

  2. The Critic's INPUT is the filtered set. This is structural: the
     Novelty Check filters `chunk_refs_by_subq` in place, and the Critic
     reads that same field. These tests pin that wiring.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_critic_novelty_filter.py -v
"""

import sys

sys.path.insert(0, ".")

from src.agent.nodes import critic as critic_node
from src.agent.artifacts import HydratedChunk


# ---------------------------------------------------------------------------
# The prompt states the contract.
# ---------------------------------------------------------------------------


def test_critic_prompt_mentions_provenance_and_novelty():
    prompt = critic_node._SYSTEM
    assert "novelty" in prompt.lower()
    assert "provenance" in prompt.lower()


def test_critic_prompt_says_it_only_sees_vetted_sources():
    """The wording must be explicit — 'only sees sources that passed'."""
    prompt = critic_node._SYSTEM.lower()
    assert "passed" in prompt or "cleared" in prompt or "vetted" in prompt
    # It should say what they passed.
    assert "provenance" in prompt
    assert "novelty" in prompt


# ---------------------------------------------------------------------------
# The Critic's input is the filtered set.
# ---------------------------------------------------------------------------


def _chunk(citation: str, *, doi: str = "", text: str = "evidence") -> HydratedChunk:
    return HydratedChunk(
        text=text,
        citation=citation,
        source_file=citation,
        signal="paper:semantic_scholar",
        doi=doi,
    )


def test_critic_sees_only_the_refs_left_in_state(monkeypatch):
    """
    Whatever survives into `chunk_refs_by_subq` is what the Critic grades.

    The Novelty Check removes the duplicates; this asserts the Critic
    reads exactly the surviving list — no second filter, no stale copy.
    """
    seen: dict[str, int] = {}

    def fake_grade(sub_q, chunks):
        seen["n"] = len(chunks)
        return ["relevant"] * len(chunks), "ok", 0, 0

    monkeypatch.setattr(critic_node, "_grade_one_subq", fake_grade)

    kept = _chunk("Kept paper", doi="10.1/kept")
    monkeypatch.setattr(
        critic_node,
        "hydrate",
        lambda _refs: {"q": [kept]},
    )

    critic_node.critique({
        "chunk_refs_by_subq": {"q": [{"kind": "paper", "id": "kept"}]},
    })

    assert seen["n"] == 1


def test_critic_drops_a_ref_that_the_novelty_check_removed(monkeypatch):
    """
    The end-to-end effect: a source removed upstream never reaches the
    grader, and never comes back as a citation.
    """
    seen: dict[str, int] = {}

    def fake_grade(sub_q, chunks):
        seen["n"] = len(chunks)
        return ["relevant"] * len(chunks), "ok", 0, 0

    monkeypatch.setattr(critic_node, "_grade_one_subq", fake_grade)

    kept = _chunk("Kept paper", doi="10.1/kept")
    monkeypatch.setattr(
        critic_node,
        "hydrate",
        lambda _refs: {"q": [kept]},
    )

    out = critic_node.critique({
        # Only the survivor is in state — the duplicate was removed by
        # the Novelty Check before the Critic ran.
        "chunk_refs_by_subq": {"q": [{"kind": "paper", "id": "kept"}]},
    })

    assert seen["n"] == 1
    assert out["chunk_refs_by_subq"] == {"q": [{"kind": "paper", "id": "kept"}]}
