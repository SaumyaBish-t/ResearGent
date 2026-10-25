from __future__ import annotations

from src.agent.artifacts import HydratedChunk, _to_hydrated
from src.agent.nodes import provenance_check
from src.agent.stream import _summarize_node_update
from src.retrieval.papers import PaperChunk


def _chunk(*, citation: str, doi: str) -> HydratedChunk:
    return HydratedChunk(
        text="evidence",
        citation=citation,
        source_file=citation,
        signal="paper:semantic_scholar",
        doi=doi,
    )


def test_semantic_scholar_doi_survives_paper_hydration() -> None:
    paper = PaperChunk(
        title="Retracted study",
        abstract="abstract",
        url="https://example.test/paper",
        authors=[],
        year=2020,
        source="semantic_scholar",
        doi="10.1234/retracted",
    )

    hydrated = _to_hydrated(paper)

    assert hydrated.doi == "10.1234/retracted"


def test_provenance_check_drops_retracted_refs_and_records_trace(monkeypatch) -> None:
    retracted = _chunk(citation="Retracted paper", doi="10.1234/retracted")
    clean = _chunk(citation="Clean paper", doi="10.1234/clean")
    refs = {
        "question": [
            {"kind": "paper", "id": "artifact-retracted"},
            {"kind": "paper", "id": "artifact-clean"},
        ]
    }

    async def fake_check_many(identifiers):
        assert set(identifiers) == {
            ("10.1234/retracted", ""),
            ("10.1234/clean", ""),
        }
        return {
            ("10.1234/retracted", ""): {
                "status": "retracted",
                "reason": "retraction",
                "notice_url": "https://example.test/retraction",
            },
            ("10.1234/clean", ""): {
                "status": "clean",
                "reason": "",
                "notice_url": "",
            },
        }

    monkeypatch.setattr(provenance_check.settings, "provenance_check_enabled", True)
    monkeypatch.setattr(
        provenance_check,
        "hydrate",
        lambda _refs: {"question": [retracted, clean]},
    )
    monkeypatch.setattr(provenance_check, "check_many", fake_check_many)

    out = provenance_check.provenance_check({"chunk_refs_by_subq": refs})

    assert out["chunk_refs_by_subq"] == {
        "question": [{"kind": "paper", "id": "artifact-clean"}]
    }
    assert out["provenance_dropped_count"] == 1
    assert out["provenance_flagged_count"] == 0
    event = out["trace"][0]
    assert event["retracted_dropped"] == 1
    assert event["blocked_sources"][0]["doi"] == "10.1234/retracted"


def test_stream_summary_exposes_blocked_sources() -> None:
    summary = _summarize_node_update(
        "provenance_check",
        {
            "trace": [
                {
                    "node": "provenance_check",
                    "clean": 1,
                    "flagged": 0,
                    "retracted_dropped": 1,
                    "blocked_sources": [
                        {"citation": "Retracted paper", "doi": "10.1234/retracted"}
                    ],
                }
            ]
        },
    )

    assert summary["retracted_dropped"] == 1
    assert summary["blocked_sources"][0]["doi"] == "10.1234/retracted"
