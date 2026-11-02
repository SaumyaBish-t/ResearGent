"""
Tests for the Vault attaching the Novelty Report as metadata (Phase 24).

The spec: "Vault: Originality report attached as metadata (plus novelty
metadata)."

The Originality Report is already written into the note. The Novelty
Report — the verdict on whether the user's own results are novel against
the prior work — must be attached ALONGSIDE it, so a reader of the vault
note can see both:

  * Originality: is this DRAFT text copied from somewhere?
  * Novelty:     is this RESULT something new?

They answer different questions and both belong on the note.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_vault_novelty_metadata.py -v
"""

import sys

sys.path.insert(0, ".")

from src.agent import vault_writer


NOVELTY_REPORT = {
    "verdict": "improvement",
    "is_novel": True,
    "overall_similarity": 0.62,
    "approach_similarity": 0.58,
    "result_similarity": 0.71,
    "requires_review": False,
    "rationale": "The result improves on every comparable metric.",
    "compared_papers": 4,
    "matched_metrics": [
        {
            "name": "BLEU-4",
            "value": 38.1,
            "paper_name": "BLEU-4",
            "paper_value": 34.8,
            "direction": "better",
            "delta": 3.3,
        }
    ],
    "papers": [],
    "user_approach": "Sparse attention with learned pruning masks.",
    "user_metrics": [{"name": "BLEU-4", "value": 38.1, "unit": ""}],
}

ORIGINALITY_REPORT = {
    "score": 0.12,
    "flags": [],
    "sentences_checked": 14,
}


# ---------------------------------------------------------------------------
# The markdown note carries the Novelty Report.
# ---------------------------------------------------------------------------


def _write_note(tmp_path, **kwargs):
    """Call the file-backed vault writer with a minimal valid payload."""
    defaults = {
        "vault_path": tmp_path,
        "output_subfolder": "research_data",
        "question": "How does sparse attention work?",
        "answer": "A" * 120,
        "sources": {"S1": {"citation": "Paper p.1", "title": "Paper"}},
        "sub_questions": ["How does sparse attention work?"],
        "is_complex": False,
        "confidence": "high",
        "rewrite_attempts": 0,
        "web_used": False,
        "papers_used": True,
        "reflection_attempts": 0,
        "run_id": "run-1",
    }
    defaults.update(kwargs)
    return vault_writer.write_run_to_vault(**defaults)


def test_note_includes_the_novelty_report(tmp_path):
    out = _write_note(
        tmp_path,
        novelty_report=NOVELTY_REPORT,
        originality_report=ORIGINALITY_REPORT,
    )
    text = out.read_text(encoding="utf-8")
    assert "Novelty Report" in text


def test_note_records_the_novelty_verdict(tmp_path):
    out = _write_note(tmp_path, novelty_report=NOVELTY_REPORT)
    text = out.read_text(encoding="utf-8")
    assert "improvement" in text


def test_note_records_the_novelty_similarity(tmp_path):
    out = _write_note(tmp_path, novelty_report=NOVELTY_REPORT)
    text = out.read_text(encoding="utf-8")
    assert "0.62" in text


def test_note_records_the_matched_metrics(tmp_path):
    out = _write_note(tmp_path, novelty_report=NOVELTY_REPORT)
    text = out.read_text(encoding="utf-8")
    assert "BLEU-4" in text
    assert "34.8" in text
    assert "38.1" in text


def test_note_records_the_user_approach(tmp_path):
    out = _write_note(tmp_path, novelty_report=NOVELTY_REPORT)
    text = out.read_text(encoding="utf-8")
    assert "sparse attention" in text.lower()


def test_note_keeps_the_originality_report(tmp_path):
    """Both reports must be present — they answer different questions."""
    out = _write_note(
        tmp_path,
        novelty_report=NOVELTY_REPORT,
        originality_report=ORIGINALITY_REPORT,
    )
    text = out.read_text(encoding="utf-8")
    assert "Originality Report" in text
    assert "Novelty Report" in text


def test_note_without_a_novelty_report_has_no_novelty_section(tmp_path):
    out = _write_note(tmp_path, originality_report=ORIGINALITY_REPORT)
    text = out.read_text(encoding="utf-8")
    assert "Novelty Report" not in text


def test_note_with_a_not_novel_verdict_records_it(tmp_path):
    """A negative verdict must be visible too — it is the important one."""
    report = {**NOVELTY_REPORT, "verdict": "not_novel", "is_novel": False}
    out = _write_note(tmp_path, novelty_report=report)
    text = out.read_text(encoding="utf-8")
    assert "not_novel" in text


# ---------------------------------------------------------------------------
# The DB-backed writer carries it in the note body too.
# ---------------------------------------------------------------------------


def test_db_writer_signature_accepts_a_novelty_report():
    """The Postgres vault writer must accept the novelty report."""
    import inspect

    sig = inspect.signature(vault_writer.write_run_to_db_vault)
    assert "novelty_report" in sig.parameters
    assert sig.parameters["novelty_report"].default is None


def test_file_writer_signature_accepts_a_novelty_report():
    import inspect

    sig = inspect.signature(vault_writer.write_run_to_vault)
    assert "novelty_report" in sig.parameters
    assert sig.parameters["novelty_report"].default is None


def test_save_signature_accepts_a_novelty_report():
    """`auto_save_run` must thread the novelty report through."""
    import inspect

    from src.agent import save

    sig = inspect.signature(save.auto_save_run)
    assert "novelty_report" in sig.parameters
    assert sig.parameters["novelty_report"].default is None
