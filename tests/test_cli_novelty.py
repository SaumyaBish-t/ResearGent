"""
Tests for the Phase 24 CLI commands.

Two commands:

  * `results validate` — check a user-results JSON file WITHOUT running
    the agent. The user should be able to confirm their payload is
    well-formed before spending money on a research run.

  * `novelty review`  — approve or reject a run paused at the Human
    Review Gate.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_cli_novelty.py -v
"""

import json
import sys

sys.path.insert(0, ".")

from typer.testing import CliRunner

from src.main import app


runner = CliRunner()

VALID_RESULTS = {
    "summary": "Sparse attention with learned pruning masks.",
    "approach": "Sparse attention with learned pruning masks.",
    "dataset": "WMT14",
    "metrics": [{"name": "BLEU-4", "value": 38.1}],
}


# ---------------------------------------------------------------------------
# The commands exist.
# ---------------------------------------------------------------------------


def test_results_validate_command_exists():
    result = runner.invoke(app, ["results", "validate", "--help"])
    assert result.exit_code == 0, result.output


def test_novelty_review_command_exists():
    result = runner.invoke(app, ["novelty", "review", "--help"])
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# results validate
# ---------------------------------------------------------------------------


def test_results_validate_accepts_a_valid_file(tmp_path):
    f = tmp_path / "r.json"
    f.write_text(json.dumps(VALID_RESULTS), encoding="utf-8")

    result = runner.invoke(app, ["results", "validate", str(f)])

    assert result.exit_code == 0, result.output
    assert "BLEU-4" in result.output


def test_results_validate_shows_the_derived_queries(tmp_path):
    f = tmp_path / "r.json"
    f.write_text(json.dumps(VALID_RESULTS), encoding="utf-8")

    result = runner.invoke(app, ["results", "validate", str(f)])

    assert "sparse attention" in result.output.lower()


def test_results_validate_rejects_a_bad_file(tmp_path):
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"summary": "s"}), encoding="utf-8")

    result = runner.invoke(app, ["results", "validate", str(f)])

    assert result.exit_code != 0


def test_results_validate_rejects_a_missing_file(tmp_path):
    result = runner.invoke(app, ["results", "validate", str(tmp_path / "nope.json")])
    assert result.exit_code != 0


def test_results_validate_rejects_malformed_json(tmp_path):
    f = tmp_path / "r.json"
    f.write_text("{not json", encoding="utf-8")

    result = runner.invoke(app, ["results", "validate", str(f)])

    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# novelty review
# ---------------------------------------------------------------------------


def test_novelty_review_requires_a_thread_id():
    result = runner.invoke(app, ["novelty", "review", "--approve"])
    assert result.exit_code != 0


def test_novelty_review_rejects_an_unknown_thread():
    result = runner.invoke(
        app, ["novelty", "review", "--thread-id", "no-such-thread", "--approve"]
    )
    # Fails because there is no paused run — NOT because the flags are wrong.
    assert result.exit_code != 0
    assert "no-such-thread" in result.output or "thread" in result.output.lower()
