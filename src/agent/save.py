"""
Auto-save policy + writer for research runs.

Centralized so both the CLI (`research --save-to-vault`) and the streaming
API (`/api/research`) follow IDENTICAL save semantics. Anywhere else that
needs to persist a run uses these helpers, never the writer directly.

Quality gates (in `should_auto_save`)
-------------------------------------
  1. settings.auto_save_to_notes must be True (master kill-switch)
  2. error must not be "no_sources_used_llm_priors"  — pure LLM-priors
     answers have no citations and would inject unverified claims into
     the brain. Saving them risks compounding hallucinations on future
     queries that retrieve them.
  3. confidence must meet settings.auto_save_min_confidence — defaults
     to "high" to keep the brain clean. Lower at your own risk.

Why gate auto-save
------------------
The compounding-knowledge loop is real and that's the point. But it also
means an error saved today is a cited "source" tomorrow. Conservative
default keeps the loop's compounding behavior POSITIVE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent.vault_writer import write_run_to_vault
from src.config import settings


_CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def should_auto_save(
    *,
    confidence: str,
    error: str | None,
) -> bool:
    """
    Decide whether a finished agent run is eligible for auto-save.

    Pure function — takes the agent's final state fields, returns a bool.
    Settings are read from the module-level `settings` singleton.
    """
    if not settings.auto_save_to_notes:
        return False

    # Never save pure-LLM-priors answers — they have no sources and would
    # poison the brain on future retrievals.
    if error == "no_sources_used_llm_priors":
        return False

    threshold = (settings.auto_save_min_confidence or "high").lower()
    if threshold == "always":
        return True

    got = _CONF_RANK.get((confidence or "").lower(), -1)
    want = _CONF_RANK.get(threshold, 2)  # default to "high" if misspelled
    return got >= want


def auto_save_run(
    *,
    question: str,
    answer: str,
    sources: dict,
    sub_questions: list[str],
    is_complex: bool,
    confidence: str,
    rewrite_attempts: int,
    web_used: bool,
    papers_used: bool,
    reflection_attempts: int,
    run_id: str,
    error: str | None = None,
) -> Path | None:
    """
    Apply gating + write the run to the notes folder.

    Returns the written note's Path on success, or None when:
      - gating rejected the run
      - no notes folder configured (resolve_notes_folder() returned None)
      - the write itself failed (logged but not raised — auto-save must
        never break the agent run's primary outcome)
    """
    if not should_auto_save(confidence=confidence, error=error):
        return None

    # Refuse to write a blank/near-blank note to the vault. The AutoGen
    # run that triggered this guard had verdict=high (threshold passed,
    # auto-save fired) but the generator returned an empty draft after
    # 139s — the resulting .md was empty. An empty note in Obsidian is
    # worse than no note: it shows up in search results and crowds the
    # daily folder. ~40 chars is the floor for "actually wrote something"
    # (header + one short sentence). Below that, treat it as a failed
    # generation and skip the save.
    if not answer or len(answer.strip()) < 40:
        return None

    notes_folder = settings.resolve_notes_folder()
    if not notes_folder:
        return None

    try:
        return write_run_to_vault(
            vault_path=notes_folder,
            output_subfolder=settings.obsidian_output_folder,
            question=question,
            answer=answer,
            sources=sources,
            sub_questions=sub_questions,
            is_complex=is_complex,
            confidence=confidence,
            rewrite_attempts=rewrite_attempts,
            web_used=web_used,
            papers_used=papers_used,
            reflection_attempts=reflection_attempts,
            run_id=run_id,
        )
    except Exception:
        # Auto-save failure must not break the run. Caller's responsibility
        # to log if desired.
        return None
