"""
Novelty Check node — Phase 24.

Compares the user's OWN experimental results against the retrieved papers
and decides whether the result is novel, an improvement over prior work,
or already known.

What it does
------------
1. Reads `user_results` from state — the GROUND TRUTH the user ingested.
   It is never generated, rewritten, or paraphrased here.

2. Hydrates the retrieved chunks and, for each paper, extracts:
     * the paper's APPROACH   — its title + abstract/text
     * the paper's METRICS     — the numbers it reports (BLEU-4 = 34.8, ...)

3. Scores the user's approach against each paper's approach (embedding
   cosine + content-word overlap) and the user's metrics against each
   paper's metrics (deterministic name-matched numeric comparison).

4. Emits a `novelty_report` and — critically — a FILTERED
   `chunk_refs_by_subq`. Papers whose approach closely matches the user's
   are dropped, so the Critic downstream only sees sources that passed
   BOTH Provenance and Novelty.

5. Sets `novelty_review_request` when the overall similarity reaches the
   review threshold, which is what makes the Human Review Gate pause.

Design constraints
------------------
* The NUMBERS are never judged by an LLM. 38.1 vs 34.8 is arithmetic.
  An LLM may later EXPLAIN the verdict; it never decides it.
* The embedder is optional. If it is unavailable, the node degrades to
  the deterministic content-overlap signal rather than failing the run.
"""

from __future__ import annotations

import time
from typing import Any

from src.agent.artifacts import HydratedChunk, hydrate
from src.agent.state import AgentState
from src.config import settings
from src.llm import embed
from src.novelty.extraction import extract_metrics
from src.novelty.scoring import (
    MatchDirection,
    compare_metric,
    content_overlap,
    decide_verdict,
    match_metrics,
    parse_numeric,
    score_approach_similarity,
)


# How much of a paper's text is used as its "approach" description.
_APPROACH_CHAR_CAP = 1200


def _flatten_match(m: Any) -> dict[str, Any]:
    """
    One canonical row shape for a matched metric.

    `MetricMatch` carries nested `user_metric` / `paper_metric` dicts, but
    the frontend (and the review-gate prompt) render a FLAT row. Emitting
    two different shapes for the same concept was a real integration bug:
    the report used the nested form while the review request used the flat
    one, so the UI showed `undefined` for every value.

    Flattening here means `novelty_report.matched_metrics` (which feeds
    both the SSE summary and `FinalEvent`) has the SAME shape as the
    per-paper rows in `novelty_review_request`.
    """
    user = m.user_metric or {}
    paper = m.paper_metric or {}
    return {
        "name": user.get("name", ""),
        "value": user.get("value"),
        "paper_name": paper.get("name", ""),
        "paper_value": paper.get("value"),
        "direction": m.direction.value,
        "delta": round(m.delta, 6),
        "relative_delta": round(m.relative_delta, 6),
        "source": (paper.get("source") or "")[:200],
        "matched": m.matched,
    }


def _paper_approach(chunk: HydratedChunk) -> str:
    """The text that describes what a paper DID."""
    parts = [chunk.doc_title or "", chunk.text or ""]
    text = "\n".join(p for p in parts if p)
    return text[:_APPROACH_CHAR_CAP].strip()


def _user_metrics(user_results: dict[str, Any]) -> list[dict[str, Any]]:
    """The user's metrics, in the shape the matcher expects."""
    out: list[dict[str, Any]] = []
    for m in (user_results.get("metrics") or [])[: settings.novelty_max_metrics_compared]:
        if not isinstance(m, dict):
            continue
        name = (m.get("name") or "").strip()
        if not name:
            continue
        # `numeric_value` was added by the ingestion normaliser; fall back
        # to parsing the raw value so a hand-built state dict still works.
        value = m.get("numeric_value")
        if value is None:
            value = parse_numeric(m.get("value"))
        out.append({"name": name, "value": value, "unit": m.get("unit", "")})
    return out


def _embed_texts(texts: list[str]) -> list[list[float]] | None:
    """
    Embed a batch, returning None if the embedder is unavailable.

    The Novelty Check must never be the reason a research run fails, so a
    provider outage degrades the approach score to the content-overlap
    signal instead of raising.
    """
    if not texts:
        return None
    try:
        vectors = embed(texts)
    except Exception:
        return None
    if not vectors or len(vectors) != len(texts):
        return None
    return vectors


def novelty_check(state: AgentState) -> dict[str, Any]:
    """
    Score the user's results against the retrieved papers.

    Returns a state update containing:
      * `chunk_refs_by_subq`   — filtered to the papers that passed
      * `novelty_dropped_refs` — the papers that did not
      * `novelty_report`       — the verdict + supporting detail
      * `novelty_review_request` — set when the review gate should fire
      * `trace`                — observability
    """
    t0 = time.perf_counter()

    refs_by_subq: dict[str, list[dict[str, str]]] = dict(
        state.get("chunk_refs_by_subq") or {}
    )
    user_results = state.get("user_results")

    # ---- Passthrough cases ------------------------------------------------
    if not settings.novelty_check_enabled or not user_results:
        return {
            "chunk_refs_by_subq": refs_by_subq,
            "novelty_report": None,
            "novelty_dropped_refs": {},
            "trace": [
                {
                    "node": "novelty_check",
                    "skipped": (
                        "disabled" if not settings.novelty_check_enabled
                        else "no_user_results"
                    ),
                }
            ],
        }

    user_approach = (
        user_results.get("approach")
        or user_results.get("methodology_notes")
        or user_results.get("summary")
        or ""
    ).strip()
    user_metric_list = _user_metrics(user_results)

    # ---- Hydrate the retrieved chunks -------------------------------------
    chunks_by_subq = hydrate(refs_by_subq)

    # Flatten to (sub_q, position_in_ref_list, ref, chunk) tuples. The
    # POSITION is carried explicitly rather than keying on `id(ref)` —
    # object identity is fragile (two equal refs would collide, and a
    # copied dict would not match). `zip` truncates to the shorter list,
    # so a ref with no hydrated chunk simply never enters `flat` and is
    # kept unscreened downstream.
    flat: list[tuple[str, int, dict[str, str], HydratedChunk]] = []
    for sub_q, chunks in chunks_by_subq.items():
        refs = refs_by_subq.get(sub_q) or []
        for pos, (ref, chunk) in enumerate(zip(refs, chunks)):
            flat.append((sub_q, pos, ref, chunk))

    if not flat:
        # Nothing retrieved — the user result stands on its own.
        report = _empty_report()
        return {
            "chunk_refs_by_subq": refs_by_subq,
            "novelty_report": report,
            "novelty_dropped_refs": {},
            "trace": [
                {
                    "node": "novelty_check",
                    "duration_ms": int((time.perf_counter() - t0) * 1000),
                    "compared_papers": 0,
                    "verdict": report["verdict"],
                    "note": "no sources to compare against",
                }
            ],
        }

    # ---- Approach vectors -------------------------------------------------
    max_papers = settings.novelty_max_papers_compared
    compared = flat[:max_papers]

    paper_approaches = [_paper_approach(c) for _, _, _, c in compared]
    texts_to_embed = ([user_approach] if user_approach else []) + paper_approaches
    vectors = _embed_texts(texts_to_embed) if texts_to_embed else None

    user_vec = vectors[0] if vectors and user_approach else None

    # ---- Per-paper comparison ---------------------------------------------
    approach_sims: list[float] = []
    all_matches = []
    paper_records: list[dict[str, Any]] = []

    for idx, (sub_q, pos, ref, chunk) in enumerate(compared):
        p_approach = paper_approaches[idx]
        p_vec = vectors[idx + 1] if vectors and user_approach else None

        if user_approach and p_approach:
            if user_vec is not None and p_vec is not None:
                approach_sim = score_approach_similarity(
                    user_approach, p_approach, vectors=(user_vec, p_vec)
                )
            else:
                approach_sim = score_approach_similarity(user_approach, p_approach)
        else:
            approach_sim = 0.0

        p_metrics = extract_metrics(chunk.text or "")
        matches = match_metrics(
            user_metrics=user_metric_list,
            paper_metrics=p_metrics,
        )
        approach_sims.append(approach_sim)
        all_matches.extend(matches)

        paper_records.append({
            "ref": ref,
            "sub_q": sub_q,
            "pos": pos,
            "title": chunk.doc_title or chunk.citation,
            "citation": chunk.citation,
            "arxiv_id": chunk.arxiv_id or "",
            "approach_similarity": round(approach_sim, 4),
            "metrics": [_flatten_match(m) for m in matches if m.matched],
        })

    # ---- Aggregate --------------------------------------------------------
    # The BEST (lowest) approach similarity across papers is what matters:
    # novelty is lost to the single closest prior work, not diluted by the
    # average across unrelated papers.
    best_approach_sim = min(approach_sims) if approach_sims else 0.0

    decision = decide_verdict(
        approach_similarity=best_approach_sim,
        result_matches=all_matches,
        approach_threshold=settings.novelty_approach_threshold,
        review_threshold=settings.novelty_review_threshold,
    )

    # ---- The novelty screen ------------------------------------------------
    # A paper is dropped when its approach matches the user's closely AND
    # it reports a metric the user also reported — that combination is the
    # "this exact thing was already done" case.
    kept_refs: dict[str, list[dict[str, str]]] = {}
    dropped_refs: dict[str, list[dict[str, str]]] = {}

    # Key the screened papers by (sub_q, position) so each ref maps to
    # exactly one comparison, with no reliance on object identity.
    screened: dict[tuple[str, int], dict[str, Any]] = {
        (rec["sub_q"], rec["pos"]): rec for rec in paper_records
    }

    for sub_q, refs in refs_by_subq.items():
        keep: list[dict[str, str]] = []
        drop: list[dict[str, str]] = []
        for pos, ref in enumerate(refs or []):
            rec = screened.get((sub_q, pos))
            if rec is None:
                # Beyond the comparison cap, or not hydrated — not
                # screened, so keep it.
                keep.append(ref)
                continue
            close_approach = (
                rec["approach_similarity"] >= settings.novelty_approach_threshold
            )
            shares_metric = bool(rec["metrics"])
            if close_approach and shares_metric:
                drop.append(ref)
            else:
                keep.append(ref)
        kept_refs[sub_q] = keep
        if drop:
            dropped_refs[sub_q] = drop

    # ---- Report ------------------------------------------------------------
    # ONE canonical flat shape for matched metrics everywhere.
    matched_flat = [_flatten_match(m) for m in all_matches if m.matched]

    report = {
        **decision.to_dict(),
        # Override the nested form that `decision.to_dict()` emits so the
        # report, the SSE summary, and the review request all agree.
        "matched_metrics": matched_flat,
        "compared_papers": len(compared),
        "papers": paper_records,
        "user_approach": user_approach[:500],
        "user_metrics": [
            {"name": m["name"], "value": m["value"], "unit": m.get("unit", "")}
            for m in user_metric_list
        ],
    }

    review_request = None
    if decision.requires_review:
        # Surface the papers that triggered the gate so the human sees WHAT
        # to look at, not just a number.
        triggering = [
            rec
            for rec in paper_records
            if rec["approach_similarity"] >= settings.novelty_approach_threshold
            or rec["metrics"]
        ]
        review_request = {
            "similarity": round(decision.overall_similarity, 4),
            "verdict": decision.verdict.value,
            "rationale": decision.rationale,
            "approach_similarity": round(decision.approach_similarity, 4),
            "result_similarity": round(decision.result_similarity, 4),
            "threshold": settings.novelty_review_threshold,
            "papers": [
                {
                    "title": rec["title"],
                    "citation": rec["citation"],
                    "arxiv_id": rec["arxiv_id"],
                    "approach_similarity": rec["approach_similarity"],
                    "metrics": rec["metrics"],
                }
                for rec in triggering[:5]
            ],
        }

    dur_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "chunk_refs_by_subq": kept_refs,
        "novelty_dropped_refs": dropped_refs,
        "novelty_report": report,
        "novelty_review_request": review_request,
        "trace": [
            {
                "node": "novelty_check",
                "duration_ms": dur_ms,
                "compared_papers": len(compared),
                "verdict": decision.verdict.value,
                "is_novel": decision.is_novel,
                "overall_similarity": round(decision.overall_similarity, 4),
                "approach_similarity": round(decision.approach_similarity, 4),
                "result_similarity": round(decision.result_similarity, 4),
                "requires_review": decision.requires_review,
                "dropped_sources": sum(len(v) for v in dropped_refs.values()),
            }
        ],
    }


def _empty_report() -> dict[str, Any]:
    """The report shape when there was nothing to compare against."""
    return {
        "verdict": "inconclusive",
        "is_novel": False,
        "overall_similarity": 0.0,
        "approach_similarity": 0.0,
        "result_similarity": 0.0,
        "requires_review": False,
        "matched_metrics": [],
        "rationale": "No prior work was retrieved to compare against.",
        "compared_papers": 0,
        "papers": [],
        "user_approach": "",
        "user_metrics": [],
    }
