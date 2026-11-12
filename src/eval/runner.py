"""
Eval runner — load a YAML suite, run each question through the agent,
score, and write to data/eval/runs.jsonl.

Why YAML and not Python
-----------------------
Eval suites are EDITED far more often than the runner itself. A YAML file
is something a non-coder can extend (add new questions, add tags). The
Python runner just executes whatever the YAML says.

Suite format
------------
    name: phase5-smoke
    description: Sanity tests after Phase 5 self-reflection
    queries:
      - id: crag-basic
        question: "What is corrective RAG?"
        tags: [definition, single-paper]
      - id: crag-vs-selfrag
        question: "Compare CRAG and Self-RAG retrieval handling"
        tags: [comparison, multi-paper, hybrid-required]
        k: 8                        # optional per-query override

Output
------
One JSONL row per query in data/eval/runs.jsonl. Each row includes:
  - suite_name, query_id, question
  - answer, citations, sub_questions, trace summary
  - scores: faithfulness, answer_relevancy, context_precision, overall
  - agent metadata: confidence, rewrite_attempts, web_used,
    reflection_attempts, ran_at_iso, latency_ms

Aggregations are then trivial:
    jq -s 'group_by(.suite_name) | map({suite: .[0].suite_name,
            mean_faith: ([.[].scores.faithfulness] | add / length)})' \\
      data/eval/runs.jsonl
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.agent import run_agent
from src.eval.metrics import EvalScores, score_run

EVAL_DIR = Path("data") / "eval"
RESULTS_FILE = EVAL_DIR / "runs.jsonl"
RETRIEVAL_RESULTS_FILE = EVAL_DIR / "retrieval_runs.jsonl"


@dataclass
class EvalQuery:
    id: str
    question: str
    tags: list[str] = field(default_factory=list)
    k: int = 8
    domain_scope: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    retrieval_k: int = 5
    category: str = ""
    reference_points: list[str] = field(default_factory=list)


@dataclass
class EvalSuite:
    name: str
    description: str = ""
    retrieval_mode: str = "hybrid"
    queries: list[EvalQuery] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "EvalSuite":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        queries_raw = data.get("queries") or []
        queries: list[EvalQuery] = []
        for q in queries_raw:
            domain = q.get("domain") or q.get("domain_scope") or []
            if isinstance(domain, str):
                domain = [domain]
            queries.append(
                EvalQuery(
                    id=str(q.get("id") or uuid.uuid4().hex[:8]),
                    question=str(q["question"]),
                    tags=list(q.get("tags") or []),
                    k=int(q.get("k") or 8),
                    domain_scope=[str(d) for d in domain],
                    expected_sources=[str(s) for s in (q.get("expected_sources") or [])],
                    retrieval_k=int(q.get("retrieval_k") or 5),
                    category=str(q.get("category") or ""),
                    reference_points=[str(point) for point in (q.get("reference_points") or [])],
                )
            )
        return cls(
            name=str(data.get("name") or path.stem),
            description=str(data.get("description") or ""),
            retrieval_mode=str(data.get("retrieval_mode") or "hybrid").lower(),
            queries=queries,
        )


@dataclass
class EvalResult:
    suite_name: str
    query_id: str
    question: str
    answer: str
    n_sources: int
    scores: dict[str, Any]
    retrieval: dict[str, Any]
    expected_sources: list[str]
    reference_points: list[str]
    domain_scope: list[str]
    category: str
    sub_questions: list[str]
    is_complex: bool
    confidence: str
    rewrite_attempts: int
    web_used: bool
    reflection_attempts: int
    ran_at_iso: str
    latency_ms: int


def _write_jsonl(rows: list[EvalResult]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def _score_retrieval(query: EvalQuery, mode: str = "hybrid") -> dict[str, Any]:
    """Measure local top-k retrieval against manually labeled source filenames."""
    if not query.expected_sources:
        return {"labeled": False}

    try:
        if mode == "bm25":
            from src.retrieval import bm25

            hits = bm25.search(query.question, k=query.retrieval_k)
            retrieved = [str(hit.metadata.get("source_file") or "") for hit in hits]
        else:
            from src.retrieval import hybrid_retrieve

            hits = hybrid_retrieve(
                query.question,
                k=query.retrieval_k,
                domains=query.domain_scope or None,
            )
            retrieved = [str(hit.source_file or "") for hit in hits]
    except Exception as exc:
        return {
            "labeled": True,
            "mode": mode,
            "error": f"{type(exc).__name__}: {exc}",
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "reciprocal_rank": 0.0,
            "hits": 0,
            "expected_count": len(query.expected_sources),
            "retrieved_sources": [],
        }

    expected = {source.casefold() for source in query.expected_sources}
    matched_ranks = [
        rank
        for rank, source in enumerate(retrieved, start=1)
        if any(gold in source.casefold() for gold in expected)
    ]
    matched_gold = {
        gold for gold in expected
        if any(gold in source.casefold() for source in retrieved)
    }
    matched_sources = sorted({retrieved[rank - 1] for rank in matched_ranks})
    k = max(1, query.retrieval_k)
    hits_n = len(matched_ranks)
    return {
        "labeled": True,
        "mode": mode,
        "expected_sources": sorted(expected),
        "precision_at_k": round(hits_n / k, 4),
        "recall_at_k": round(len(matched_gold) / max(len(expected), 1), 4),
        "reciprocal_rank": round(1.0 / matched_ranks[0], 4) if matched_ranks else 0.0,
        "hits": hits_n,
        "expected_count": len(expected),
        "retrieved_sources": retrieved,
        "matched_sources": matched_sources,
    }


def run_suite(
    suite: EvalSuite,
    *,
    on_query_complete=None,
    skip_metrics: bool = False,
) -> list[EvalResult]:
    """
    Run every query in the suite, score each, persist, return results.

    `on_query_complete(idx, total, result)` — optional callback for live
    progress in the CLI.
    `skip_metrics=True` — runs the agent but doesn't compute RAGAS scores.
    Useful for quick smoke runs or when you just want to populate the
    observability log.
    """
    results: list[EvalResult] = []
    total = len(suite.queries)

    for i, q in enumerate(suite.queries, start=1):
        retrieval = _score_retrieval(q, suite.retrieval_mode)
        t0 = time.perf_counter()
        agent_result = run_agent(
            q.question,
            k=q.k,
            use_checkpointer=False,
            domain_scope=q.domain_scope or None,
        )
        agent_ms = int((time.perf_counter() - t0) * 1000)

        if skip_metrics:
            scores = EvalScores()
        else:
            chunks = list(agent_result.sources.values()) if agent_result.sources else []
            scores = score_run(
                question=q.question,
                answer=agent_result.answer,
                cited_chunks=chunks,
            )

        result = EvalResult(
            suite_name=suite.name,
            query_id=q.id,
            question=q.question,
            answer=agent_result.answer,
            n_sources=len(agent_result.sources or {}),
            retrieval=retrieval,
            expected_sources=list(q.expected_sources),
            reference_points=list(q.reference_points),
            domain_scope=list(q.domain_scope),
            category=q.category,
            scores={
                "faithfulness": round(scores.faithfulness, 3),
                "answer_relevancy": round(scores.answer_relevancy, 3),
                "context_precision": round(scores.context_precision, 3),
                "overall": round(scores.overall, 3),
                "n_claims": scores.n_claims,
                "n_supported_claims": scores.n_supported_claims,
                "n_chunks_evaluated": scores.n_chunks_evaluated,
                "n_useful_chunks": scores.n_useful_chunks,
            },
            sub_questions=agent_result.sub_questions,
            is_complex=agent_result.is_complex,
            confidence=agent_result.confidence,
            rewrite_attempts=agent_result.rewrite_attempts,
            web_used=agent_result.web_used,
            reflection_attempts=agent_result.reflection_attempts,
            ran_at_iso=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            latency_ms=agent_ms,
        )
        results.append(result)

        if on_query_complete:
            on_query_complete(i, total, result)

    _write_jsonl(results)
    return results


def run_retrieval_suite(suite: EvalSuite) -> list[dict[str, Any]]:
    """Run only labeled local retrieval; no generation or LLM judge calls."""
    rows: list[dict[str, Any]] = []
    for q in suite.queries:
        row = {
            "suite_name": suite.name,
            "query_id": q.id,
            "question": q.question,
            "domain_scope": q.domain_scope,
            "category": q.category,
            "expected_sources": q.expected_sources,
            "reference_points": q.reference_points,
            "retrieval": _score_retrieval(q, suite.retrieval_mode),
            "ran_at_iso": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        }
        rows.append(row)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with RETRIEVAL_RESULTS_FILE.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def summarize_retrieval_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize top-k retrieval against the benchmark's labeled source set."""
    labeled = [row["retrieval"] for row in rows if row.get("retrieval", {}).get("labeled")]
    if not labeled:
        return {"n": 0}
    n = len(labeled)
    return {
        "n": n,
        "mean_precision_at_k": round(sum(r.get("precision_at_k", 0) for r in labeled) / n, 3),
        "mean_recall_at_k": round(sum(r.get("recall_at_k", 0) for r in labeled) / n, 3),
        "mean_reciprocal_rank": round(sum(r.get("reciprocal_rank", 0) for r in labeled) / n, 3),
        "no_gold_hit": sum(1 for r in labeled if r.get("hits", 0) == 0),
        "retrieval_errors": sum(1 for r in labeled if r.get("error")),
    }


def summarize_results(results: list[EvalResult]) -> dict[str, Any]:
    """Aggregate scores across a result set. Used by CLI display + JSON output."""
    if not results:
        return {"n": 0}
    n = len(results)
    f = [r.scores["faithfulness"] for r in results]
    a = [r.scores["answer_relevancy"] for r in results]
    p = [r.scores["context_precision"] for r in results]
    o = [r.scores["overall"] for r in results]

    def _stats(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(sum(xs) / len(xs), 3),
            "min": round(min(xs), 3),
            "max": round(max(xs), 3),
        }

    return {
        "n": n,
        "faithfulness": _stats(f),
        "answer_relevancy": _stats(a),
        "context_precision": _stats(p),
        "overall": _stats(o),
        "mean_latency_ms": int(sum(r.latency_ms for r in results) / n),
        "retrieval": {
            "n_labeled": sum(1 for r in results if r.retrieval.get("labeled")),
            "mean_precision_at_k": round(
                sum(r.retrieval.get("precision_at_k", 0.0) for r in results if r.retrieval.get("labeled"))
                / max(sum(1 for r in results if r.retrieval.get("labeled")), 1), 3
            ),
            "mean_recall_at_k": round(
                sum(r.retrieval.get("recall_at_k", 0.0) for r in results if r.retrieval.get("labeled"))
                / max(sum(1 for r in results if r.retrieval.get("labeled")), 1), 3
            ),
            "mean_reciprocal_rank": round(
                sum(r.retrieval.get("reciprocal_rank", 0.0) for r in results if r.retrieval.get("labeled"))
                / max(sum(1 for r in results if r.retrieval.get("labeled")), 1), 3
            ),
        },
        "n_reflections_triggered": sum(1 for r in results if r.reflection_attempts > 0),
        "n_web_used": sum(1 for r in results if r.web_used),
    }
