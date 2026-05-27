"""
Per-call LLM observability.

Every chat() / embed() call emits a JSONL record so we can answer:
  - "Which provider is slowest in practice?"
  - "How many tokens did this query burn?"
  - "Which model is being silently rate-limited and falling through to cascade?"
  - "What's the latency distribution at the FAST tier?"

Why JSONL on disk
-----------------
Zero infra. Append-only. Trivially `tail -f`-able for debugging. The eval
pipeline in Phase 6 reads this same file — no second data path to maintain.

Design rules
------------
- Logging is ALWAYS non-fatal — if disk is full or the file is locked, the
  LLM call must still succeed. Hence the broad except in `_write`.
- Logging happens INSIDE the provider layer, not the agent layer, so every
  call is captured regardless of which agent (or test, or CLI) made it.
- Records are flat — no nested objects — to make `jq` / pandas analysis easy.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.config import ModelTier, settings


@dataclass
class CallRecord:
    """One row of llm_calls.jsonl — kept flat on purpose."""

    call_id: str
    ts: str                            # ISO-8601 UTC
    op: str                            # "chat" | "embed"
    tier: str                          # ModelTier.value
    provider: str
    model: str
    duration_ms: int
    ok: bool
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cascade_step: int = 0              # 0 = primary, 1 = first fallback, etc.
    extra: dict[str, Any] = field(default_factory=dict)


def _write(record: CallRecord) -> None:
    if not settings.observability_enabled:
        return
    try:
        path = Path(settings.observability_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    except Exception:
        # Observability MUST NEVER break the LLM call path.
        pass


@contextmanager
def track(
    op: str,
    tier: ModelTier,
    provider: str,
    model: str,
    cascade_step: int = 0,
) -> Iterator[dict]:
    """
    Time a block and emit a CallRecord on exit.

    Usage:
        with track("chat", tier, provider, model) as ctx:
            resp = client.chat.completions.create(...)
            ctx["usage"] = resp.usage          # optional — captures token counts

    Always emits a record, even on exception. The yielded dict is the place
    callers stash anything they want included in the final record.
    """
    start = time.perf_counter()
    ctx: dict = {}
    err: str | None = None
    ok = True
    try:
        yield ctx
    except Exception as e:
        ok = False
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        dur_ms = int((time.perf_counter() - start) * 1000)
        usage = ctx.get("usage")
        in_tok = getattr(usage, "prompt_tokens", None) if usage else None
        out_tok = getattr(usage, "completion_tokens", None) if usage else None
        tot_tok = getattr(usage, "total_tokens", None) if usage else None

        rec = CallRecord(
            call_id=uuid.uuid4().hex[:12],
            ts=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            op=op,
            tier=tier.value if isinstance(tier, ModelTier) else str(tier),
            provider=provider,
            model=model,
            duration_ms=dur_ms,
            ok=ok,
            error=err,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
            cascade_step=cascade_step,
            extra=ctx.get("extra", {}),
        )
        _write(rec)


# ---------------------------------------------------------------------------
# Aggregations — consumed by the `researgent stats` CLI command.
# ---------------------------------------------------------------------------


def load_records(path: str | None = None, limit: int | None = None) -> list[dict]:
    """Read raw JSONL records. Tolerates partial / corrupt lines."""
    p = Path(path or settings.observability_log_path)
    if not p.exists():
        return []
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if limit:
        rows = rows[-limit:]
    return rows


def summarize(records: list[dict]) -> dict:
    """
    Aggregate stats for display:
      - per-tier and per-(tier,provider) counts, success rate, avg/p95 latency
      - total tokens used (input/output/total)
      - cascade fallback usage
    """
    if not records:
        return {
            "total_calls": 0,
            "by_tier": {},
            "by_tier_provider": {},
            "tokens": {"input": 0, "output": 0, "total": 0},
            "cascade_used": 0,
        }

    def _agg() -> dict:
        return {"count": 0, "ok": 0, "fail": 0, "lat_ms": [], "in_tok": 0, "out_tok": 0}

    by_tier: dict[str, dict] = {}
    by_pair: dict[str, dict] = {}
    cascade_used = 0
    in_total = out_total = tot_total = 0

    for r in records:
        tier = r.get("tier", "?")
        prov = r.get("provider", "?")
        pair = f"{tier}/{prov}"

        for bucket, key in ((by_tier, tier), (by_pair, pair)):
            slot = bucket.setdefault(key, _agg())
            slot["count"] += 1
            slot["ok" if r.get("ok") else "fail"] += 1
            if r.get("duration_ms") is not None:
                slot["lat_ms"].append(int(r["duration_ms"]))
            slot["in_tok"] += int(r.get("input_tokens") or 0)
            slot["out_tok"] += int(r.get("output_tokens") or 0)

        if int(r.get("cascade_step") or 0) > 0:
            cascade_used += 1
        in_total += int(r.get("input_tokens") or 0)
        out_total += int(r.get("output_tokens") or 0)
        tot_total += int(r.get("total_tokens") or 0)

    def _finalize(bucket: dict) -> dict:
        out = {}
        for k, v in bucket.items():
            lats = sorted(v["lat_ms"])
            avg = int(sum(lats) / len(lats)) if lats else 0
            p95 = lats[int(len(lats) * 0.95) - 1] if len(lats) >= 20 else (lats[-1] if lats else 0)
            out[k] = {
                "count": v["count"],
                "ok": v["ok"],
                "fail": v["fail"],
                "success_rate": round(v["ok"] / v["count"], 3) if v["count"] else 0,
                "avg_ms": avg,
                "p95_ms": p95,
                "in_tok": v["in_tok"],
                "out_tok": v["out_tok"],
            }
        return out

    return {
        "total_calls": len(records),
        "by_tier": _finalize(by_tier),
        "by_tier_provider": _finalize(by_pair),
        "tokens": {"input": in_total, "output": out_total, "total": tot_total},
        "cascade_used": cascade_used,
    }
