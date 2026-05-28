"""
FastAPI app — streaming /api/research endpoint + static web UI at /.

Endpoints
---------
  GET  /                       — single-file web UI (see src/api/web/index.html)
  GET  /api/research?q=...&k=8 — SSE stream: one event per agent node
  GET  /api/status             — provider routing + observability snapshot
  GET  /api/stats              — aggregated LLM call stats
  POST /api/ingest             — (Phase 6b nicety) trigger an ingest from the UI

Why SSE not WebSocket
---------------------
The agent is server-push only. There's no need for bidirectional comms.
SSE is one-line easier to consume from the browser (EventSource API),
auto-reconnects, and survives reverse-proxy quirks better than WS.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from src.agent.stream import stream_agent
from src.llm import list_status
from src.llm.observability import load_records, summarize


WEB_DIR = Path(__file__).parent / "web"


def create_app() -> FastAPI:
    app = FastAPI(
        title="ResearGent",
        description="Agentic research engine — Corrective RAG + Self-Reflection",
        version="0.10.0",
    )

    # ---- UI ----
    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    # ---- Streaming agent endpoint ----
    @app.get("/api/research")
    async def research(
        q: str = Query(..., description="The research question"),
        k: int = Query(8, description="Total chunks budget across all sub-questions"),
    ):
        """
        Server-Sent Events stream. Emits JSON-encoded events:
          - run_started     {run_id, question, ts}
          - node_complete   {node, summary, ts}
          - final           {answer, sources, ...}
          - error           {error, ts}
        """
        async def event_gen():
            loop = asyncio.get_running_loop()
            gen = stream_agent(q, k=k)
            while True:
                try:
                    event = await loop.run_in_executor(None, _next_or_none, gen)
                except Exception as e:
                    yield {"event": "error", "data": json.dumps({"error": str(e)})}
                    return
                if event is None:
                    return
                yield {
                    "event": event.get("type", "message"),
                    "data": json.dumps(event, default=str),
                }

        return EventSourceResponse(event_gen())

    # ---- Diagnostic endpoints ----
    @app.get("/api/status")
    async def status() -> JSONResponse:
        return JSONResponse(list_status())

    @app.get("/api/stats")
    async def stats(last: int = Query(0)) -> JSONResponse:
        recs = load_records(limit=last or None)
        return JSONResponse(summarize(recs))

    return app


def _next_or_none(gen) -> Any:
    """Pull the next item from a sync generator; return None on StopIteration."""
    try:
        return next(gen)
    except StopIteration:
        return None
