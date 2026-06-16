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

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.sessions import SessionMiddleware

from src.agent.stream import stream_agent
from src.auth.deps import current_user
from src.auth.routes import router as auth_router
from src.auth.users import User
from src.billing import quota, threads
from src.billing.routes import router as billing_router
from src.config import settings
from src.llm import list_status
from src.llm.observability import load_records, summarize


WEB_DIR = Path(__file__).parent / "web"


def create_app() -> FastAPI:
    app = FastAPI(
        title="ResearGent",
        description="Agentic research engine — Corrective RAG + Self-Reflection",
        version="0.10.0",
    )

    # ---- CORS ----
    # The bundled single-file UI is same-origin so it never needed CORS. The
    # new Next.js / React-Three-Fiber frontend runs on a DIFFERENT origin in
    # dev (:3000 → :8000), so the browser pre-flights and blocks the SSE
    # connection without these headers. Origins are configurable via
    # CORS_ALLOW_ORIGINS in .env; defaults cover the Next dev server.
    _origins = settings.cors_origins_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        # When allowing all origins, credentials must be off per the CORS spec.
        allow_credentials=_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- SessionMiddleware (for OAuth state only) ----
    # authlib's `authorize_redirect` stashes a CSRF state + nonce in
    # request.session, which Starlette implements as a separate signed cookie.
    # This is INDEPENDENT of our app's auth — our session JWT lives in its own
    # HttpOnly cookie set by `set_session_cookie`. Both can coexist.
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        same_site="lax",
        https_only=settings.cookie_secure,
        # Short-lived: only needed for the 30 seconds between /auth/google and
        # /auth/callback. Long TTL would just expand the CSRF replay window.
        max_age=600,
    )

    # ---- Auth routes (/auth/google, /auth/callback, /auth/me, /auth/logout) ----
    app.include_router(auth_router)

    # ---- Billing + history routes (/api/usage, /api/threads, /billing/*) ----
    app.include_router(billing_router)

    # ---- UI ----
    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    # ---- Streaming agent endpoint ----
    # Auth + quota + thread persistence are wired here (not via middleware) so
    # the SSE stream owns the persistence side-effect when the run finishes.
    @app.get("/api/research")
    async def research(
        q: str = Query(..., description="The research question"),
        k: int = Query(8, description="Total chunks budget across all sub-questions"),
        thread_id: str | None = Query(
            None,
            description="Continue an existing thread (omit to start a new one).",
        ),
        user: User = Depends(current_user),
    ):
        """
        Server-Sent Events stream. Emits JSON-encoded events:
          - run_started     {run_id, question, ts, thread_id, turn_index}
          - node_complete   {node, summary, ts}
          - final           {answer, sources, ...}
          - error           {error, ts}

        Auth: required (cookie). Quotas:
          - new thread:      blocked by `thread_cap`  → HTTP 402
          - follow-up:       blocked by `turn_cap`    → HTTP 402
          - admins + active subscribers bypass both.
        """
        # ---- Quota + thread resolution (BEFORE opening the SSE stream) ----
        if thread_id:
            thread = threads.get_thread(thread_id=thread_id, user_id=user.id)
            if not thread:
                raise HTTPException(404, "Thread not found.")
            decision = quota.check_can_add_turn(user=user, thread_id=thread.id)
        else:
            thread = None
            decision = quota.check_can_create_thread(user)

        if not decision.allowed:
            # 402 Payment Required — the frontend reads the JSON body to know
            # which kind of paywall to show (out of threads vs out of turns).
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "quota_exceeded",
                    **decision.to_dict(),
                },
            )

        # Reserve the thread + turn_index NOW so the SSE stream can echo them
        # back in `run_started`. The turn ROW itself is inserted only after
        # the run completes (we need the final answer to populate it).
        if thread is None:
            thread = threads.create_thread(user_id=user.id, title=q)
        turn_index = threads.next_turn_index(thread_id=thread.id)

        # If this is a follow-up, prepend prior Q+A as context. The new
        # question's retrieval + critic loop still runs against the new Q only.
        prior_turns = threads.list_turns(thread_id=thread.id) if turn_index > 0 else []
        prefix = threads.build_context_prefix(prior_turns) if prior_turns else ""
        effective_q = (
            f"{prefix}\n\n[Current question]\n{q}".strip() if prefix else q
        )

        async def event_gen():
            loop = asyncio.get_running_loop()
            gen = stream_agent(effective_q, k=k)
            final_event: dict | None = None
            while True:
                try:
                    event = await loop.run_in_executor(None, _next_or_none, gen)
                except Exception as e:
                    yield {"event": "error", "data": json.dumps({"error": str(e)})}
                    return
                if event is None:
                    break

                # Augment run_started + final with thread/turn metadata so the
                # UI can update the sidebar without a separate fetch.
                if event.get("type") == "run_started":
                    event["thread_id"] = thread.id
                    event["turn_index"] = turn_index
                    event["question"] = q   # echo the ORIGINAL question, not the prefixed one
                elif event.get("type") == "final":
                    final_event = event
                    event["thread_id"] = thread.id
                    event["turn_index"] = turn_index

                yield {
                    "event": event.get("type", "message"),
                    "data": json.dumps(event, default=str),
                }

            # Run finished. Persist the turn (best-effort — failures here
            # shouldn't break the already-delivered SSE stream).
            if final_event is not None:
                try:
                    threads.add_turn(
                        thread_id=thread.id,
                        turn_index=turn_index,
                        question=q,
                        answer=final_event.get("answer"),
                        confidence=final_event.get("confidence"),
                        score=final_event.get("score"),
                        sources=final_event.get("sources") or [],
                        run_id=final_event.get("run_id"),
                    )
                except Exception as e:
                    # Surface the persistence failure as a trailing event so
                    # the UI can show a warning without crashing the run view.
                    yield {
                        "event": "warning",
                        "data": json.dumps({
                            "where": "persist_turn",
                            "error": f"{type(e).__name__}: {e}",
                        }),
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
