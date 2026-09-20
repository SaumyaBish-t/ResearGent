"""FastAPI app — streaming /api/research endpoint + static web UI at /.

Endpoints
---------
  GET  /                       — single-file web UI (see src/api/web/index.html)
  GET  /api/research?q=...&k=8 — SSE stream: one event per agent node
  GET  /api/status             — provider routing + observability snapshot
  GET  /api/stats              — aggregated LLM call stats
  GET  /api/memory/topics      — user's research memory (topics + gaps)
  GET  /api/review             — literature review generation (SSE)

Why SSE not WebSocket
---------------------
The agent is server-push only. There's no need for bidirectional comms.
SSE is one-line easier to consume from the browser (EventSource API),
auto-reconnects, and survives reverse-proxy quirks better than WS.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.sessions import SessionMiddleware

from src.agent.stream import stream_agent
from src.agent.review_stream import stream_review
from src.auth.deps import current_user
from src.auth.routes import router as auth_router
from src.auth.users import User
from src.billing import quota, threads
from src.billing.routes import router as billing_router
from src.config import settings
from src.llm import list_status
from src.llm.observability import load_records, summarize
from src.memory import store as memory_store

WEB_DIR = Path(__file__).parent / "web"


def create_app() -> FastAPI:
    app = FastAPI(
        title="ResearGent",
        description="Agentic research engine — Corrective RAG + Self-Reflection",
        version="0.11.0",
    )

    _origins = settings.cors_origins_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth_router)
    app.include_router(billing_router)

    # ── User context middleware ────────────────────────────────────────────
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response as StarletteResponse
    from src.auth.context import current_user_id as _ctx_user_id

    class UserContextMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            # Try to resolve the user from the session cookie
            uid: str | None = None
            user_data = request.session.get("user")
            if user_data and isinstance(user_data, dict):
                uid = user_data.get("id")
            token = _ctx_user_id.set(uid)
            try:
                response = await call_next(request)
            finally:
                _ctx_user_id.reset(token)
            return response

    app.add_middleware(UserContextMiddleware)
    # SessionMiddleware added AFTER UserContextMiddleware so it runs FIRST
    # (last-added = outermost in Starlette). It must populate scope["session"]
    # before UserContextMiddleware reads request.session at dispatch time.
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        same_site="lax",
        https_only=settings.cookie_secure,
        max_age=600,
    )

    @app.get("/health", include_in_schema=False)
    async def health() -> dict:
        return {"ok": True}

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    # ── Streaming agent endpoint ──────────────────────────────────────────
    @app.get("/api/research")
    async def research(
        q: str = Query(..., description="The research question"),
        k: int = Query(8, description="Total chunks budget across all sub-questions"),
        domain: str | None = Query(
            None,
            description="Comma-separated domain id(s) to scope retrieval to "
            "(agentic_ai, quant_finance, time_series). "
            "Omit to search across all domains.",
        ),
        thread_id: str | None = Query(
            None,
            description="Continue an existing thread (omit to start a new one).",
        ),
        user: User = Depends(current_user),
    ):
        """SSE stream of agent execution events."""
        domain_scope: list[str] | None = None
        if domain:
            domain_scope = [d.strip() for d in domain.split(",") if d.strip()]

        # Build memory context
        memory_context = memory_store.build_memory_context(
            user_id=user.id, domain_scope=domain_scope
        )
        effective_q = q
        if memory_context:
            effective_q = f"{memory_context}\n\n[New question]\n{q}"
        elif thread_id:
            thread = threads.get_thread(thread_id=thread_id, user_id=user.id)
            prior_turns = threads.list_turns(thread_id=thread.id) if thread else []
            prefix = threads.build_context_prefix(prior_turns) if prior_turns else ""
            if prefix:
                effective_q = f"{prefix}\n\n[Current question]\n{q}".strip()

        # Quota + thread resolution
        if thread_id:
            thread = threads.get_thread(thread_id=thread_id, user_id=user.id)
            if not thread:
                raise HTTPException(404, "Thread not found.")
            decision = quota.check_can_add_turn(user=user, thread_id=thread.id)
        else:
            thread = None
            decision = quota.check_can_create_thread(user)

        if not decision.allowed:
            raise HTTPException(
                status_code=402,
                detail={"error": "quota_exceeded", **decision.to_dict()},
            )

        if thread is None:
            thread = threads.create_thread(user_id=user.id, title=q)
        turn_index = threads.next_turn_index(thread_id=thread.id)

        async def event_gen():
            loop = asyncio.get_running_loop()
            gen = stream_agent(effective_q, k=k, domain_scope=domain_scope)
            final_event: dict | None = None
            memory_payload: dict | None = None
            while True:
                try:
                    event = await loop.run_in_executor(None, _next_or_none, gen)
                except Exception as e:
                    yield {"event": "error", "data": json.dumps({"error": str(e)})}
                    return
                if event is None:
                    break

                if event.get("type") == "run_started":
                    event["thread_id"] = thread.id
                    event["turn_index"] = turn_index
                    event["question"] = q
                    if domain_scope:
                        event["domain_scope"] = domain_scope
                elif event.get("type") == "final":
                    final_event = event
                    event["thread_id"] = thread.id
                    event["turn_index"] = turn_index
                elif event.get("type") == "memory_extracted":
                    memory_payload = event.get("payload")

                yield {
                    "event": event.get("type", "message"),
                    "data": json.dumps(event, default=str),
                }

            # Persist turn
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
                    yield {
                        "event": "warning",
                        "data": json.dumps(
                            {"where": "persist_turn", "error": f"{type(e).__name__}: {e}"}
                        ),
                    }

            # Persist memory
            if memory_payload:
                try:
                    result = memory_store.ingest_extracted_memory(
                        user_id=user.id,
                        topics=memory_payload.get("topics") or [],
                        entities=memory_payload.get("entities") or [],
                        relationships=memory_payload.get("relationships") or [],
                        gaps=memory_payload.get("gaps") or [],
                    )
                    yield {"event": "memory_persisted", "data": json.dumps(result)}
                except Exception as e:
                    yield {
                        "event": "warning",
                        "data": json.dumps(
                            {"where": "persist_memory", "error": f"{type(e).__name__}: {e}"}
                        ),
                    }

        return EventSourceResponse(event_gen())

    # ── BibTeX export ──────────────────────────────────────────────────────
    @app.get("/api/threads/{thread_id}/turns/{turn_index}/export.bib")
    async def export_bibtex(
        thread_id: str,
        turn_index: int,
        user: User = Depends(current_user),
    ):
        """Export a turn's sources as BibTeX."""
        thread = threads.get_thread(thread_id=thread_id, user_id=user.id)
        if not thread:
            raise HTTPException(404, "Thread not found.")
        turn = threads.get_turn(thread_id=thread_id, turn_index=turn_index)
        if not turn:
            raise HTTPException(404, "Turn not found.")
        from src.export.bibtex import sources_to_bibtex

        bib = sources_to_bibtex(turn.sources, turn_index=turn_index)
        return PlainTextResponse(
            content=bib,
            media_type="text/x-bibtex",
            headers={
                "Content-Disposition": f"attachment; filename=researgent-turn-{turn_index}.bib"
            },
        )

    # ── Memory API ──────────────────────────────────────────────────────────
    @app.get("/api/memory/topics")
    async def memory_topics(
        limit: int = Query(20, description="Max topics to return"),
        user: User = Depends(current_user),
    ):
        topics = memory_store.get_user_topics(user_id=user.id, limit=limit)
        return JSONResponse(
            [
                {
                    "id": t.id,
                    "label": t.label,
                    "domain": t.domain,
                    "count": t.count,
                    "last_seen": t.last_seen.isoformat(),
                }
                for t in topics
            ]
        )

    @app.get("/api/memory/gaps")
    async def memory_gaps(
        limit: int = Query(10, description="Max gaps to return"),
        user: User = Depends(current_user),
    ):
        gaps = memory_store.get_open_gaps(user_id=user.id, limit=limit)
        return JSONResponse(
            [
                {
                    "id": g.id,
                    "topic_id": g.topic_id,
                    "gap_description": g.gap_description,
                    "status": g.status,
                    "created_at": g.created_at.isoformat(),
                }
                for g in gaps
            ]
        )

    @app.get("/api/memory/context")
    async def memory_context(
        domain: str | None = Query(None, description="Optional domain filter"),
        user: User = Depends(current_user),
    ):
        domain_scope = [d.strip() for d in domain.split(",")] if domain else None
        ctx = memory_store.build_memory_context(user_id=user.id, domain_scope=domain_scope)
        return PlainTextResponse(ctx)

    # ── Provenance Reflags API (Phase 23) ─────────────────────────────────────
    @app.get("/api/provenance/reflags")
    async def provenance_reflags(
        since_days: int | None = Query(None, description="Only show reflags from last N days"),
        unnotified_only: bool = Query(True, description="Only show un-notified reflags"),
        user: User = Depends(current_user),
    ):
        from src.db import connection
        with connection() as conn:
            with conn.cursor() as cur:
                if since_days:
                    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
                    if unnotified_only:
                        cur.execute(
                            """
                            SELECT doi, arxiv_id, old_status, new_status, detected_at, notified
                            FROM provenance_reflags
                            WHERE detected_at >= %s AND notified = false
                            ORDER BY detected_at DESC
                            """,
                            (cutoff,),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT doi, arxiv_id, old_status, new_status, detected_at, notified
                            FROM provenance_reflags
                            WHERE detected_at >= %s
                            ORDER BY detected_at DESC
                            """,
                            (cutoff,),
                        )
                else:
                    if unnotified_only:
                        cur.execute(
                            """
                            SELECT doi, arxiv_id, old_status, new_status, detected_at, notified
                            FROM provenance_reflags
                            WHERE notified = false
                            ORDER BY detected_at DESC
                            """
                        )
                    else:
                        cur.execute(
                            """
                            SELECT doi, arxiv_id, old_status, new_status, detected_at, notified
                            FROM provenance_reflags
                            ORDER BY detected_at DESC
                            """
                        )
                rows = cur.fetchall()
                return JSONResponse([
                    {
                        "doi": r[0],
                        "arxiv_id": r[1],
                        "old_status": r[2],
                        "new_status": r[3],
                        "detected_at": r[4].isoformat(),
                        "notified": r[5],
                    }
                    for r in rows
                ])

    @app.post("/api/provenance/reflags/mark-notified")
    async def mark_reflags_notified(
        ids: list[str],
        user: User = Depends(current_user),
    ):
        from src.db import connection
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE provenance_reflags SET notified = true WHERE id = ANY(%s)",
                    (ids,),
                )
        return JSONResponse({"updated": cur.rowcount})

    # ── Literature Review endpoint ──────────────────────────────────────────
    @app.get("/api/review")
    async def literature_review(
        q: str = Query(..., description="The literature review topic or request"),
        user: User = Depends(current_user),
    ):
        """SSE stream for literature review generation."""
        decision = quota.check_can_create_review(user)
        if not decision.allowed:
            raise HTTPException(
                status_code=402,
                detail={"error": "quota_exceeded", **decision.to_dict()},
            )

        async def review_event_gen():
            loop = asyncio.get_running_loop()
            gen = stream_review(q)
            while True:
                try:
                    event = await loop.run_in_executor(None, _next_or_none, gen)
                except Exception as e:
                    yield {"event": "error", "data": json.dumps({"error": str(e)})}
                    return
                if event is None:
                    break

                if event.get("type") == "review_complete":
                    final_title = event.get("title")
                    final_markdown = event.get("markdown")
                    final_duration = event.get("duration_ms", 0)

                    # Persist the review and add its ID to the event
                    try:
                        from src.billing import reviews
                        db_rev = reviews.create_review(
                            user_id=user.id,
                            query=q,
                            title=final_title or "Literature Review",
                            markdown=final_markdown,
                            duration_ms=final_duration,
                        )
                        event["review_id"] = str(db_rev.id)
                    except Exception as e:
                        yield {
                            "event": "warning",
                            "data": json.dumps(
                                {"where": "persist_review", "error": f"{type(e).__name__}: {e}"}
                            ),
                        }

                yield {
                    "event": event.get("type", "message"),
                    "data": json.dumps(event, default=str),
                }

        return EventSourceResponse(review_event_gen())

    @app.get("/api/reviews")
    async def list_user_reviews(
        user: User = Depends(current_user),
    ):
        """List literature reviews for the current user."""
        from src.billing import reviews
        items = reviews.list_reviews(user_id=user.id)
        return JSONResponse({"reviews": [
            {
                "id": str(r.id),
                "title": r.title,
                "query": r.query,
                "created_at": r.created_at.isoformat(),
            }
            for r in items
        ]})

    @app.get("/api/reviews/{review_id}")
    async def get_user_review(
        review_id: str,
        user: User = Depends(current_user),
    ):
        """Get a specific literature review by ID."""
        from src.billing import reviews
        r = reviews.get_review(review_id=review_id, user_id=user.id)
        if not r:
            raise HTTPException(404, "Review not found.")
        return JSONResponse({
            "id": str(r.id),
            "query": r.query,
            "title": r.title,
            "markdown": r.markdown,
            "duration_ms": r.duration_ms,
            "created_at": r.created_at.isoformat(),
        })

    @app.get("/api/reviews/{review_id}/pdf")
    async def download_review_pdf(
        review_id: str,
        user: User = Depends(current_user),
    ):
        """Generate and download a beautifully styled PDF of a literature review."""
        from src.billing import reviews
        from src.export.pdf import markdown_to_pdf_bytes

        r = reviews.get_review(review_id=review_id, user_id=user.id)
        if not r:
            raise HTTPException(404, "Review not found.")
        if not r.markdown:
            raise HTTPException(400, "Review markdown content is empty.")

        pdf_bytes = markdown_to_pdf_bytes(r.title, r.markdown)
        
        # Build safe filename
        safe_title = "".join(c if c.isalnum() else "_" for c in r.title)
        filename = f"{safe_title[:50]}.pdf"

        from fastapi.responses import Response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            },
        )

    # ── Document upload endpoint ────────────────────────────────────────────
    from fastapi import UploadFile, File, Form

    @app.post("/api/documents")
    async def upload_document(
        file: UploadFile = File(...),
        domain: str | None = Form(None),
        user: User = Depends(current_user),
    ):
        """Upload a PDF or Markdown note for ingestion."""
        import tempfile
        from src.auth.context import current_user_id as _ctx_uid

        _ctx_uid.set(str(user.id))

        filename = file.filename or "upload"
        content_bytes = await file.read()

        if filename.lower().endswith(".pdf"):
            # Write to a temp file, ingest, clean up
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content_bytes)
                tmp_path = Path(tmp.name)
            try:
                from src.ingest.pipeline import ingest_file
                result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: ingest_file(tmp_path, domain=domain, verbose=False)
                )
            finally:
                tmp_path.unlink(missing_ok=True)
            return JSONResponse(result)

        elif filename.lower().endswith(".md"):
            content_str = content_bytes.decode("utf-8", errors="ignore")
            from src.ingest.pipeline import ingest_db_note
            inserted = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: ingest_db_note(
                    user_id=str(user.id),
                    path=f"uploads/{filename}",
                    content=content_str,
                    domain=domain,
                ),
            )
            return JSONResponse({"filename": filename, "chunks_inserted": inserted})

        raise HTTPException(400, "Only .pdf and .md files are supported.")

    # ── Diagnostic endpoints ────────────────────────────────────────────────
    @app.get("/api/status")
    async def status() -> JSONResponse:
        return JSONResponse(list_status())

    @app.get("/api/stats")
    async def stats(last: int = Query(0)) -> JSONResponse:
        recs = load_records(limit=last or None)
        return JSONResponse(summarize(recs))

    return app


def _next_or_none(gen) -> Any:
    try:
        return next(gen)
    except StopIteration:
        return None