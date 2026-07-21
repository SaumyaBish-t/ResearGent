"""Review streamer — wraps the literature review pipeline as an SSE-compatible generator."""
from __future__ import annotations

import time
import uuid
from typing import Any, Iterator

from src.agent.nodes.review_gatherer import run_review


def stream_review(request: str) -> Iterator[dict[str, Any]]:
    """
    Run the literature review pipeline and yield SSE-compatible events.
    Wraps run_review() generator output.
    """
    rid = uuid.uuid4().hex[:12]

    yield {
        "type": "review_started",
        "run_id": rid,
        "request": request,
        "ts": time.time(),
    }

    try:
        for event in run_review(request):
            event["run_id"] = rid
            yield event
            if event.get("type") == "review_complete":
                break
            if event.get("type") == "review_error":
                return
    except Exception as e:
        yield {
            "type": "review_error",
            "run_id": rid,
            "error": f"{type(e).__name__}: {e}",
            "ts": time.time(),
        }