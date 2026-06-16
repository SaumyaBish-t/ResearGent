"""
Billing + history HTTP routes.

  GET  /api/usage             — quota snapshot for the signed-in user
  GET  /api/threads           — user's research threads, newest first
  GET  /api/threads/{id}      — single thread with full turn history
  POST /billing/checkout      — create a Razorpay subscription, return id+key
  POST /billing/webhook       — Razorpay → us, mark subs active/cancelled/...

The /api/research endpoint lives in src/api/app.py (it's tightly coupled to
the SSE wiring) but it also persists turns via src/billing/threads.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from src.auth.deps import current_user
from src.auth.users import User
from src.billing import quota, razorpay_client, subscription, threads
from src.config import settings


router = APIRouter(tags=["billing"])


# ---------------------------------------------------------------------------
# Usage + history
# ---------------------------------------------------------------------------


@router.get("/api/usage")
def usage(user: User = Depends(current_user)) -> dict[str, Any]:
    """Quota snapshot — drives the frontend usage chip + paywall trigger."""
    return quota.usage_snapshot(user)


@router.get("/api/threads")
def list_user_threads(user: User = Depends(current_user)) -> dict[str, Any]:
    """List threads for the signed-in user, newest first."""
    rows = threads.list_threads(user_id=user.id)
    return {
        "threads": [
            {
                "id": t.id,
                "title": t.title,
                "created_at": t.created_at.isoformat(),
            }
            for t in rows
        ]
    }


@router.get("/api/threads/{thread_id}")
def get_user_thread(thread_id: str, user: User = Depends(current_user)) -> dict[str, Any]:
    """Fetch a single thread + every turn in it. 404 if not the user's."""
    thread = threads.get_thread(thread_id=thread_id, user_id=user.id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found.")
    turns = threads.list_turns(thread_id=thread.id)
    return {
        "thread": {
            "id": thread.id,
            "title": thread.title,
            "created_at": thread.created_at.isoformat(),
        },
        "turns": [
            {
                "turn_index": t.turn_index,
                "question": t.question,
                "answer": t.answer,
                "confidence": t.confidence,
                "score": t.score,
                "sources": t.sources,
                "run_id": t.run_id,
                "created_at": t.created_at.isoformat(),
            }
            for t in turns
        ],
    }


# ---------------------------------------------------------------------------
# Razorpay: one-time "lifetime unlock" payment
# ---------------------------------------------------------------------------
# Flow:
#   1. POST /billing/checkout  → backend creates a Razorpay ORDER, returns
#      {key_id, order_id, amount, customer, price_inr}.
#   2. Frontend opens Razorpay Checkout with that order. User pays.
#   3. Razorpay returns {razorpay_payment_id, razorpay_order_id, razorpay_signature}
#      to the frontend's `handler` callback.
#   4. Frontend POSTs them to /billing/verify. Backend verifies the HMAC and
#      flips the user to "lifetime active". Done — no webhook involved.


@router.post("/billing/checkout")
def create_checkout(user: User = Depends(current_user)) -> dict[str, Any]:
    """
    Create a Razorpay ORDER for the lifetime-unlock price and return the
    data the frontend Checkout widget needs.
    """
    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        raise HTTPException(500, "Razorpay not configured. Set RAZORPAY_KEY_ID + RAZORPAY_KEY_SECRET in .env.")

    try:
        order = razorpay_client.create_order(
            amount_inr=settings.razorpay_price_inr,
            notes={"user_id": user.id, "email": user.email},
            receipt=f"researgent-lifetime-{user.id[:8]}",
        )
    except Exception as e:
        raise HTTPException(502, f"Razorpay order create failed: {e}")

    return {
        "key_id": settings.razorpay_key_id,
        "order_id": order["id"],
        "amount": order["amount"],            # paise
        "currency": order["currency"],
        "customer": {"name": user.name or "", "email": user.email},
        "price_inr": settings.razorpay_price_inr,
    }


@router.post("/billing/verify")
async def verify_payment(
    request: Request,
    user: User = Depends(current_user),
) -> JSONResponse:
    """
    Verify the success payload Razorpay's Checkout returns to the frontend.

    Body: { razorpay_order_id, razorpay_payment_id, razorpay_signature }
    On match: mark user lifetime-active. Returns {is_subscribed: true}.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Bad JSON.")

    order_id = body.get("razorpay_order_id")
    payment_id = body.get("razorpay_payment_id")
    signature = body.get("razorpay_signature")
    if not (order_id and payment_id and signature):
        raise HTTPException(400, "Missing payment fields.")

    if not razorpay_client.verify_payment_signature(
        order_id=order_id, payment_id=payment_id, signature=signature
    ):
        raise HTTPException(400, "Bad payment signature.")

    subscription.mark_lifetime_active(user_id=user.id, razorpay_payment_id=payment_id)
    return JSONResponse({"is_subscribed": True})
