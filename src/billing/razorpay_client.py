"""
Razorpay client wrapper + signature verification helpers.

Why wrap the SDK at all?
  - One place to handle "credentials not set" — every route can ask for the
    client and get a clear RuntimeError instead of an obscure AttributeError.
  - One place to verify webhook signatures + payment-success signatures so
    they share the same constant-time-compare path.
"""

from __future__ import annotations

import hashlib
import hmac
from functools import lru_cache
from typing import Any, Optional

from src.config import settings


@lru_cache(maxsize=1)
def _client():
    """Return a cached Razorpay client, or raise if not configured."""
    import razorpay  # type: ignore

    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        raise RuntimeError(
            "Razorpay not configured. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in .env."
        )
    return razorpay.Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))


def create_order(
    *,
    amount_inr: int,
    notes: Optional[dict[str, str]] = None,
    receipt: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create a Razorpay ORDER for a one-time payment.

    `amount_inr` is whole rupees (we don't sell fractional units). Razorpay's
    API takes paise — multiply by 100. Returns the raw order dict; `id` is what
    the frontend Checkout needs.
    """
    return _client().order.create({
        "amount": int(amount_inr) * 100,
        "currency": "INR",
        "receipt": receipt or "researgent-lifetime",
        "payment_capture": 1,
        "notes": notes or {},
    })


def verify_payment_signature(
    *,
    order_id: str,
    payment_id: str,
    signature: str,
) -> bool:
    """
    Verify the success signature Razorpay returns to the frontend after a
    one-time payment.

    Spec: HMAC-SHA256(key_secret, "<order_id>|<payment_id>") == signature.
    Constant-time compare to defeat timing attacks.
    """
    secret = settings.razorpay_key_secret
    if not secret:
        return False
    payload = f"{order_id}|{payment_id}".encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")
