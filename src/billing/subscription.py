"""
Subscription state lookup.

`is_active(user_id)` is the single source of truth for "is this user a paying
subscriber right now". Used by the quota check + the frontend `/auth/me` to
show subscribed UI.

A subscription is active when:
  - status is one of {"active", "authenticated", "created"} AND
  - current_period_end is either NULL (e.g. just-created, awaiting first charge)
    OR in the future.

Razorpay subscription statuses (from their docs):
  created | authenticated | active | pending | halted |
  cancelled | completed | expired
We treat the first three as "currently entitled".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.db import connection


_ACTIVE_STATUSES = ("active", "authenticated", "created")


def is_active(user_id: str) -> bool:
    """True if the user has any active subscription row."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT status, current_period_end
            FROM subscriptions
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (user_id,),
        )
        row = cur.fetchone()

    if not row:
        return False
    if row["status"] not in _ACTIVE_STATUSES:
        return False
    end = row.get("current_period_end")
    if end is None:
        # Just-created subs may not have a period end yet — still entitled.
        return True
    # psycopg returns tz-aware datetimes for TIMESTAMPTZ; compare in UTC.
    return end > datetime.now(timezone.utc)


def mark_lifetime_active(*, user_id: str, razorpay_payment_id: str) -> None:
    """
    Flip the user to "lifetime unlimited" after a verified one-time payment.

    We reuse the subscriptions table — status='active' + current_period_end=NULL
    is the canonical "entitled forever" shape, and `is_active()` already treats
    NULL period_end as still-entitled.
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO subscriptions (
                user_id, razorpay_subscription_id, razorpay_customer_id,
                status, current_period_end, updated_at
            ) VALUES (%s, %s, NULL, 'active', NULL, now())
            ON CONFLICT (razorpay_subscription_id) DO UPDATE SET
                status     = 'active',
                updated_at = now()
            """,
            (user_id, f"payment:{razorpay_payment_id}"),
        )


def upsert_subscription(
    *,
    user_id: str,
    razorpay_subscription_id: str,
    razorpay_customer_id: Optional[str],
    status: str,
    current_period_end: Optional[datetime],
) -> None:
    """
    Insert-or-update a subscription row keyed on razorpay_subscription_id.
    Called from the webhook handler on every state-change event.
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO subscriptions (
                user_id, razorpay_subscription_id, razorpay_customer_id,
                status, current_period_end, updated_at
            ) VALUES (%s, %s, %s, %s, %s, now())
            ON CONFLICT (razorpay_subscription_id) DO UPDATE SET
                status              = EXCLUDED.status,
                current_period_end  = EXCLUDED.current_period_end,
                razorpay_customer_id = COALESCE(EXCLUDED.razorpay_customer_id,
                                                subscriptions.razorpay_customer_id),
                updated_at          = now()
            """,
            (
                user_id,
                razorpay_subscription_id,
                razorpay_customer_id,
                status,
                current_period_end,
            ),
        )
