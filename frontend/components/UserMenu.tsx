"use client";

import { useEffect, useRef, useState } from "react";
import { useAgentStore } from "@/lib/store";

/**
 * Top-right avatar + dropdown. Shows the signed-in user's email + current
 * usage, plus Sign out. Subscribers get a "PRO" chip, admins get "ADMIN".
 */
export default function UserMenu() {
  const user = useAgentStore((s) => s.user);
  const usage = useAgentStore((s) => s.usage);
  const signOut = useAgentStore((s) => s.signOut);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Click-outside to close.
  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [open]);

  if (!user) return null;

  const tier = user.is_admin
    ? "ADMIN"
    : usage?.is_subscribed
      ? "LIFETIME"
      : "FREE";

  const initials = (user.name || user.email)
    .split(/\s+|@/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() || "")
    .join("") || "·";

  return (
    <div ref={ref} className="pointer-events-auto relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 rounded-full border border-line bg-white/[0.04] py-1 pl-1 pr-3 transition hover:border-accent/40 hover:bg-white/[0.07]"
      >
        {user.picture ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={user.picture}
            alt=""
            referrerPolicy="no-referrer"
            className="h-7 w-7 rounded-full"
          />
        ) : (
          <span className="flex h-7 w-7 items-center justify-center rounded-full bg-accent/20 font-mono text-[11px] text-accent">
            {initials}
          </span>
        )}
        <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-ink-dim">
          {tier}
        </span>
      </button>

      {open && (
        <div className="glass absolute right-0 top-[42px] z-40 w-[260px] overflow-hidden rounded-xl p-3">
          <div className="px-1 py-1">
            <div className="truncate text-[13px] text-ink">{user.name || user.email}</div>
            {user.name && (
              <div className="truncate font-mono text-[10.5px] text-ink-mute">
                {user.email}
              </div>
            )}
          </div>

          <div className="hairline my-2" />

          {usage && (
            <div className="rounded-lg border border-line bg-white/[0.02] px-3 py-2 text-[11.5px] text-ink-dim">
              {usage.is_admin ? (
                <span className="text-good">Unlimited (admin)</span>
              ) : usage.is_subscribed ? (
                <span className="text-good">Unlimited (subscribed)</span>
              ) : (
                <div className="flex items-center justify-between">
                  <span className="font-mono uppercase tracking-widest text-ink-mute">
                    this month
                  </span>
                  <span className="font-mono tabular-nums text-ink">
                    {usage.threads_used_this_month} / {usage.threads_limit}
                  </span>
                </div>
              )}
            </div>
          )}

          <button
            onClick={() => {
              setOpen(false);
              void signOut();
            }}
            className="mt-2 w-full rounded-lg border border-line px-3 py-2 text-left text-[12px] text-ink-dim transition hover:border-bad/40 hover:text-bad"
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
