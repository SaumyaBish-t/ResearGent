"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useAgentStore, type ThreadSummary } from "@/lib/store";

/**
 * Collapsible left drawer of the user's past research threads.
 *
 * Closed: a thin vertical handle on the left edge.
 * Open:   a 320px-wide panel with one row per thread; clicking a row hydrates
 *         the dashboard from that thread (latest answer + sources).
 *
 * Hidden during a run so it can't fight the LogPanel for the same space.
 */
export default function HistorySidebar() {
  const mode = useAgentStore((s) => s.mode);
  const threads = useAgentStore((s) => s.threads);
  const reviews = useAgentStore((s) => s.reviews);
  const currentThreadId = useAgentStore((s) => s.currentThreadId);
  const currentReviewId = useAgentStore((s) => s.currentReviewId);
  const openThread = useAgentStore((s) => s.openThread);
  const openReview = useAgentStore((s) => s.openReview);
  const reset = useAgentStore((s) => s.reset);
  const running = useAgentStore((s) => s.running);
  const reviewRunning = useAgentStore((s) => s.reviewRunning);
  const [open, setOpen] = useState(false);

  const busy = running || reviewRunning;

  // Don't compete with the LogPanel during an active run.
  if (busy) return null;

  const isReview = mode === "review";
  const items = isReview ? reviews : threads;
  const currentId = isReview ? currentReviewId : currentThreadId;
  const title = isReview ? "review history" : "research history";
  const emptyMessage = isReview ? "No previous reviews yet." : "No previous research yet.";
  const newButtonLabel = isReview ? "+ new review" : "+ new research";
  const ariaLabel = isReview ? "Open review history" : "Open research history";

  return (
    <>
      {/* Edge handle — visible always when closed */}
      {!open && (
        <button
          onClick={() => setOpen(true)}
          className="pointer-events-auto absolute left-0 top-1/2 z-20 flex h-24 w-6 -translate-y-1/2 items-center justify-center rounded-r-lg border border-l-0 border-line bg-[rgba(11,14,22,0.85)] backdrop-blur transition hover:border-accent/40 hover:bg-[rgba(14,18,28,0.95)]"
          aria-label={ariaLabel}
        >
          <div className="flex flex-col items-center gap-1.5">
            <span className="h-1 w-1 rounded-full bg-accent" />
            <span className="h-1 w-1 rounded-full bg-ink-dim" />
            <span className="h-1 w-1 rounded-full bg-ink-dim" />
          </div>
        </button>
      )}

      <AnimatePresence>
        {open && (
          <motion.aside
            initial={{ x: -340, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -340, opacity: 0 }}
            transition={{ type: "spring", stiffness: 220, damping: 28 }}
            className="glass pointer-events-auto absolute left-4 top-16 z-20 flex h-[calc(100%-5.5rem)] w-[320px] flex-col overflow-hidden rounded-2xl"
          >
            <header className="flex items-center justify-between px-5 pt-4 pb-3">
              <div className="font-mono text-[10px] uppercase tracking-[0.28em] text-ink-dim">
                {title}
              </div>
              <button
                onClick={() => setOpen(false)}
                className="rounded-md border border-line px-2 py-1 text-ink-dim transition hover:border-accent/40 hover:text-accent"
                aria-label="Close history"
              >
                <span className="text-xs leading-none">×</span>
              </button>
            </header>

            <div className="hairline mx-5" />

            <div className="flex-1 overflow-y-auto px-3 py-3">
              {items.length === 0 ? (
                <div className="px-2 py-6 text-center text-[12px] text-ink-mute">
                  {emptyMessage}
                </div>
              ) : (
                <ul className="space-y-1">
                  {items.map((t) => (
                    <ThreadRow
                      key={t.id}
                      title={t.title}
                      created_at={t.created_at}
                      active={t.id === currentId}
                      onClick={() => {
                        setOpen(false);
                        if (isReview) {
                          void openReview(t.id);
                        } else {
                          void openThread(t.id);
                        }
                      }}
                    />
                  ))}
                </ul>
              )}
            </div>

            <footer className="border-t border-line px-5 py-3">
              <button
                onClick={() => {
                  setOpen(false);
                  reset();
                }}
                className="w-full rounded-lg border border-line bg-white/[0.02] px-3 py-2 text-center font-mono text-[11px] uppercase tracking-[0.22em] text-ink-dim transition hover:border-accent/40 hover:text-accent"
              >
                {newButtonLabel}
              </button>
            </footer>
          </motion.aside>
        )}
      </AnimatePresence>
    </>
  );
}

function ThreadRow({
  title,
  created_at,
  active,
  onClick,
}: {
  title: string;
  created_at: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <li>
      <button
        onClick={onClick}
        className={`flex w-full flex-col items-start gap-1 rounded-lg px-3 py-2 text-left transition ${
          active
            ? "bg-accent/10 text-ink"
            : "text-ink-dim hover:bg-white/[0.03] hover:text-ink"
        }`}
      >
        <div className="line-clamp-2 text-[12.5px] leading-snug">{title}</div>
        <div className="font-mono text-[9.5px] uppercase tracking-widest text-ink-mute">
          {formatStamp(created_at)}
        </div>
      </button>
    </li>
  );
}

function formatStamp(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString([], {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  } catch {
    return iso;
  }
}
