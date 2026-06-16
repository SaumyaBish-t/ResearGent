"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useAgentStore, type ThreadTurn } from "@/lib/store";
import type { Source } from "@/lib/types";

/**
 * Chat-style conversation view of the current thread.
 *
 * Replaces the old single-answer modal: every persisted turn renders as its
 * own Q+A block, stacked top-to-bottom. The most recent turn auto-expands,
 * prior turns are collapsed (one click to expand). While a follow-up is in
 * flight, a "thinking" placeholder turn shows at the bottom.
 */
export default function ResultModal() {
  const threadTurns = useAgentStore((s) => s.threadTurns);
  const finalOutput = useAgentStore((s) => s.finalOutput);
  const savedPath = useAgentStore((s) => s.savedPath);
  const saveSkippedReason = useAgentStore((s) => s.saveSkippedReason);
  const error = useAgentStore((s) => s.error);
  const running = useAgentStore((s) => s.running);
  const liveQuery = useAgentStore((s) => s.query);
  const [dismissed, setDismissed] = useState(false);

  // Re-open whenever a fresh run arrives (or errors).
  const runId = finalOutput?.run_id ?? null;
  useEffect(() => {
    if (finalOutput || error || running) setDismissed(false);
  }, [runId, error, finalOutput, running]);

  // Esc-to-close.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setDismissed(true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Auto-scroll to the latest turn when one arrives or when streaming starts.
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [threadTurns.length, running]);

  const open =
    !dismissed && (threadTurns.length > 0 || running || !!error);
  if (!open) return null;

  const totalTurns = threadTurns.length + (running ? 1 : 0);

  return (
    <AnimatePresence mode="wait">
      {open && (
        <motion.div
          key="modal"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="pointer-events-auto absolute inset-0 z-20 flex items-center justify-center bg-black/70 p-6 backdrop-blur-xl"
          onClick={() => setDismissed(true)}
        >
          <motion.div
            initial={{ scale: 0.96, y: 20, opacity: 0 }}
            animate={{ scale: 1, y: 0, opacity: 1 }}
            exit={{ scale: 0.97, opacity: 0 }}
            transition={{ type: "spring", stiffness: 220, damping: 24 }}
            style={{ background: "rgba(10, 13, 20, 0.96)" }}
            className="glass relative flex max-h-[88vh] w-[min(1100px,96vw)] flex-col overflow-hidden rounded-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Accent rail */}
            <div className="h-px w-full bg-gradient-to-r from-transparent via-accent/60 to-transparent" />

            {/* Header */}
            <header className="flex items-start justify-between gap-4 px-7 pt-5 pb-4">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${error ? "bg-bad" : running ? "bg-accent" : "bg-good"}`}
                  />
                  <span className="font-mono text-[11.5px] uppercase tracking-[0.32em] text-ink-dim">
                    {error ? "run failed" : "research thread"}
                  </span>
                </div>
                <div className="mt-2 font-mono text-[11px] uppercase tracking-widest text-ink-mute">
                  <span className="tabular-nums text-ink">{totalTurns}</span>
                  <span className="ml-1.5">turn{totalTurns === 1 ? "" : "s"}</span>
                </div>
              </div>
              <button
                onClick={() => setDismissed(true)}
                aria-label="Close"
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-line text-ink-dim transition hover:border-accent/40 hover:bg-white/[0.03] hover:text-accent"
              >
                <span className="text-base leading-none">×</span>
              </button>
            </header>

            <div className="hairline mx-7" />

            {/* Conversation */}
            <div
              ref={scrollRef}
              className="flex-1 overflow-y-auto px-7 py-6"
            >
              {error && threadTurns.length === 0 ? (
                <ErrorBlock error={error} />
              ) : (
                <div className="space-y-5">
                  {threadTurns.map((t, i) => (
                    <TurnBlock
                      key={t.turn_index}
                      turn={t}
                      defaultOpen={i === threadTurns.length - 1 && !running}
                    />
                  ))}
                  {running && <LiveTurn question={liveQuery} />}
                  {error && threadTurns.length > 0 && (
                    <ErrorBlock error={error} />
                  )}
                </div>
              )}
            </div>

            {/* Footer (save status of latest turn) */}
            {(savedPath || saveSkippedReason) && !running && (
              <footer className="border-t border-line bg-white/[0.015] px-7 py-3 font-mono text-[10.5px]">
                {savedPath ? (
                  <span className="text-good">
                    <span className="mr-1.5">✓</span>
                    saved → <span className="text-good/90">{savedPath}</span>
                  </span>
                ) : (
                  <span className="text-warn">
                    <span className="mr-1.5">!</span>
                    save skipped —{" "}
                    <span className="text-warn/80">{saveSkippedReason}</span>
                  </span>
                )}
                <span className="float-right text-ink-mute">
                  <span className="kbd">esc</span>
                  <span className="ml-2 uppercase tracking-widest">to close</span>
                </span>
              </footer>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ---------------------------------------------------------------------------
// One Q+A in the conversation. Prior turns collapse to title only; expand on
// click. Latest turn auto-expands.
// ---------------------------------------------------------------------------
function TurnBlock({
  turn,
  defaultOpen,
}: {
  turn: ThreadTurn;
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  useEffect(() => setOpen(defaultOpen), [defaultOpen]);

  const confTone =
    turn.confidence === "high"
      ? "text-good"
      : turn.confidence === "medium"
        ? "text-warn"
        : "text-bad";

  return (
    <section className="overflow-hidden rounded-xl border border-line bg-white/[0.02]">
      {/* Question header — always visible, click toggles answer */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start gap-3 px-4 py-3 text-left transition hover:bg-white/[0.02]"
      >
        <span className="mt-[2px] flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-accent/15 font-mono text-[10.5px] font-semibold text-accent">
          Q{turn.turn_index + 1}
        </span>
        <span className="flex-1 text-[14px] leading-snug text-ink">
          {turn.question}
        </span>
        <span className="mt-[2px] flex items-center gap-2">
          {turn.confidence && (
            <span className={`font-mono text-[10px] uppercase tracking-widest ${confTone}`}>
              {turn.confidence}
            </span>
          )}
          <span
            className={`font-mono text-[14px] text-ink-mute transition-transform ${open ? "rotate-90" : ""}`}
          >
            ›
          </span>
        </span>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
            className="overflow-hidden"
          >
            <div className="hairline mx-4" />
            <div className="px-4 py-4">
              <article className="prose-invert max-w-none text-[15.5px] leading-[1.75] text-slate-200 [&_a]:text-accent [&_a]:underline [&_a]:decoration-accent/40 [&_a]:underline-offset-2 hover:[&_a]:decoration-accent [&_code]:rounded-md [&_code]:border [&_code]:border-line [&_code]:bg-white/[0.04] [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[13px] [&_code]:text-accent [&_h1]:mb-3 [&_h1]:mt-1 [&_h1]:text-[21px] [&_h1]:font-semibold [&_h1]:tracking-tight [&_h1]:text-ink [&_h2]:mb-2 [&_h2]:mt-5 [&_h2]:text-[17px] [&_h2]:font-semibold [&_h2]:tracking-tight [&_h2]:text-ink [&_h3]:mb-2 [&_h3]:mt-4 [&_h3]:text-[15px] [&_h3]:font-semibold [&_h3]:text-ink [&_li]:my-1.5 [&_p]:my-3 [&_strong]:text-ink [&_ul]:my-3 [&_ul]:list-disc [&_ul]:space-y-1.5 [&_ul]:pl-5">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {turn.answer || "_(empty answer)_"}
                </ReactMarkdown>
              </article>

              {turn.sources.length > 0 && (
                <SourcesList sources={turn.sources} />
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}

function SourcesList({ sources }: { sources: Source[] }) {
  return (
    <div className="mt-5 border-t border-line pt-4">
      <div className="mb-2 flex items-center gap-2">
        <span className="h-px flex-1 bg-line" />
        <span className="font-mono text-[9.5px] uppercase tracking-[0.32em] text-ink-mute">
          sources · {sources.length}
        </span>
        <span className="h-px flex-1 bg-line" />
      </div>
      <ul className="grid gap-2">
        {sources.map((s) => (
          <li
            key={s.tag}
            className="grid grid-cols-[auto_auto_1fr] items-baseline gap-3 rounded-lg border border-line bg-white/[0.015] px-3 py-2 transition hover:border-accent/30 hover:bg-white/[0.03]"
          >
            <span className="font-mono text-[12px] font-semibold tracking-wide text-accent">
              {s.tag}
            </span>
            <span className="rounded-md border border-line bg-white/[0.04] px-1.5 py-[1px] font-mono text-[10px] uppercase tracking-widest text-ink-dim">
              {s.signal}
            </span>
            <span className="break-words text-[13px] leading-relaxed text-slate-300">
              {s.citation}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function LiveTurn({ question }: { question: string }) {
  return (
    <section className="overflow-hidden rounded-xl border border-accent/30 bg-accent/[0.04]">
      <div className="flex items-start gap-3 px-4 py-3">
        <span className="mt-[2px] flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-accent/20 font-mono text-[10.5px] font-semibold text-accent">
          Q?
        </span>
        <span className="flex-1 text-[14px] leading-snug text-ink">
          {question}
        </span>
        <span className="mt-[2px] flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
          <span className="font-mono text-[10px] uppercase tracking-widest text-accent">
            thinking
          </span>
        </span>
      </div>
      <div className="hairline mx-4" />
      <div className="px-4 py-4 text-[13px] text-ink-dim">
        Agents are working — watch the network behind this modal. The answer
        appears here when the run completes.
      </div>
    </section>
  );
}

function ErrorBlock({ error }: { error: string }) {
  return (
    <div className="rounded-xl border border-bad/30 bg-bad/[0.06] p-4">
      <div className="mb-1 font-mono text-[10px] uppercase tracking-widest text-bad/80">
        error
      </div>
      <p className="text-sm leading-relaxed text-bad">{error}</p>
    </div>
  );
}
