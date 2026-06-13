"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useAgentStore } from "@/lib/store";

export default function ResultModal() {
  const finalOutput = useAgentStore((s) => s.finalOutput);
  const sources = useAgentStore((s) => s.sources);
  const savedPath = useAgentStore((s) => s.savedPath);
  const saveSkippedReason = useAgentStore((s) => s.saveSkippedReason);
  const error = useAgentStore((s) => s.error);
  const [dismissed, setDismissed] = useState(false);

  // Re-open for every NEW result/error. `dismissed` is local state that would
  // otherwise stay true across runs.
  const runId = finalOutput?.run_id ?? null;
  useEffect(() => {
    if (finalOutput || error) setDismissed(false);
  }, [runId, error, finalOutput]);

  // Esc-to-close.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setDismissed(true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const open = !!(finalOutput || error) && !dismissed;
  const key = finalOutput?.run_id ?? (error ? "error" : "none");

  const confTone =
    finalOutput?.confidence === "high"
      ? "text-good"
      : finalOutput?.confidence === "medium"
        ? "text-warn"
        : "text-bad";

  return (
    <AnimatePresence mode="wait">
      {open && (
        <motion.div
          key={key}
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
            // Near-opaque card — keeps answer text crisp over the 3D scene.
            style={{ background: "rgba(10, 13, 20, 0.96)" }}
            className="glass relative flex max-h-[88vh] w-[min(1100px,96vw)] flex-col overflow-hidden rounded-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Accent rail (top edge) — gives the card a clear identity. */}
            <div className="h-px w-full bg-gradient-to-r from-transparent via-accent/60 to-transparent" />

            <header className="flex items-start justify-between gap-4 px-7 pt-5 pb-4">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${error ? "bg-bad" : "bg-good"}`}
                  />
                  <span className="font-mono text-[11.5px] uppercase tracking-[0.32em] text-ink-dim">
                    {error ? "run failed" : "research result"}
                  </span>
                </div>
                {finalOutput && (
                  <div className="mt-3 flex flex-wrap items-center gap-2 font-mono text-[11.5px] uppercase tracking-widest text-ink-mute">
                    <Pill>
                      conf{" "}
                      <span className={confTone}>
                        {finalOutput.confidence}
                      </span>
                      {typeof finalOutput.score === "number" && (
                        <span className="ml-1 tabular-nums text-ink-dim">
                          {finalOutput.score.toFixed(2)}
                        </span>
                      )}
                    </Pill>
                    <Pill>
                      <span className="tabular-nums text-ink">
                        {sources.length}
                      </span>{" "}
                      sources
                    </Pill>
                    {finalOutput.web_used && <Pill>web</Pill>}
                    {finalOutput.rewrite_attempts > 0 && (
                      <Pill>
                        <span className="tabular-nums text-ink">
                          {finalOutput.rewrite_attempts}
                        </span>{" "}
                        rewrites
                      </Pill>
                    )}
                    {finalOutput.reflection_attempts > 0 && (
                      <Pill>
                        <span className="tabular-nums text-ink">
                          {finalOutput.reflection_attempts}
                        </span>{" "}
                        reflections
                      </Pill>
                    )}
                  </div>
                )}
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

            <div className="flex-1 overflow-y-auto px-7 py-6">
              {error ? (
                <div className="rounded-xl border border-bad/30 bg-bad/[0.06] p-4">
                  <div className="mb-1 font-mono text-[10px] uppercase tracking-widest text-bad/80">
                    error
                  </div>
                  <p className="text-sm leading-relaxed text-bad">{error}</p>
                </div>
              ) : (
                <article className="prose-invert max-w-none text-[16px] leading-[1.75] text-slate-200 [&_a]:text-accent [&_a]:underline [&_a]:decoration-accent/40 [&_a]:underline-offset-2 hover:[&_a]:decoration-accent [&_code]:rounded-md [&_code]:border [&_code]:border-line [&_code]:bg-white/[0.04] [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[13.5px] [&_code]:text-accent [&_h1]:mb-3 [&_h1]:mt-1 [&_h1]:text-[22px] [&_h1]:font-semibold [&_h1]:tracking-tight [&_h1]:text-ink [&_h2]:mb-2 [&_h2]:mt-6 [&_h2]:text-[18px] [&_h2]:font-semibold [&_h2]:tracking-tight [&_h2]:text-ink [&_h3]:mb-2 [&_h3]:mt-5 [&_h3]:text-[15px] [&_h3]:font-semibold [&_h3]:text-ink [&_li]:my-1.5 [&_p]:my-3.5 [&_strong]:text-ink [&_ul]:my-3.5 [&_ul]:list-disc [&_ul]:space-y-1.5 [&_ul]:pl-5">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {finalOutput?.answer || "_(empty answer)_"}
                  </ReactMarkdown>
                </article>
              )}

              {sources.length > 0 && (
                <div className="mt-8 border-t border-line pt-5">
                  <div className="mb-3 flex items-center gap-2">
                    <span className="h-px flex-1 bg-line" />
                    <span className="font-mono text-[10px] uppercase tracking-[0.32em] text-ink-mute">
                      sources · {sources.length}
                    </span>
                    <span className="h-px flex-1 bg-line" />
                  </div>
                  <ul className="grid gap-2.5">
                    {sources.map((s) => (
                      <li
                        key={s.tag}
                        className="grid grid-cols-[auto_auto_1fr] items-baseline gap-3 rounded-lg border border-line bg-white/[0.015] px-3.5 py-2.5 transition hover:border-accent/30 hover:bg-white/[0.03]"
                      >
                        <span className="font-mono text-[12.5px] font-semibold tracking-wide text-accent">
                          {s.tag}
                        </span>
                        <span className="rounded-md border border-line bg-white/[0.04] px-1.5 py-[1px] font-mono text-[10.5px] uppercase tracking-widest text-ink-dim">
                          {s.signal}
                        </span>
                        <span className="break-words text-[13.5px] leading-relaxed text-slate-300">
                          {s.citation}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {(savedPath || saveSkippedReason) && (
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

function Pill({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-line bg-white/[0.03] px-2 py-[3px] text-ink-dim">
      {children}
    </span>
  );
}
