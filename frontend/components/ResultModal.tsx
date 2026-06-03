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
  // otherwise stay true across runs — so after closing one answer, the next
  // run's answer would silently never appear. Reset it whenever a fresh
  // run_id (or error) arrives.
  const runId = finalOutput?.run_id ?? null;
  useEffect(() => {
    if (finalOutput || error) setDismissed(false);
  }, [runId, error, finalOutput]);

  const open = !!(finalOutput || error) && !dismissed;

  // Re-show the modal whenever a fresh result arrives.
  const key = finalOutput?.run_id ?? (error ? "error" : "none");

  return (
    <AnimatePresence mode="wait">
      {open && (
        <motion.div
          key={key}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="pointer-events-auto absolute inset-0 z-20 flex items-center justify-center bg-black/75 p-6 backdrop-blur-md"
          onClick={() => setDismissed(true)}
        >
          <motion.div
            initial={{ scale: 0.92, y: 24, opacity: 0 }}
            animate={{ scale: 1, y: 0, opacity: 1 }}
            exit={{ scale: 0.96, opacity: 0 }}
            transition={{ type: "spring", stiffness: 220, damping: 24 }}
            // Near-opaque card (overrides the translucent .glass bg) so the
            // answer text stays crisp regardless of the 3D scene behind it.
            style={{ background: "rgba(7, 10, 18, 0.97)" }}
            className="glass flex max-h-[82vh] w-[min(820px,94vw)] flex-col rounded-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <header className="flex items-start justify-between gap-4 border-b border-edge/60 px-6 py-4">
              <div>
                <div className="text-xs font-semibold uppercase tracking-widest text-accent">
                  {error ? "run failed" : "research result"}
                </div>
                {finalOutput && (
                  <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-slate-400">
                    <span>
                      confidence{" "}
                      <span
                        className={
                          finalOutput.confidence === "high"
                            ? "text-good"
                            : finalOutput.confidence === "medium"
                              ? "text-warn"
                              : "text-bad"
                        }
                      >
                        {finalOutput.confidence} · {finalOutput.score?.toFixed(2)}
                      </span>
                    </span>
                    <span>· {sources.length} sources</span>
                    {finalOutput.web_used && <span>· web</span>}
                    {finalOutput.rewrite_attempts > 0 && (
                      <span>· {finalOutput.rewrite_attempts} rewrites</span>
                    )}
                    {finalOutput.reflection_attempts > 0 && (
                      <span>· {finalOutput.reflection_attempts} reflections</span>
                    )}
                  </div>
                )}
              </div>
              <button
                onClick={() => setDismissed(true)}
                className="rounded-lg border border-edge px-2.5 py-1 text-xs text-slate-400 transition hover:border-accent/60 hover:text-accent"
              >
                ✕
              </button>
            </header>

            <div className="flex-1 overflow-y-auto px-6 py-5">
              {error ? (
                <p className="text-sm text-bad">{error}</p>
              ) : (
                <article className="prose-invert text-sm leading-relaxed text-slate-200 [&_a]:text-accent [&_code]:text-accent [&_h1]:mb-2 [&_h1]:text-lg [&_h2]:mb-2 [&_h2]:mt-4 [&_h2]:text-base [&_li]:my-1 [&_p]:my-2 [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {finalOutput?.answer || "_(empty answer)_"}
                  </ReactMarkdown>
                </article>
              )}

              {sources.length > 0 && (
                <div className="mt-6 border-t border-edge/60 pt-4">
                  <div className="mb-2 text-xs font-semibold uppercase tracking-widest text-slate-400">
                    sources
                  </div>
                  <ul className="space-y-2">
                    {sources.map((s) => (
                      <li key={s.tag} className="text-[12px] text-slate-400">
                        <span className="font-semibold text-accent">{s.tag}</span>{" "}
                        <span className="text-slate-500">[{s.signal}]</span>{" "}
                        {s.citation}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {(savedPath || saveSkippedReason) && (
              <footer className="border-t border-edge/60 px-6 py-3 text-[11px]">
                {savedPath ? (
                  <span className="text-good">✓ saved → {savedPath}</span>
                ) : (
                  <span className="text-warn">save skipped — {saveSkippedReason}</span>
                )}
              </footer>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
