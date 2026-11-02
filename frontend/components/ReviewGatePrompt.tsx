"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useAgentStore } from "@/lib/store";

/**
 * Human Review Gate prompt (Phase 24).
 *
 * The Novelty Check pauses the pipeline when the user's own results look
 * closely related to prior work. This is where the human decides whether the
 * comparison is fair — "same method, better number" might be a genuine
 * improvement, or it might be a mis-specified baseline.
 *
 * The decision goes to POST /api/novelty/review, which resumes the paused
 * LangGraph run from the gate.
 */
export default function ReviewGatePrompt() {
  const request = useAgentStore((s) => s.reviewRequest);
  const resuming = useAgentStore((s) => s.resumingReview);
  const error = useAgentStore((s) => s.resultsError);
  const resumeReview = useAgentStore((s) => s.resumeReview);

  return (
    <AnimatePresence>
      {request && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="pointer-events-auto absolute inset-0 z-40 flex items-center justify-center p-4"
        >
          {/* Scrim */}
          <div className="absolute inset-0 bg-base/70 backdrop-blur-sm" />

          <motion.div
            initial={{ opacity: 0, y: 14, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 14, scale: 0.98 }}
            transition={{ type: "spring", stiffness: 260, damping: 26 }}
            className="glass-bright relative w-[min(520px,94vw)] overflow-hidden rounded-2xl"
          >
            {/* Header */}
            <div className="flex items-center gap-2.5 border-b border-line px-5 py-3.5">
              <span className="h-2 w-2 rounded-full bg-warn dot-glow" style={{ color: "#fbbf24" }} />
              <span className="font-mono text-[10px] uppercase tracking-[0.28em] text-warn">
                human review gate
              </span>
              <span className="ml-auto font-mono text-[10px] tabular-nums text-ink-mute">
                {request.similarity.toFixed(2)}
              </span>
            </div>

            <div className="px-5 py-4">
              <p className="text-[13px] leading-relaxed text-ink">
                Your results look related to work we already found.
                {request.verdict ? (
                  <>
                    {" "}
                    We read this as{" "}
                    <span className="font-mono text-accent">{request.verdict}</span>.
                  </>
                ) : null}
              </p>

              {request.rationale && (
                <p className="mt-2 text-[12px] leading-relaxed text-ink-dim">
                  {request.rationale}
                </p>
              )}

              {/* The metric comparison — the actual evidence */}
              {(request.matched_metrics ?? []).length > 0 && (
                <div className="mt-3.5 rounded-xl border border-line bg-white/[0.02] px-3 py-2.5">
                  <div className="mb-1.5 font-mono text-[9px] uppercase tracking-[0.24em] text-ink-mute">
                    metric comparison
                  </div>
                  {(request.matched_metrics ?? []).map((m, i) => (
                    <div
                      key={i}
                      className="flex items-baseline justify-between gap-3 font-mono text-[11px]"
                    >
                      <span className="text-ink-dim">{m.name}</span>
                      <span className="tabular-nums">
                        <span className="text-accent">{String(m.value)}</span>
                        <span className="text-ink-mute"> vs </span>
                        <span className="text-ink-dim">{String(m.paper_value)}</span>
                        {m.direction && (
                          <span
                            className={
                              m.direction === "better"
                                ? " text-good"
                                : m.direction === "worse"
                                  ? " text-bad"
                                  : " text-ink-mute"
                            }
                          >
                            {" "}
                            {m.direction}
                          </span>
                        )}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              <p className="mt-3.5 text-[11.5px] leading-relaxed text-ink-mute">
                Approve to continue — the answer will present this comparison and
                treat your numbers as fixed. Reject to stop the run.
              </p>

              {error && (
                <p className="mt-3 rounded-lg border border-bad/30 bg-bad/10 px-2.5 py-2 text-[11px] leading-relaxed text-bad">
                  {error}
                </p>
              )}

              <div className="mt-4 flex items-center gap-2">
                <button
                  type="button"
                  disabled={resuming}
                  onClick={() => void resumeReview(true)}
                  className="flex-1 rounded-lg bg-ink px-3 py-2.5 font-mono text-[10.5px] uppercase tracking-[0.18em] text-[#050507] transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {resuming ? "resuming…" : "approve & continue"}
                </button>
                <button
                  type="button"
                  disabled={resuming}
                  onClick={() => void resumeReview(false)}
                  className="rounded-lg border border-line px-3.5 py-2.5 font-mono text-[10.5px] uppercase tracking-[0.18em] text-ink-dim transition hover:border-bad/40 hover:text-bad disabled:cursor-not-allowed disabled:opacity-40"
                >
                  reject
                </button>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
