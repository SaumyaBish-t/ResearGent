"use client";

import { useEffect } from "react";

type Props = { onClose: () => void };

const retrievalRate = 19 / 24;
const retrievalTarget = 20 / 24;

function MetricBar({
  label,
  value,
  detail,
  target,
  pending = false,
}: {
  label: string;
  value: number;
  detail: string;
  target?: number;
  pending?: boolean;
}) {
  return (
    <div className="space-y-2.5">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-sm text-ink/90">{label}</span>
        <span className={`shrink-0 font-mono text-xs ${pending ? "text-amber-300" : "text-accent"}`}>
          {detail}
        </span>
      </div>
      <div className="relative h-3 overflow-visible rounded-full bg-white/[0.07]">
        {!pending && (
          <div
            className="h-full rounded-full bg-gradient-to-r from-cyan-500/70 to-emerald-400"
            style={{ width: `${value * 100}%` }}
          />
        )}
        {target !== undefined && (
          <div
            className="absolute -top-1 h-5 border-l border-dashed border-white/75"
            style={{ left: `${target * 100}%` }}
            title={`Target: ${Math.round(target * 100)}%`}
          />
        )}
      </div>
      {target !== undefined && (
        <div className="flex justify-between font-mono text-[9px] uppercase tracking-[0.16em] text-ink-mute">
          <span>{pending ? "human review pending" : "measured baseline"}</span>
          <span>target {target === retrievalTarget ? "20/24" : ">75%"}</span>
        </div>
      )}
    </div>
  );
}

export default function BenchmarkPanel({ onClose }: Props) {
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  return (
    <div className="pointer-events-auto fixed inset-0 z-[80] flex items-center justify-center bg-black/65 p-4 backdrop-blur-sm">
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="benchmark-title"
        className="glass-bright max-h-[min(680px,88vh)] w-[min(620px,94vw)] overflow-y-auto rounded-2xl p-5 sm:p-7"
      >
        <header className="flex items-start justify-between gap-4">
          <div>
            <div className="mb-2 font-mono text-[9px] uppercase tracking-[0.32em] text-ink-mute">
              Labeled evaluation · 24 questions
            </div>
            <h2 id="benchmark-title" className="text-lg font-semibold text-ink">
              Research quality snapshot
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close evaluation panel"
            className="rounded-lg border border-line px-3 py-1.5 font-mono text-xs text-ink-dim transition hover:text-ink"
          >
            CLOSE
          </button>
        </header>

        <div className="mt-7 space-y-7">
          <MetricBar
            label="Expected paper found in top 5"
            value={retrievalRate}
            detail="19 / 24 · 79.2%"
            target={retrievalTarget}
          />
          <MetricBar
            label="Answer claims matched to reference points"
            value={0}
            detail="Not measured"
            target={0.75}
            pending
          />
        </div>

        <div className="mt-7 grid grid-cols-2 gap-3">
          <div className="rounded-xl border border-line bg-white/[0.025] p-3">
            <div className="font-mono text-[9px] uppercase tracking-[0.2em] text-ink-mute">MRR</div>
            <div className="mt-1 text-xl font-medium text-ink">0.771</div>
          </div>
          <div className="rounded-xl border border-line bg-white/[0.025] p-3">
            <div className="font-mono text-[9px] uppercase tracking-[0.2em] text-ink-mute">Recall@5</div>
            <div className="mt-1 text-xl font-medium text-ink">79.2%</div>
          </div>
        </div>

        <p className="mt-5 text-xs leading-5 text-ink-dim">
          Retrieval result is a local BM25 baseline, not a live-system score. The 20/24 mark is the retrieval target. Answer matching needs human grading against the labeled reference points; the 75% mark is a target, not a measured result.
        </p>
        <div className="mt-3 font-mono text-[9px] uppercase tracking-[0.18em] text-ink-mute">
          Agentic AI · Time Series · Quant Finance · run 2026-11-13
        </div>
      </section>
    </div>
  );
}
