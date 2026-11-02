"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAgentStore } from "@/lib/store";
import type { UserResultsInput } from "@/lib/store";

/**
 * Results Ingestion panel (Phase 24).
 *
 * Lets the user declare their OWN experimental results before a run. Those
 * results are GROUND TRUTH: the backend stores the numbers verbatim and the
 * Generator is instructed to write around them without altering them.
 *
 * Once ingested, every subsequent run routes to /api/research-with-results,
 * so the Retriever searches for the RELATED PRIOR WORK and the Novelty Check
 * has something to compare the user's numbers against.
 */

interface MetricRow {
  name: string;
  value: string;
  unit: string;
}

const EMPTY_ROW: MetricRow = { name: "", value: "", unit: "" };

export default function ResultsPanel() {
  const userResults = useAgentStore((s) => s.userResults);
  const ingesting = useAgentStore((s) => s.ingestingResults);
  const resultsError = useAgentStore((s) => s.resultsError);
  const ingestResults = useAgentStore((s) => s.ingestResults);
  const clearResults = useAgentStore((s) => s.clearResults);
  const running = useAgentStore((s) => s.running);

  const [open, setOpen] = useState(false);
  const [summary, setSummary] = useState("");
  const [approach, setApproach] = useState("");
  const [dataset, setDataset] = useState("");
  const [methodology, setMethodology] = useState("");
  const [rows, setRows] = useState<MetricRow[]>([{ ...EMPTY_ROW }]);

  const setRow = (i: number, patch: Partial<MetricRow>) =>
    setRows((rs) => rs.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));

  const submit = async () => {
    const metrics = rows
      .filter((r) => r.name.trim() && r.value.trim())
      .map((r) => ({
        name: r.name.trim(),
        value: r.value.trim(),
        ...(r.unit.trim() ? { unit: r.unit.trim() } : {}),
      }));

    const payload: UserResultsInput = {
      summary: summary.trim(),
      metrics,
      ...(approach.trim() ? { approach: approach.trim() } : {}),
      ...(dataset.trim() ? { dataset: dataset.trim() } : {}),
      ...(methodology.trim() ? { methodology_notes: methodology.trim() } : {}),
    };

    const ok = await ingestResults(payload);
    if (ok) setOpen(false);
  };

  // ---- Ingested state: show what will actually be compared ----
  if (userResults) {
    return (
      <div className="pointer-events-auto absolute bottom-24 left-5 z-20 w-[min(340px,86vw)]">
        <div className="glass rounded-xl px-3.5 py-3">
          <div className="mb-2 flex items-center justify-between">
            <span className="font-mono text-[9px] uppercase tracking-[0.28em] text-good">
              your results
            </span>
            <button
              type="button"
              onClick={clearResults}
              disabled={running}
              className="font-mono text-[9px] uppercase tracking-widest text-ink-mute transition hover:text-ink disabled:opacity-40"
            >
              clear
            </button>
          </div>

          <p className="mb-2 line-clamp-2 text-[11.5px] leading-relaxed text-ink-dim">
            {userResults.summary}
          </p>

          <div className="space-y-0.5">
            {(userResults.metrics ?? []).map((m, i) => (
              <div
                key={i}
                className="flex items-baseline justify-between gap-2 font-mono text-[10.5px]"
              >
                <span className="text-ink-mute">{m.name}</span>
                {/* Printed VERBATIM — the backend never rounds this. */}
                <span className="text-accent">
                  {String(m.value)}
                  {m.unit ? ` ${m.unit}` : ""}
                </span>
              </div>
            ))}
          </div>

          {(userResults.search_queries ?? []).length > 0 && (
            <div className="mt-2 border-t border-line pt-2">
              <div className="mb-1 font-mono text-[9px] uppercase tracking-[0.24em] text-ink-mute">
                will search for
              </div>
              {(userResults.search_queries ?? []).slice(0, 3).map((q, i) => (
                <div key={i} className="truncate font-mono text-[10px] text-ink-dim">
                  · {q}
                </div>
              ))}
            </div>
          )}

          <p className="mt-2 font-mono text-[9px] leading-relaxed text-ink-mute">
            next run compares these against the prior work
          </p>
        </div>
      </div>
    );
  }

  // ---- Collapsed toggle ----
  return (
    <div className="pointer-events-auto absolute bottom-24 left-5 z-20 w-[min(340px,86vw)]">
      <AnimatePresence initial={false}>
        {!open ? (
          <motion.button
            key="toggle"
            type="button"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setOpen(true)}
            disabled={running}
            className="glass rounded-xl px-3.5 py-2.5 font-mono text-[10px] uppercase tracking-[0.2em] text-ink-dim transition hover:text-ink disabled:opacity-40"
          >
            + add your own results
          </motion.button>
        ) : (
          <motion.div
            key="form"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            className="glass-bright rounded-xl px-3.5 py-3"
          >
            <div className="mb-2 flex items-center justify-between">
              <span className="font-mono text-[9px] uppercase tracking-[0.28em] text-ink-mute">
                your results
              </span>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="font-mono text-[9px] uppercase tracking-widest text-ink-mute transition hover:text-ink"
              >
                close
              </button>
            </div>

            <p className="mb-3 text-[10.5px] leading-relaxed text-ink-mute">
              Declare what you measured. These numbers are treated as fact —
              the answer is written around them and never alters them.
            </p>

            <div className="space-y-2">
              <input
                value={summary}
                onChange={(e) => setSummary(e.target.value)}
                placeholder="What did you do? (required)"
                className="w-full rounded-lg border border-line bg-white/[0.03] px-2.5 py-1.5 text-[12px] text-ink placeholder:text-ink-mute/70 focus:border-accent/40 focus:outline-none"
              />
              <input
                value={approach}
                onChange={(e) => setApproach(e.target.value)}
                placeholder="Approach / method (defaults to summary)"
                className="w-full rounded-lg border border-line bg-white/[0.03] px-2.5 py-1.5 text-[12px] text-ink placeholder:text-ink-mute/70 focus:border-accent/40 focus:outline-none"
              />
              <input
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                placeholder="Dataset (e.g. WMT14 en-de)"
                className="w-full rounded-lg border border-line bg-white/[0.03] px-2.5 py-1.5 text-[12px] text-ink placeholder:text-ink-mute/70 focus:border-accent/40 focus:outline-none"
              />
              <input
                value={methodology}
                onChange={(e) => setMethodology(e.target.value)}
                placeholder="Methodology notes (optional)"
                className="w-full rounded-lg border border-line bg-white/[0.03] px-2.5 py-1.5 text-[12px] text-ink placeholder:text-ink-mute/70 focus:border-accent/40 focus:outline-none"
              />
            </div>

            {/* Metrics — name + value, the comparison keys */}
            <div className="mt-3">
              <div className="mb-1.5 font-mono text-[9px] uppercase tracking-[0.24em] text-ink-mute">
                metrics
              </div>
              <div className="space-y-1.5">
                {rows.map((row, i) => (
                  <div key={i} className="flex items-center gap-1.5">
                    <input
                      value={row.name}
                      onChange={(e) => setRow(i, { name: e.target.value })}
                      placeholder="BLEU-4"
                      className="min-w-0 flex-1 rounded-lg border border-line bg-white/[0.03] px-2 py-1 font-mono text-[11px] text-ink placeholder:text-ink-mute/60 focus:border-accent/40 focus:outline-none"
                    />
                    <input
                      value={row.value}
                      onChange={(e) => setRow(i, { value: e.target.value })}
                      placeholder="38.1"
                      className="w-16 shrink-0 rounded-lg border border-line bg-white/[0.03] px-2 py-1 font-mono text-[11px] text-accent placeholder:text-ink-mute/60 focus:border-accent/40 focus:outline-none"
                    />
                    <input
                      value={row.unit}
                      onChange={(e) => setRow(i, { unit: e.target.value })}
                      placeholder="%"
                      className="w-12 shrink-0 rounded-lg border border-line bg-white/[0.03] px-2 py-1 font-mono text-[11px] text-ink-dim placeholder:text-ink-mute/60 focus:border-accent/40 focus:outline-none"
                    />
                    {rows.length > 1 && (
                      <button
                        type="button"
                        onClick={() => setRows((rs) => rs.filter((_, x) => x !== i))}
                        className="shrink-0 font-mono text-[12px] text-ink-mute transition hover:text-ink"
                        aria-label="remove metric"
                      >
                        ×
                      </button>
                    )}
                  </div>
                ))}
              </div>
              <button
                type="button"
                onClick={() => setRows((rs) => [...rs, { ...EMPTY_ROW }])}
                className="mt-1.5 font-mono text-[9px] uppercase tracking-widest text-ink-mute transition hover:text-ink"
              >
                + metric
              </button>
            </div>

            {resultsError && (
              <p className="mt-2 rounded-lg border border-bad/30 bg-bad/10 px-2 py-1.5 text-[10.5px] leading-relaxed text-bad">
                {resultsError}
              </p>
            )}

            <button
              type="button"
              onClick={submit}
              disabled={ingesting || !summary.trim()}
              className="mt-3 w-full rounded-lg bg-ink px-3 py-2 font-mono text-[10px] uppercase tracking-[0.18em] text-[#050507] transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-40"
            >
              {ingesting ? "checking…" : "add results"}
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
