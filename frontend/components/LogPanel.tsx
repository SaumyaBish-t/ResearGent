"use client";

import { useEffect, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useAgentStore } from "@/lib/store";

const LEVEL_COLOR: Record<string, string> = {
  info: "text-slate-200",
  success: "text-good",
  warn: "text-warn",
  error: "text-bad",
};

const LEVEL_GLYPH: Record<string, string> = {
  info: "·",
  success: "✓",
  warn: "!",
  error: "✕",
};

export default function LogPanel() {
  const logs = useAgentStore((s) => s.logs);
  const running = useAgentStore((s) => s.running);
  const finished = useAgentStore((s) => s.finished);
  const runId = useAgentStore((s) => s.runId);
  const confidenceScore = useAgentStore((s) => s.confidenceScore);
  const confidenceLabel = useAgentStore((s) => s.confidenceLabel);

  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [logs.length]);

  const open = running || finished;
  const confColor =
    confidenceLabel === "high"
      ? "text-good"
      : confidenceLabel === "medium"
        ? "text-warn"
        : "text-bad";

  return (
    <AnimatePresence>
      {open && (
        <motion.aside
          initial={{ x: -380, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: -380, opacity: 0 }}
          transition={{ type: "spring", stiffness: 200, damping: 26 }}
          className="glass pointer-events-auto absolute left-4 top-16 flex h-[calc(100%-5.5rem)] w-[360px] flex-col overflow-hidden rounded-2xl"
        >
          {/* Header — title + live run id */}
          <header className="flex items-center justify-between px-5 pt-4 pb-3">
            <div className="flex items-center gap-2">
              <span
                className={`h-1.5 w-1.5 rounded-full ${running ? "bg-accent dot-glow" : "bg-good"}`}
                style={{ color: running ? "#22d3ee" : "#34d399" }}
              />
              <div className="font-mono text-[10px] uppercase tracking-[0.28em] text-ink-dim">
                execution trace
              </div>
            </div>
            {runId && (
              <span className="font-mono text-[10px] tabular-nums text-ink-mute">
                {runId}
              </span>
            )}
          </header>

          {/* Confidence row — only when known */}
          {confidenceLabel && (
            <div className="mx-5 mb-2 flex items-center justify-between rounded-lg border border-line bg-white/[0.02] px-3 py-2">
              <span className="font-mono text-[10px] uppercase tracking-widest text-ink-mute">
                confidence
              </span>
              <span className={`font-mono text-xs tracking-wide ${confColor}`}>
                {confidenceLabel}
                {typeof confidenceScore === "number" && (
                  <span className="ml-2 text-ink-mute">·</span>
                )}
                {typeof confidenceScore === "number" && (
                  <span className="ml-2 tabular-nums">
                    {confidenceScore.toFixed(2)}
                  </span>
                )}
              </span>
            </div>
          )}

          <div className="hairline mx-5" />

          {/* Log stream */}
          <div
            ref={scrollRef}
            className="flex-1 space-y-1 overflow-y-auto px-5 py-3 font-mono text-[11.5px] leading-relaxed"
          >
            {logs.length === 0 && (
              <div className="py-4 text-center text-ink-mute">
                waiting for events
                <span className="caret" />
              </div>
            )}
            {logs.map((l) => (
              <div
                key={l.id}
                className="group grid grid-cols-[auto_auto_1fr] items-baseline gap-2 rounded px-1 py-[2px] transition hover:bg-white/[0.02]"
              >
                <span className="shrink-0 tabular-nums text-ink-mute/70">
                  {new Date(l.ts).toLocaleTimeString([], {
                    hour12: false,
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  })}
                </span>
                <span className="shrink-0 text-accent/80">
                  <span className="text-ink-mute">[</span>
                  {l.node}
                  <span className="text-ink-mute">]</span>
                </span>
                <span className={`break-words ${LEVEL_COLOR[l.level] ?? "text-slate-200"}`}>
                  <span className="mr-1 opacity-60">
                    {LEVEL_GLYPH[l.level] ?? "·"}
                  </span>
                  {l.message}
                </span>
              </div>
            ))}
          </div>

          {/* Footer — small status strip */}
          <footer className="border-t border-line px-5 py-2.5 font-mono text-[10px] uppercase tracking-[0.22em] text-ink-mute">
            <span>{logs.length}</span>
            <span className="ml-1">events</span>
            <span className="float-right">
              {running ? "live" : finished ? "complete" : "—"}
            </span>
          </footer>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
