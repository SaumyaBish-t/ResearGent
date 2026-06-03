"use client";

import { useEffect, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useAgentStore } from "@/lib/store";

const LEVEL_COLOR: Record<string, string> = {
  info: "text-slate-300",
  success: "text-good",
  warn: "text-warn",
  error: "text-bad",
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
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [logs.length]);

  const open = running || finished;

  return (
    <AnimatePresence>
      {open && (
        <motion.aside
          initial={{ x: -380, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: -380, opacity: 0 }}
          transition={{ type: "spring", stiffness: 200, damping: 26 }}
          className="glass pointer-events-auto absolute left-4 top-4 flex h-[calc(100%-2rem)] w-[340px] flex-col rounded-2xl"
        >
          <header className="flex items-center justify-between border-b border-edge/60 px-4 py-3">
            <div className="text-xs font-semibold uppercase tracking-widest text-accent">
              execution trace
            </div>
            {runId && (
              <span className="text-[10px] text-slate-500">run {runId}</span>
            )}
          </header>

          {confidenceLabel && (
            <div className="flex items-center gap-3 border-b border-edge/60 px-4 py-2 text-xs">
              <span className="text-slate-400">confidence</span>
              <span
                className={
                  confidenceLabel === "high"
                    ? "text-good"
                    : confidenceLabel === "medium"
                      ? "text-warn"
                      : "text-bad"
                }
              >
                {confidenceLabel}
                {typeof confidenceScore === "number" &&
                  ` · ${confidenceScore.toFixed(2)}`}
              </span>
            </div>
          )}

          <div
            ref={scrollRef}
            className="flex-1 space-y-1 overflow-y-auto px-4 py-3 text-[12px] leading-relaxed"
          >
            {logs.length === 0 && (
              <div className="text-slate-500">waiting for events…</div>
            )}
            {logs.map((l) => (
              <div key={l.id} className="flex gap-2">
                <span className="shrink-0 text-slate-600">
                  {new Date(l.ts).toLocaleTimeString([], {
                    hour12: false,
                    minute: "2-digit",
                    second: "2-digit",
                  })}
                </span>
                <span className="shrink-0 text-accent/70">[{l.node}]</span>
                <span className={LEVEL_COLOR[l.level] ?? "text-slate-300"}>
                  {l.message}
                </span>
              </div>
            ))}
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
