"use client";

import { motion } from "framer-motion";
import { useAgentStore } from "@/lib/store";

/**
 * Two-tab mode selector: Research | Review
 * Sits beside the status indicator in the header.
 */
export default function ModeSwitcher() {
  const mode = useAgentStore((s) => s.mode);
  const setMode = useAgentStore((s) => s.setMode);
  const running = useAgentStore((s) => s.running);
  const reviewRunning = useAgentStore((s) => s.reviewRunning);
  const disabled = running || reviewRunning;

  const tabs = [
    { id: "research" as const, label: "Research" },
    { id: "review" as const, label: "Review" },
  ];

  return (
    <div className="pointer-events-auto flex items-center gap-0.5 rounded-lg bg-white/[0.03] p-0.5 border border-line">
      {tabs.map((tab) => {
        const active = mode === tab.id;
        return (
          <button
            key={tab.id}
            type="button"
            disabled={disabled}
            onClick={() => setMode(tab.id)}
            className={`rounded-md px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.12em] transition ${
              active
                ? "bg-accent/20 text-accent"
                : "text-ink-dim hover:text-ink"
            } disabled:opacity-40 disabled:cursor-not-allowed`}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}