"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { useAgentStore } from "@/lib/store";

const DOMAIN_CHIPS = [
  { id: "", label: "All" },
  { id: "agentic_ai", label: "Agentic AI" },
  { id: "quant_finance", label: "Quant Finance" },
  { id: "time_series", label: "Time-Series" },
] as const;

export default function SearchBar() {
  const running = useAgentStore((s) => s.running);
  const reviewRunning = useAgentStore((s) => s.reviewRunning);
  const hasQueried = useAgentStore((s) => s.hasQueried);
  const finished = useAgentStore((s) => s.finished);
  const selectedDomain = useAgentStore((s) => s.selectedDomain);
  const setDomain = useAgentStore((s) => s.setDomain);
  const mode = useAgentStore((s) => s.mode);
  const startRun = useAgentStore((s) => s.startRun);
  const startReview = useAgentStore((s) => s.startReview);
  const reset = useAgentStore((s) => s.reset);
  const scrolled = useAgentStore((s) => s.scrollProgress > 0.05);
  const [value, setValue] = useState("");
  const [focused, setFocused] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const busy = running || reviewRunning;
  const inReview = mode === "review";

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "/" && document.activeElement?.tagName !== "INPUT") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!value.trim() || busy) return;
    if (inReview) {
      startReview(value.trim());
    } else {
      startRun(value.trim());
    }
  };

  const docked = hasQueried || scrolled;

  const statusColor = busy
    ? "bg-accent"
    : finished
      ? "bg-good"
      : "bg-ink-mute/60";

  return (
    <motion.div
      initial={false}
      animate={{
        top: docked ? "93%" : "47%",
        scale: docked ? 1 : 1.1,
        x: "-50%",
        y: "-50%",
      }}
      transition={{ type: "spring", stiffness: 210, damping: 28 }}
      style={{ position: "fixed", left: "50%" }}
      className="pointer-events-auto z-30 flex w-[min(680px,92vw)] flex-col items-center gap-2"
    >
      {/* Domain chips — only in research mode */}
      {!inReview && (
        <div className="flex items-center gap-1.5">
          {DOMAIN_CHIPS.map((chip) => {
            const active = selectedDomain === chip.id;
            return (
              <button
                key={chip.id}
                type="button"
                onClick={() => setDomain(chip.id)}
                disabled={busy}
                className={`rounded-full px-3 py-1 font-mono text-[10.5px] uppercase tracking-[0.12em] transition ${
                  active
                    ? "bg-accent/20 text-accent border border-accent/40"
                    : "bg-white/[0.04] text-ink-dim border border-line hover:border-accent/30 hover:text-ink"
                } disabled:opacity-40`}
              >
                {chip.label}
              </button>
            );
          })}
        </div>
      )}

      <motion.form
        onSubmit={submit}
        className={`${focused ? "glass-bright" : "glass"} flex items-center gap-3 rounded-2xl px-4 py-3 transition-shadow duration-300`}
      >
        {/* Status indicator */}
        <div className="relative flex h-3 w-3 shrink-0 items-center justify-center">
          <span className={`h-2 w-2 rounded-full ${statusColor} dot-glow`} style={{ color: running ? "#22d3ee" : finished ? "#34d399" : "transparent" }} />
          {running && (
            <span className="absolute inset-0 animate-pulse-ring rounded-full bg-accent/40" />
          )}
        </div>

        <span className="select-none font-mono text-[11px] uppercase tracking-[0.22em] text-ink-mute">
          {inReview ? "review" : "ask"}
        </span>

        <input
          ref={inputRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          disabled={busy}
          placeholder={
            inReview
              ? "topic for literature review…"
              : hasQueried
                ? "Ask a follow-up…"
                : "what would you like to research?"
          }
          autoFocus
          className="flex-1 bg-transparent px-1 py-1.5 text-[15px] text-ink placeholder:text-ink-mute/70 focus:outline-none disabled:opacity-60"
        />

        {!value && !busy && (
          <span className="hidden items-center gap-1 sm:flex">
            <span className="kbd">/</span>
            <span className="font-mono text-[10px] uppercase tracking-widest text-ink-mute">
              focus
            </span>
          </span>
        )}

        {!inReview && finished && !running && (
          <button
            type="button"
            onClick={() => {
              reset();
              setValue("");
            }}
            className="rounded-lg border border-line px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-widest text-ink-dim transition hover:border-line hover:bg-white/[0.03] hover:text-ink"
          >
            new
          </button>
        )}

        <button
          type="submit"
          disabled={busy || !value.trim()}
          className={`group relative overflow-hidden rounded-lg px-4 py-2 font-mono text-[11px] uppercase tracking-[0.18em] transition disabled:cursor-not-allowed disabled:opacity-40 ${
            busy
              ? "bg-accent/20 text-accent"
              : value.trim()
                ? "bg-ink text-[#050507] hover:bg-white"
                : "bg-white/[0.06] text-ink-dim"
          }`}
        >
          <span className="relative z-10">
            {busy ? (inReview ? "writing" : "thinking") : inReview ? "generate" : "research"}
          </span>
          {busy && (
            <span className="shimmer absolute inset-x-0 bottom-0 h-px" />
          )}
        </button>
      </motion.form>
    </motion.div>
  );
}