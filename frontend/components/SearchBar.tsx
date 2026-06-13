"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { useAgentStore } from "@/lib/store";

export default function SearchBar() {
  const running = useAgentStore((s) => s.running);
  const hasQueried = useAgentStore((s) => s.hasQueried);
  const finished = useAgentStore((s) => s.finished);
  const startRun = useAgentStore((s) => s.startRun);
  const reset = useAgentStore((s) => s.reset);
  // Docked once a query runs OR as soon as the user scrolls past the hero, so
  // the bar never overlaps the spotlighted nodes during the narrative.
  const scrolled = useAgentStore((s) => s.scrollProgress > 0.05);
  const [value, setValue] = useState("");
  const [focused, setFocused] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Keyboard shortcut: "/" jumps focus into the bar (classic dev tool gesture).
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
    if (!value.trim() || running) return;
    startRun(value.trim());
  };

  const docked = hasQueried || scrolled;

  const statusColor = running
    ? "bg-accent"
    : finished
      ? "bg-good"
      : "bg-ink-mute/60";

  return (
    <motion.form
      onSubmit={submit}
      initial={false}
      animate={{
        top: docked ? "93%" : "47%",
        scale: docked ? 1 : 1.1,
        x: "-50%",
        y: "-50%",
      }}
      transition={{ type: "spring", stiffness: 210, damping: 28 }}
      style={{ position: "fixed", left: "50%" }}
      className={`${focused ? "glass-bright" : "glass"} pointer-events-auto z-30 flex w-[min(680px,92vw)] items-center gap-3 rounded-2xl px-4 py-3 transition-shadow duration-300`}
    >
      {/* Status indicator with optional ping. */}
      <div className="relative flex h-3 w-3 shrink-0 items-center justify-center">
        <span className={`h-2 w-2 rounded-full ${statusColor} dot-glow`} style={{ color: running ? "#22d3ee" : finished ? "#34d399" : "transparent" }} />
        {running && (
          <span className="absolute inset-0 animate-pulse-ring rounded-full bg-accent/40" />
        )}
      </div>

      {/* Soft prompt prefix to anchor the input as a "command". */}
      <span className="select-none font-mono text-[11px] uppercase tracking-[0.22em] text-ink-mute">
        ask
      </span>

      <input
        ref={inputRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        disabled={running}
        placeholder={
          hasQueried
            ? "Ask a follow-up…"
            : "what would you like to research?"
        }
        autoFocus
        className="flex-1 bg-transparent px-1 py-1.5 text-[15px] text-ink placeholder:text-ink-mute/70 focus:outline-none disabled:opacity-60"
      />

      {!value && !running && (
        <span className="hidden items-center gap-1 sm:flex">
          <span className="kbd">/</span>
          <span className="font-mono text-[10px] uppercase tracking-widest text-ink-mute">
            focus
          </span>
        </span>
      )}

      {finished && !running && (
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
        disabled={running || !value.trim()}
        className={`group relative overflow-hidden rounded-lg px-4 py-2 font-mono text-[11px] uppercase tracking-[0.18em] transition disabled:cursor-not-allowed disabled:opacity-40 ${
          running
            ? "bg-accent/20 text-accent"
            : value.trim()
              ? "bg-ink text-[#050507] hover:bg-white"
              : "bg-white/[0.06] text-ink-dim"
        }`}
      >
        <span className="relative z-10">
          {running ? "thinking" : "research"}
        </span>
        {running && (
          <span className="shimmer absolute inset-x-0 bottom-0 h-px" />
        )}
      </button>
    </motion.form>
  );
}
