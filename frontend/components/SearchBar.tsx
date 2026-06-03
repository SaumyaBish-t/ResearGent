"use client";

import { useState } from "react";
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

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!value.trim() || running) return;
    startRun(value.trim());
  };

  const docked = hasQueried || scrolled;

  return (
    <motion.form
      // NOTE: no layoutId. The center↔dock morph is driven entirely by the
      // `animate` prop below. A layoutId here makes framer's layout projection
      // override the translateX(-50%), shifting the docked bar off-center.
      onSubmit={submit}
      initial={false}
      animate={{
        top: docked ? "93%" : "47%",
        scale: docked ? 1 : 1.12,
        // Centering translate lives in `animate` (not `style`): framer-motion
        // composes the transform here, so it isn't dropped when it animates
        // `scale`. Keeping x/y in style left the transform as `none` → the bar
        // anchored its left edge at center and drifted right.
        x: "-50%",
        y: "-50%",
      }}
      transition={{ type: "spring", stiffness: 210, damping: 28 }}
      style={{ position: "fixed", left: "50%" }}
      className="glass pointer-events-auto z-30 flex w-[min(680px,92vw)] items-center gap-3 rounded-2xl px-4 py-3"
    >
      <div
        className={`h-2.5 w-2.5 shrink-0 rounded-full ${
          running
            ? "animate-pulse bg-white"
            : finished
              ? "bg-good"
              : "bg-slate-600"
        }`}
        title={running ? "running" : finished ? "done" : "idle"}
      />
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        disabled={running}
        placeholder={
          hasQueried
            ? "Ask a follow-up…"
            : "Ask ResearGent a research question…"
        }
        autoFocus
        className="flex-1 bg-transparent px-2 py-1.5 text-base text-slate-100 placeholder:text-slate-500 focus:outline-none disabled:opacity-60"
      />
      {finished && !running && (
        <button
          type="button"
          onClick={() => {
            reset();
            setValue("");
          }}
          className="rounded-xl border border-white/15 px-3 py-2 text-xs text-slate-300 transition hover:border-white/40 hover:text-white"
        >
          back
        </button>
      )}
      <button
        type="submit"
        disabled={running || !value.trim()}
        className="rounded-xl bg-white px-4 py-2 text-sm font-semibold text-[#050507] transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {running ? "running…" : "research"}
      </button>
    </motion.form>
  );
}
