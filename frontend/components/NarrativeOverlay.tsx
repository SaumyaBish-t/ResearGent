"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { STORY, storyNodeAt } from "@/lib/graph-config";
import { useAgentStore } from "@/lib/store";

/**
 * Intro chrome that lives on the HTML layer: the hero title, a scroll hint, and
 * the 01–07 progress rail. The per-node DESCRIPTIONS are spatially anchored to
 * the 3D nodes via drei <Html> (see AgentNode).
 */
export default function NarrativeOverlay() {
  const scrollProgress = useAgentStore((s) => s.scrollProgress);
  const reduced = useReducedMotion();

  const activeNode = storyNodeAt(scrollProgress);
  const storyStops = STORY.filter((s) => s.node);
  const activeStepIdx = storyStops.findIndex((s) => s.node === activeNode);

  return (
    <>
      {/* Progress rail (right edge) — 01..07 */}
      <div className="pointer-events-none absolute right-6 top-1/2 hidden -translate-y-1/2 flex-col gap-4 md:flex">
        <div className="mb-1 font-mono text-[9px] uppercase tracking-[0.32em] text-ink-mute">
          phase
        </div>
        {storyStops.map((s, i) => {
          const active = i === activeStepIdx;
          const passed = i < activeStepIdx;
          return (
            <div
              key={s.step}
              className="flex items-center justify-end gap-2.5"
            >
              <span
                className={`font-mono text-[10px] tabular-nums tracking-widest transition-all duration-300 ${
                  active
                    ? "text-accent"
                    : passed
                      ? "text-ink-dim"
                      : "text-ink-mute/50"
                }`}
              >
                {s.step}
              </span>
              <span
                className={`h-px transition-all duration-500 ease-out ${
                  active
                    ? "w-9 bg-accent shadow-[0_0_8px_rgba(34,211,238,0.6)]"
                    : passed
                      ? "w-5 bg-ink-dim/60"
                      : "w-3 bg-ink-mute/40"
                }`}
              />
            </div>
          );
        })}
      </div>

      {/* Hero title (only near the top) */}
      <AnimatePresence>
        {scrollProgress < 0.1 && (
          <motion.div
            key="hero"
            // x: "-50%" must live in animate (not just className) — framer
            // composes a fresh `transform` for the y-tween that would otherwise
            // wipe out Tailwind's `-translate-x-1/2`, leaving the hero
            // anchored at its left edge.
            initial={{ opacity: 0, y: 8, x: "-50%" }}
            animate={{ opacity: 1, y: 0, x: "-50%" }}
            exit={{ opacity: 0, x: "-50%", transition: { duration: 0.22 } }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="pointer-events-none absolute left-1/2 top-[15%] text-center"
          >
            {/* eyebrow */}
            <div className="mb-3 flex items-center justify-center gap-2 font-mono text-[10px] uppercase tracking-[0.42em] text-ink-mute">
              <span className="h-px w-6 bg-line" />
              corrective-rag · multi-agent
              <span className="h-px w-6 bg-line" />
            </div>

            <h1 className="bg-gradient-to-b from-white via-white to-slate-400 bg-clip-text text-[44px] font-semibold tracking-tight text-transparent sm:text-[56px]">
              ResearGent
            </h1>

            <p className="mx-auto mt-3 max-w-md text-[13.5px] leading-relaxed text-ink-dim">
              A living map of agents that retrieve, critique, and rewrite their
              way to a grounded answer.
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Scroll hint */}
      <AnimatePresence>
        {scrollProgress < 0.06 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
            className="pointer-events-none absolute bottom-28 left-1/2 -translate-x-1/2 text-center"
          >
            <div className="font-mono text-[10px] uppercase tracking-[0.32em] text-ink-mute">
              scroll to detonate the core
            </div>
            {!reduced && (
              <motion.div
                animate={{ y: [0, 8, 0], opacity: [0.4, 1, 0.4] }}
                transition={{
                  repeat: Infinity,
                  duration: 1.8,
                  ease: "easeInOut",
                }}
                className="mx-auto mt-3 h-4 w-px bg-gradient-to-b from-accent to-transparent"
              />
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
