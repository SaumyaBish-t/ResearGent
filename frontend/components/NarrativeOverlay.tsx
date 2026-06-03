"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { STORY, storyNodeAt } from "@/lib/graph-config";
import { useAgentStore } from "@/lib/store";

/**
 * Intro chrome that lives on the HTML layer: the hero title, a scroll hint, and
 * the 01–07 progress rail. The per-node DESCRIPTIONS are no longer here — they
 * are spatially anchored to the 3D nodes via drei <Html> (see AgentNode).
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
      <div className="pointer-events-none absolute right-6 top-1/2 hidden -translate-y-1/2 flex-col gap-3 md:flex">
        {storyStops.map((s, i) => {
          const active = i === activeStepIdx;
          return (
            <div key={s.step} className="flex items-center justify-end gap-2">
              <span
                className={`font-mono text-[10px] tabular-nums tracking-widest transition-colors ${
                  active ? "text-accent" : "text-slate-600"
                }`}
              >
                {s.step}
              </span>
              <span
                className={`h-px transition-all ${
                  active ? "w-8 bg-accent" : "w-4 bg-slate-700"
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
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, transition: { duration: 0.18 } }}
            className="pointer-events-none absolute left-1/2 top-[16%] -translate-x-1/2 text-center"
          >
            <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
              ResearGent
            </h1>
            <p className="mt-2 font-mono text-xs uppercase tracking-[0.35em] text-accent/70">
              multi-agent corrective-rag core
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
            className="pointer-events-none absolute bottom-28 left-1/2 -translate-x-1/2 text-center"
          >
            <div className="font-mono text-[11px] uppercase tracking-[0.3em] text-slate-500">
              scroll to detonate the core
            </div>
            {!reduced && (
              <motion.div
                animate={{ y: [0, 7, 0] }}
                transition={{ repeat: Infinity, duration: 1.6, ease: "easeInOut" }}
                className="mx-auto mt-2 h-3 w-px bg-accent/60"
              />
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
