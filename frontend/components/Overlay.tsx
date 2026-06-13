"use client";

import { useEffect, useState } from "react";
import { useAgentStore } from "@/lib/store";
import SearchBar from "./SearchBar";
import LogPanel from "./LogPanel";
import ResultModal from "./ResultModal";
import NarrativeOverlay from "./NarrativeOverlay";

/**
 * The 2D HTML overlay above the WebGL canvas. Wrapper is `pointer-events-none`
 * so the scene stays interactive; widgets re-enable pointer events themselves.
 *
 * Two phases, switched by `hasQueried`:
 *   - intro  → scrollytelling narrative cards
 *   - active → live execution trace + result modal
 * The SearchBar is rendered in BOTH phases (it morphs between them via its
 * shared `layoutId`), so it lives outside the conditional.
 */
export default function Overlay() {
  const hasQueried = useAgentStore((s) => s.hasQueried);
  const running = useAgentStore((s) => s.running);
  const finished = useAgentStore((s) => s.finished);

  // The overlay uses framer-motion (animated inline transforms) whose
  // server/client markup differs, which trips React hydration. Gate to
  // client-only render after mount — eliminates the mismatch.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return null;

  const statusLabel = running ? "running" : finished ? "ready" : "idle";
  const statusColor = running
    ? "text-accent"
    : finished
      ? "text-good"
      : "text-ink-mute";

  return (
    <div className="pointer-events-none absolute inset-0 z-10">
      {/* Top-left wordmark — quiet, monospace, tabular. */}
      <header className="absolute left-5 top-4 flex items-center gap-3">
        <div className="relative h-2 w-2">
          <span
            className={`absolute inset-0 rounded-full ${running ? "bg-accent" : finished ? "bg-good" : "bg-ink-mute/60"}`}
          />
          {running && (
            <span className="absolute inset-0 animate-pulse-ring rounded-full bg-accent/40" />
          )}
        </div>
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-[11px] font-medium tracking-[0.22em] text-ink/90">
            RESEARGENT
          </span>
          <span className="font-mono text-[9px] uppercase tracking-[0.32em] text-ink-mute">
            v1
          </span>
        </div>
      </header>

      {/* Top-right meta — status + agentic-network subtitle. */}
      <header className="absolute right-5 top-4 flex items-center gap-3 text-right">
        <div className="font-mono text-[9px] uppercase tracking-[0.3em] text-ink-mute">
          agentic network
        </div>
        <span className="h-3 w-px bg-line" />
        <div className={`font-mono text-[10px] tracking-widest ${statusColor}`}>
          {statusLabel}
        </div>
      </header>

      {hasQueried ? (
        <>
          <LogPanel />
          <ResultModal />
        </>
      ) : (
        <NarrativeOverlay />
      )}

      {/* Persistent, morphing search bar (centered → docked). */}
      <SearchBar />
    </div>
  );
}
