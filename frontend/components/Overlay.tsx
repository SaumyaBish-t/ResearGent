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

  // The overlay uses framer-motion (animated inline transforms) whose
  // server/client markup differs, which trips React hydration. Since this is
  // an interactive client app (the WebGL scene is already ssr:false), gate the
  // overlay to client-only render after mount — eliminates the mismatch.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return null;

  return (
    <div className="pointer-events-none absolute inset-0 z-10">
      <header className="absolute right-5 top-4 text-right">
        <div className="text-sm font-semibold tracking-widest text-slate-200">
          RESEARGENT
        </div>
        <div className="text-[10px] uppercase tracking-[0.3em] text-slate-500">
          agentic network
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
