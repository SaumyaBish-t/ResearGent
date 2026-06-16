"use client";

import { useEffect, useState } from "react";
import { useAgentStore } from "@/lib/store";
import HistorySidebar from "./HistorySidebar";
import LogPanel from "./LogPanel";
import NarrativeOverlay from "./NarrativeOverlay";
import PaywallModal from "./PaywallModal";
import ResultModal from "./ResultModal";
import SearchBar from "./SearchBar";
import SignInLanding from "./SignInLanding";
import UserMenu from "./UserMenu";

/**
 * The 2D HTML overlay above the WebGL canvas. Wrapper is `pointer-events-none`
 * so the scene stays interactive; widgets re-enable pointer events themselves.
 *
 * Three phases:
 *   - auth gate → SignInLanding (no session)
 *   - intro     → scrollytelling narrative cards
 *   - active    → live execution trace + result modal
 * SearchBar + UserMenu + HistorySidebar + PaywallModal render in every signed-in
 * phase (each component decides its own visibility).
 */
export default function Overlay() {
  const hasQueried = useAgentStore((s) => s.hasQueried);
  const running = useAgentStore((s) => s.running);
  const finished = useAgentStore((s) => s.finished);

  const authReady = useAgentStore((s) => s.authReady);
  const user = useAgentStore((s) => s.user);
  const bootstrap = useAgentStore((s) => s.bootstrap);

  // Gate to client-only render after mount — framer-motion's transforms differ
  // server vs client, which trips React hydration.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  useEffect(() => {
    if (mounted) void bootstrap();
  }, [mounted, bootstrap]);

  if (!mounted) return null;

  // ---- Auth gate ----
  if (!authReady) {
    return <div className="pointer-events-none absolute inset-0 z-10" />;
  }
  if (!user) {
    return (
      <div className="absolute inset-0 z-10">
        <SignInLanding />
      </div>
    );
  }

  const statusLabel = running ? "running" : finished ? "ready" : "idle";
  const statusColor = running
    ? "text-accent"
    : finished
      ? "text-good"
      : "text-ink-mute";

  return (
    <div className="pointer-events-none absolute inset-0 z-10">
      {/* Top-left wordmark */}
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

      {/* Top-right: status + user menu */}
      <header className="absolute right-5 top-4 flex items-center gap-3">
        <div className="font-mono text-[9px] uppercase tracking-[0.3em] text-ink-mute">
          agentic network
        </div>
        <span className="h-3 w-px bg-line" />
        <div className={`font-mono text-[10px] tracking-widest ${statusColor}`}>
          {statusLabel}
        </div>
        <span className="h-3 w-px bg-line" />
        <UserMenu />
      </header>

      {hasQueried ? (
        <>
          <LogPanel />
          <ResultModal />
        </>
      ) : (
        <NarrativeOverlay />
      )}

      {/* History sidebar — auto-hides during a live run. */}
      <HistorySidebar />

      {/* Persistent, morphing search bar (centered → docked). */}
      <SearchBar />

      {/* Paywall — only renders when paywall.open is true. */}
      <PaywallModal />
    </div>
  );
}
