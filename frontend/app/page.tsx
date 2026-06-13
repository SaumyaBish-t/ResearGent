"use client";

import dynamic from "next/dynamic";
import Overlay from "@/components/Overlay";

// The 3D scene touches `window`/WebGL, so it must never render on the server.
const Scene = dynamic(() => import("@/components/Scene"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <div className="relative h-1 w-40 overflow-hidden rounded-full bg-white/[0.04]">
          <span className="shimmer absolute inset-y-0 left-0 w-1/3" />
        </div>
        <div className="font-mono text-[10px] uppercase tracking-[0.32em] text-ink-mute">
          initializing agentic network
        </div>
      </div>
    </div>
  ),
});

export default function Home() {
  return (
    <main className="relative h-full w-full bg-base">
      {/* Phase 3: the 3D agentic network */}
      <Scene />
      {/* Phase 4: the 2D UI overlay (search, logs, result modal) */}
      <Overlay />
    </main>
  );
}
