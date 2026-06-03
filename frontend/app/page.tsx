"use client";

import dynamic from "next/dynamic";
import Overlay from "@/components/Overlay";

// The 3D scene touches `window`/WebGL, so it must never render on the server.
const Scene = dynamic(() => import("@/components/Scene"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center text-accent/70">
      initializing agentic network…
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
