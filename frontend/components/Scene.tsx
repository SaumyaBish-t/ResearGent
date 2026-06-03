"use client";

import { useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { ScrollControls, Stars } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import {
  NODES,
  SCROLL_PAGES,
  STORY_NODE_ORDER,
  TOOLTIP_BY_NODE,
  revealedCount,
  storyNodeAt,
} from "@/lib/graph-config";
import { useAgentStore } from "@/lib/store";
import AgentNode from "./AgentNode";
import Edges from "./Edges";
import Particles from "./Particles";
import { DashboardCamera, ScrollCamera, ScrollReporter } from "./CameraRig";

/** Shared geometry: neural-core/particles + edges + the 11 agent nodes. */
function Network({ mode, reduced }: { mode: "scroll" | "dashboard"; reduced: boolean }) {
  const nodeStatuses = useAgentStore((s) => s.nodeStatuses);
  const activeEdge = useAgentStore((s) => s.activeEdge);
  // Both selectors return primitives that change only a handful of times, so
  // Network re-renders ~7× across the whole scroll — never per frame.
  const activeStoryNode = useAgentStore((s) =>
    mode === "scroll" ? storyNodeAt(s.scrollProgress) : null,
  );
  const revealStep = useAgentStore((s) =>
    mode === "scroll" ? revealedCount(s.scrollProgress) : 99,
  );

  const isRevealed = (id: (typeof NODES)[number]["id"]) => {
    if (mode === "dashboard") return true;
    const si = STORY_NODE_ORDER.indexOf(id);
    if (si >= 0) return si < revealStep; // story node emerged
    return revealStep >= 4; // supporting nodes condense in mid-journey
  };
  const edgesVisible = mode === "dashboard" || revealStep > 0;

  return (
    <group>
      <Particles mode={mode} reduced={reduced} />
      {edgesVisible && <Edges activeEdge={activeEdge} />}
      {NODES.map((n) => (
        <AgentNode
          key={n.id}
          label={n.label}
          sublabel={n.sublabel}
          position={n.position}
          status={nodeStatuses[n.id]}
          accent={n.accent}
          shape={n.shape}
          tooltip={TOOLTIP_BY_NODE[n.id]}
          showTooltip={mode === "scroll" && activeStoryNode === n.id}
          mode={mode}
          revealed={isRevealed(n.id)}
        />
      ))}
    </group>
  );
}

function Lights() {
  return (
    <>
      <ambientLight intensity={0.35} />
      <directionalLight position={[6, 10, 8]} intensity={0.8} color="#dfe6f0" />
      <pointLight position={[-8, -4, -6]} intensity={30} color="#3a4a7a" />
    </>
  );
}

export default function Scene() {
  const hasQueried = useAgentStore((s) => s.hasQueried);
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  return (
    <div className="absolute inset-0">
      <Canvas
        camera={{ position: [0, 0, 17], fov: 55, near: 0.1, far: 200 }}
        dpr={[1, 2]}
        gl={{ antialias: true }}
      >
        <color attach="background" args={["#040409"]} />
        <fog attach="fog" args={["#040409", 18, 52]} />

        <Lights />
        <Stars radius={90} depth={50} count={1200} factor={2.5} saturation={0} fade speed={0.4} />

        {hasQueried ? (
          <>
            <Network mode="dashboard" reduced={reduced} />
            <DashboardCamera />
          </>
        ) : (
          <ScrollControls pages={SCROLL_PAGES} damping={0.25}>
            <ScrollReporter />
            <ScrollCamera reduced={reduced} />
            <Network mode="scroll" reduced={reduced} />
          </ScrollControls>
        )}

        {/* Razor-sharp glow on emissive nodes + additive particles. */}
        <EffectComposer>
          <Bloom
            intensity={0.85}
            luminanceThreshold={0.18}
            luminanceSmoothing={0.32}
            mipmapBlur
            radius={0.7}
          />
        </EffectComposer>
      </Canvas>
    </div>
  );
}
