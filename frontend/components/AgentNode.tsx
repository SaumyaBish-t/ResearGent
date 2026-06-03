"use client";

import { useMemo, useRef, type MutableRefObject } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import type { NodeStatus } from "@/lib/types";
import type { NodeAccent, NodeShape } from "@/lib/graph-config";

const ORIGIN = new THREE.Vector3(0, 0, 0);

// Portal target so <Html> escapes the <ScrollControls> scroll container.
// Inside ScrollControls the default portal is the scrolled element, which
// offsets every label/tooltip by scrollTop and drifts them off-screen. Render
// into document.body instead → positioned in stable viewport space.
const bodyPortal: MutableRefObject<HTMLElement> | undefined =
  typeof document !== "undefined"
    ? { current: document.body }
    : undefined;

// Bioluminescent role accents.
const ACCENT_HEX: Record<NodeAccent, string> = {
  cyan: "#22d3ee",
  violet: "#a855f7",
  orange: "#fb923c",
  emerald: "#34d399",
};

// Live status hues override the idle role color.
const STATUS_HEX: Partial<Record<NodeStatus, string>> = {
  success: "#34d399",
  warn: "#fbbf24",
  error: "#f87171",
};

interface Props {
  label: string;
  sublabel: string;
  position: [number, number, number];
  status: NodeStatus;
  accent: NodeAccent;
  shape?: NodeShape;
  tooltip?: string;
  showTooltip?: boolean;
  /** "scroll" → node emerges from the core when revealed; "dashboard" → formed. */
  mode?: "scroll" | "dashboard";
  /** Whether this node has emerged from the core yet (scroll mode). */
  revealed?: boolean;
}

function Geometry({ shape }: { shape?: NodeShape }) {
  // Modest scale — prominent but not screen-filling.
  if (shape === "octa") return <octahedronGeometry args={[0.66, 0]} />;
  if (shape === "box") return <boxGeometry args={[0.88, 0.88, 0.88]} />;
  return <icosahedronGeometry args={[0.6, 1]} />;
}

export default function AgentNode({
  label,
  sublabel,
  position,
  status,
  accent,
  shape,
  tooltip,
  showTooltip,
  mode = "dashboard",
  revealed = true,
}: Props) {
  const groupRef = useRef<THREE.Group>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  const matRef = useRef<THREE.MeshStandardMaterial>(null);
  const haloRef = useRef<THREE.Mesh>(null);

  const targetPos = useMemo(() => new THREE.Vector3(...position), [position]);
  const formed = mode === "dashboard" || revealed;
  const formRef = useRef(formed ? 1 : 0);

  const active = status === "processing";
  // Resolve color: live status wins, else the node's role accent.
  const color = STATUS_HEX[status] ?? ACCENT_HEX[accent];

  useFrame((state, delta) => {
    const t = state.clock.elapsedTime;
    const mesh = meshRef.current;
    const mat = matRef.current;
    const halo = haloRef.current;
    const group = groupRef.current;
    if (!mesh || !mat || !group) return;

    // Emergence: fly out from the core (origin → target) and scale up.
    formRef.current = THREE.MathUtils.damp(formRef.current, formed ? 1 : 0, 7, delta);
    const f = formRef.current;
    group.position.lerpVectors(ORIGIN, targetPos, f);
    group.scale.setScalar(Math.max(0.0001, f));

    mesh.rotation.y += active ? 0.025 : 0.004;
    mesh.rotation.x += active ? 0.012 : 0.002;

    if (active) {
      const pulse = 1 + Math.sin(t * 4) * 0.12;
      mesh.scale.setScalar(pulse);
      mat.emissiveIntensity = 2.2 + Math.sin(t * 4) * 0.8;
    } else {
      mesh.scale.setScalar(THREE.MathUtils.lerp(mesh.scale.x, 1, 0.1));
      // Idle nodes still glow (bloom picks this up) so the network reads as a
      // live bioluminescent map even before a query.
      mat.emissiveIntensity = THREE.MathUtils.lerp(
        mat.emissiveIntensity,
        status === "idle" ? 1.1 : 1.5,
        0.1,
      );
    }

    if (halo) {
      const haloMat = halo.material as THREE.MeshBasicMaterial;
      const target = active ? 0.22 : status === "idle" ? 0.07 : 0.12;
      haloMat.opacity = THREE.MathUtils.lerp(haloMat.opacity, target, 0.1);
      halo.scale.setScalar(active ? 1 + Math.sin(t * 4) * 0.06 : 1);
    }
  });

  return (
    <group ref={groupRef}>
      {/* glow halo */}
      <mesh ref={haloRef}>
        <sphereGeometry args={[1.05, 24, 24]} />
        <meshBasicMaterial color={color} transparent opacity={0} depthWrite={false} />
      </mesh>

      {/* core node — dark glass body, bioluminescent emissive edge */}
      <mesh ref={meshRef}>
        <Geometry shape={shape} />
        <meshStandardMaterial
          ref={matRef}
          color="#0a0e16"
          emissive={color}
          emissiveIntensity={1.1}
          metalness={0.9}
          roughness={0.25}
          flatShading
        />
      </mesh>

      {/* sharp wireframe edge */}
      <mesh scale={1.03}>
        <Geometry shape={shape} />
        <meshBasicMaterial color={color} wireframe transparent opacity={0.5} />
      </mesh>

      {/* monospace title label — only once the node has emerged */}
      {formed && (
        <Html
          center
          position={[0, -1.25, 0]}
          occlude={false}
          // Below the 2D overlay (z-10) so the result modal's scrim covers
          // these labels instead of them bleeding through the answer text.
          // Still above the WebGL canvas (z-auto), so labels stay visible.
          zIndexRange={[9, 0]}
          portal={bodyPortal}
          style={{ pointerEvents: "none", userSelect: "none" }}
        >
          <div className="flex select-none flex-col items-center whitespace-nowrap">
            <span
              className="rounded-md bg-black/55 px-2 py-0.5 font-mono text-[13px] font-semibold tracking-wide"
              style={{ color: status === "idle" ? "#dbe4f0" : color }}
            >
              {label}
            </span>
            <span className="mt-1 font-mono text-[9px] uppercase tracking-widest text-slate-400">
              {sublabel}
            </span>
          </div>
        </Html>
      )}

      {/* spatially-anchored description tooltip (scroll narrative) */}
      {tooltip && showTooltip && (
        <Html
          // No distanceFactor/center: drei's projection throws the anchor
          // off-screen at close camera range when those are combined. A plain
          // overlay pinned at the node's projected point stays a constant,
          // legible size and is reliably on-screen whenever the node is.
          position={[1.0, 0, 0]}
          occlude={false}
          zIndexRange={[100, 0]}
          portal={bodyPortal}
          style={{ pointerEvents: "none", userSelect: "none" }}
        >
          <div
            className="w-72 -translate-y-1/2 select-none rounded-xl border border-white/10 bg-black/75 p-4 shadow-2xl backdrop-blur-md"
            style={{ borderLeftWidth: 2, borderLeftColor: color }}
          >
            <h3
              className="mb-1.5 font-mono text-base font-bold uppercase tracking-wider"
              style={{ color }}
            >
              {label}
            </h3>
            <p className="font-sans text-sm leading-relaxed text-slate-300">
              {tooltip}
            </p>
          </div>
        </Html>
      )}
    </group>
  );
}
