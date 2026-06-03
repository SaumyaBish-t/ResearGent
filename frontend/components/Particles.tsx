"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { STORY } from "@/lib/graph-config";
import { useAgentStore } from "@/lib/store";

const COUNT = 2600;
const FORM_BY = 0.5; // core fully routed into nodes by 50% scroll
const CORE_RADIUS = 2.6;

function smoothstep(e0: number, e1: number, x: number) {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

interface Props {
  /** "scroll" → snap follows scrollProgress; "dashboard" → fully formed. */
  mode: "scroll" | "dashboard";
  reduced?: boolean;
}

/**
 * "Structured Chaos" — at 0% scroll the particles form a dense, glowing Neural
 * Core (raw, unstructured data) slowly rotating at the origin. As the user
 * scrolls, the core explodes and each particle magnetically routes to one of
 * the 7 agent nodes. In dashboard mode the network stays formed and breathes.
 */
export default function Particles({ mode, reduced = false }: Props) {
  const pointsRef = useRef<THREE.Points>(null);
  const snapRef = useRef(mode === "dashboard" ? 1 : 0);

  const { positions, core, targets } = useMemo(() => {
    const storyNodes = STORY.filter((s) => s.node).map((s) => s.look);
    const positions = new Float32Array(COUNT * 3);
    const core = new Float32Array(COUNT * 3); // dense central sphere
    const targets = new Float32Array(COUNT * 3); // per-node nebula

    for (let i = 0; i < COUNT; i++) {
      // --- dense core position (denser toward the center) ---
      const cr = CORE_RADIUS * Math.pow(Math.random(), 0.7);
      const ct = Math.random() * Math.PI * 2;
      const cp = Math.acos(2 * Math.random() - 1);
      core[i * 3] = cr * Math.sin(cp) * Math.cos(ct);
      core[i * 3 + 1] = cr * Math.cos(cp);
      core[i * 3 + 2] = cr * Math.sin(cp) * Math.sin(ct);

      // --- node target (tight nebula around the assigned node) ---
      const cluster = storyNodes[i % storyNodes.length];
      const r = 0.18 + Math.random() * 0.5;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      targets[i * 3] = cluster[0] + r * Math.sin(phi) * Math.cos(theta);
      targets[i * 3 + 1] = cluster[1] + r * Math.cos(phi);
      targets[i * 3 + 2] = cluster[2] + r * Math.sin(phi) * Math.sin(theta);

      positions[i * 3] = core[i * 3];
      positions[i * 3 + 1] = core[i * 3 + 1];
      positions[i * 3 + 2] = core[i * 3 + 2];
    }
    return { positions, core, targets };
  }, []);

  useFrame((state, delta) => {
    const pts = pointsRef.current;
    if (!pts) return;
    const t = state.clock.elapsedTime * (reduced ? 0.3 : 1);

    const targetSnap =
      mode === "dashboard"
        ? 1
        : smoothstep(0, FORM_BY, useAgentStore.getState().scrollProgress);
    snapRef.current = THREE.MathUtils.damp(snapRef.current, targetSnap, 6, delta);
    const snap = snapRef.current;

    // Slow core rotation while unstructured.
    const a = t * 0.12 * (1 - snap);
    const cosA = Math.cos(a);
    const sinA = Math.sin(a);
    const breathe = mode === "dashboard" ? 0.06 : 1;

    const attr = pts.geometry.getAttribute("position") as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;

    for (let i = 0; i < COUNT; i++) {
      const cx = core[i * 3];
      const cy = core[i * 3 + 1];
      const cz = core[i * 3 + 2];
      // rotate core point around Y
      const rx = cx * cosA - cz * sinA;
      const rz = cx * sinA + cz * cosA;
      const ry = cy + Math.sin(t * 0.6 + i) * 0.08;

      const tx = targets[i * 3];
      const ty = targets[i * 3 + 1];
      const tz = targets[i * 3 + 2];

      const jx = Math.sin(t * 0.8 + i) * 0.04 * breathe;
      const jy = Math.cos(t * 0.7 + i * 1.3) * 0.04 * breathe;

      arr[i * 3] = THREE.MathUtils.lerp(rx, tx + jx, snap);
      arr[i * 3 + 1] = THREE.MathUtils.lerp(ry, ty + jy, snap);
      arr[i * 3 + 2] = THREE.MathUtils.lerp(rz, tz, snap);
    }
    attr.needsUpdate = true;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        color="#9fe8ff"
        size={0.055}
        sizeAttenuation
        transparent
        opacity={0.85}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}
