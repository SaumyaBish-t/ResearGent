"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Line } from "@react-three/drei";
import * as THREE from "three";
import { EDGES, NODE_BY_ID } from "@/lib/graph-config";
import type { NodeId } from "@/lib/types";

interface Props {
  activeEdge: { from: NodeId; to: NodeId } | null;
}

const PARTICLE_COUNT = 6;

/**
 * Renders the static edge lines between nodes, plus a burst of glowing
 * particles that travel along whichever edge data is currently flowing on.
 */
export default function Edges({ activeEdge }: Props) {
  return (
    <group>
      {EDGES.map((e, i) => {
        const a = NODE_BY_ID[e.from]?.position;
        const b = NODE_BY_ID[e.to]?.position;
        if (!a || !b) return null;
        const isActive =
          !!activeEdge &&
          ((activeEdge.from === e.from && activeEdge.to === e.to) ||
            (activeEdge.from === e.to && activeEdge.to === e.from));
        return (
          <Line
            key={i}
            points={[a, b]}
            color={isActive ? "#22d3ee" : "#26304a"}
            lineWidth={isActive ? 2.4 : 1}
            transparent
            opacity={isActive ? 0.9 : 0.32}
          />
        );
      })}

      {activeEdge && <FlowParticles activeEdge={activeEdge} />}
    </group>
  );
}

function FlowParticles({ activeEdge }: { activeEdge: { from: NodeId; to: NodeId } }) {
  const groupRef = useRef<THREE.Group>(null);

  const { start, end } = useMemo(() => {
    const a = NODE_BY_ID[activeEdge.from].position;
    const b = NODE_BY_ID[activeEdge.to].position;
    return {
      start: new THREE.Vector3(...a),
      end: new THREE.Vector3(...b),
    };
  }, [activeEdge]);

  useFrame((state) => {
    const g = groupRef.current;
    if (!g) return;
    const t = state.clock.elapsedTime;
    g.children.forEach((child, i) => {
      // Evenly spaced phases, looping 0→1 along the edge.
      const phase = (t * 0.8 + i / PARTICLE_COUNT) % 1;
      child.position.lerpVectors(start, end, phase);
      const m = (child as THREE.Mesh).material as THREE.MeshBasicMaterial;
      // Fade in at the start, out at the end.
      m.opacity = Math.sin(phase * Math.PI);
    });
  });

  return (
    <group ref={groupRef}>
      {Array.from({ length: PARTICLE_COUNT }).map((_, i) => (
        <mesh key={i}>
          <sphereGeometry args={[0.075, 8, 8]} />
          <meshBasicMaterial color="#67e8f9" transparent opacity={0.95} />
        </mesh>
      ))}
    </group>
  );
}
