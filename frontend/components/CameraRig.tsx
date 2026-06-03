"use client";

import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { useScroll } from "@react-three/drei";
import * as THREE from "three";
import { DASHBOARD_VIEW, STORY } from "@/lib/graph-config";
import { useAgentStore } from "@/lib/store";

// Reusable temporaries — never allocate inside useFrame.
const _cam = new THREE.Vector3();
const _look = new THREE.Vector3();
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();

function easeInOutCubic(x: number) {
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
}

/** Sample the choreographed camera pose for a scroll offset (0..1). */
function sampleStory(offset: number, outCam: THREE.Vector3, outLook: THREE.Vector3) {
  if (offset <= STORY[0].at) {
    outCam.set(...STORY[0].cam);
    outLook.set(...STORY[0].look);
    return;
  }
  for (let i = 1; i < STORY.length; i++) {
    const b = STORY[i];
    if (offset <= b.at) {
      const a = STORY[i - 1];
      const span = b.at - a.at || 1;
      const e = easeInOutCubic((offset - a.at) / span);
      outCam.lerpVectors(_a.set(...a.cam), _b.set(...b.cam), e);
      outLook.lerpVectors(_a.set(...a.look), _b.set(...b.look), e);
      return;
    }
  }
  const last = STORY[STORY.length - 1];
  outCam.set(...last.cam);
  outLook.set(...last.look);
}

/**
 * Scroll-driven camera. MUST be rendered inside <ScrollControls>. Reads the
 * scroll offset each frame, samples the choreography, and critically-damps the
 * camera toward the pose for buttery, interruptible motion.
 */
export function ScrollCamera({ reduced = false }: { reduced?: boolean }) {
  const scroll = useScroll();
  const lookRef = useRef(new THREE.Vector3(...STORY[0].look));

  useFrame((state, delta) => {
    sampleStory(scroll.offset, _cam, _look);
    const lambda = reduced ? 18 : 4.5;
    state.camera.position.x = THREE.MathUtils.damp(state.camera.position.x, _cam.x, lambda, delta);
    state.camera.position.y = THREE.MathUtils.damp(state.camera.position.y, _cam.y, lambda, delta);
    state.camera.position.z = THREE.MathUtils.damp(state.camera.position.z, _cam.z, lambda, delta);
    lookRef.current.x = THREE.MathUtils.damp(lookRef.current.x, _look.x, lambda, delta);
    lookRef.current.y = THREE.MathUtils.damp(lookRef.current.y, _look.y, lambda, delta);
    lookRef.current.z = THREE.MathUtils.damp(lookRef.current.z, _look.z, lambda, delta);
    state.camera.lookAt(lookRef.current);
  });

  return null;
}

/**
 * Locked dashboard camera. Rendered OUTSIDE <ScrollControls> once a query is
 * submitted; smoothly pulls the camera out to the System Dashboard Map View
 * and holds it there while the network fires live.
 */
export function DashboardCamera() {
  const lookRef = useRef(new THREE.Vector3());
  const initialized = useRef(false);

  useFrame((state, delta) => {
    if (!initialized.current) {
      lookRef.current.copy(state.camera.getWorldDirection(_look).add(state.camera.position));
      initialized.current = true;
    }
    _cam.set(...DASHBOARD_VIEW.cam);
    _look.set(...DASHBOARD_VIEW.look);
    state.camera.position.x = THREE.MathUtils.damp(state.camera.position.x, _cam.x, 3, delta);
    state.camera.position.y = THREE.MathUtils.damp(state.camera.position.y, _cam.y, 3, delta);
    state.camera.position.z = THREE.MathUtils.damp(state.camera.position.z, _cam.z, 3, delta);
    lookRef.current.x = THREE.MathUtils.damp(lookRef.current.x, _look.x, 3, delta);
    lookRef.current.y = THREE.MathUtils.damp(lookRef.current.y, _look.y, 3, delta);
    lookRef.current.z = THREE.MathUtils.damp(lookRef.current.z, _look.z, 3, delta);
    state.camera.lookAt(lookRef.current);
  });

  return null;
}

/**
 * Writes the live scroll offset into the Zustand store (quantized) so the HTML
 * narrative overlay can react. MUST be inside <ScrollControls>.
 */
export function ScrollReporter() {
  const scroll = useScroll();
  const setScrollProgress = useAgentStore((s) => s.setScrollProgress);
  useFrame(() => setScrollProgress(scroll.offset));
  return null;
}
