/**
 * Static description of the agent network for the 3D scene + the scrollytelling
 * narrative.
 *
 * Two concerns live here:
 *   1. NODES / EDGES — the full live-dashboard graph (mirrors src/agent/graph.py).
 *      Positions for the 7 "story" nodes use the curated narrative coordinates;
 *      the remaining backend nodes are tucked into the lower region so the live
 *      dashboard can still light every node a `node_complete` event names.
 *   2. STORY — the ordered 7-stop camera choreography (positions, lookAt
 *      targets, and overlay copy) driven by scroll progress.
 */

import type { NodeId } from "./types";

export type NodeShape = "ico" | "octa" | "box";
// Bioluminescent role accent: cyan = retrieval, violet = processing,
// orange = critic, emerald = vault.
export type NodeAccent = "cyan" | "violet" | "orange" | "emerald";

export interface NodeDef {
  id: NodeId;
  label: string;
  sublabel: string;
  position: [number, number, number];
  accent: NodeAccent;
  shape?: NodeShape;
}

export interface EdgeDef {
  from: NodeId;
  to: NodeId;
}

// Curated narrative coordinates (the 7 spotlighted nodes) + supporting nodes.
export const NODES: NodeDef[] = [
  { id: "planner", label: "Planner", sublabel: "decompose", position: [0, 4, 0], accent: "violet" },
  { id: "retriever", label: "Local Retriever", sublabel: "vector search", position: [-2, 2, -2], accent: "cyan" },
  { id: "critic", label: "The Critic", sublabel: "temporal grader", position: [0, 0, 0], accent: "orange", shape: "octa" },
  { id: "rewriter", label: "Rewriter", sublabel: "query retry", position: [2, 1, -1], accent: "violet" },
  { id: "paper_discovery", label: "S2 / arXiv", sublabel: "paper fetch", position: [-3, -2, 2], accent: "cyan" },
  { id: "web_fallback", label: "Tavily Web", sublabel: "live fallback", position: [3, -2, 2], accent: "cyan" },
  { id: "vault", label: "Obsidian Vault", sublabel: "save gate", position: [0, -5, 0], accent: "emerald", shape: "box" },

  // Supporting nodes (not spotlighted in the narrative, but live in the
  // dashboard so every streamed node has a home).
  { id: "generator", label: "Generator", sublabel: "cite + write", position: [1.6, -3.2, -0.6], accent: "violet" },
  { id: "reflector", label: "Reflector", sublabel: "gap audit", position: [-1.6, -3.6, -0.6], accent: "violet" },
  { id: "llm_reasoning", label: "LLM Priors", sublabel: "last resort", position: [4.6, -3.6, 1], accent: "violet" },
  { id: "no_answer", label: "No Answer", sublabel: "graceful stop", position: [-4.6, -3.8, 1], accent: "violet" },
];

export const NODE_BY_ID: Record<NodeId, NodeDef> = NODES.reduce(
  (acc, n) => {
    acc[n.id] = n;
    return acc;
  },
  {} as Record<NodeId, NodeDef>,
);

// Directed edges — every transition the graph can make (see graph.py).
export const EDGES: EdgeDef[] = [
  { from: "planner", to: "retriever" },
  { from: "retriever", to: "critic" },
  { from: "critic", to: "generator" },
  { from: "critic", to: "rewriter" },
  { from: "critic", to: "paper_discovery" },
  { from: "critic", to: "web_fallback" },
  { from: "rewriter", to: "critic" },
  { from: "paper_discovery", to: "critic" },
  { from: "web_fallback", to: "critic" },
  { from: "web_fallback", to: "llm_reasoning" },
  { from: "web_fallback", to: "no_answer" },
  { from: "retriever", to: "paper_discovery" },
  { from: "retriever", to: "web_fallback" },
  { from: "retriever", to: "llm_reasoning" },
  { from: "generator", to: "reflector" },
  { from: "reflector", to: "retriever" }, // loopback
  { from: "generator", to: "vault" }, // synthetic save edge
];

/** Does a directed edge exist between two nodes (in either declared direction)? */
export function edgeExists(from: NodeId, to: NodeId): boolean {
  return EDGES.some(
    (e) => (e.from === from && e.to === to) || (e.from === to && e.to === from),
  );
}

// ---- Scrollytelling camera choreography -----------------------------------

export interface StoryStop {
  /** Scroll offset (0..1) at which the camera fully settles on this stop. */
  at: number;
  /** Spotlighted node id (null = the establishing hero/vortex shot). */
  node: NodeId | null;
  step: string; // "01".."07"
  title: string;
  role: string; // overlay copy
  /** Camera world position at this stop. */
  cam: [number, number, number];
  /** Camera lookAt target at this stop. */
  look: [number, number, number];
  /** Which side the glass overlay card sits on (avoids the centered search). */
  side: "left" | "right";
}

export const STORY: StoryStop[] = [
  {
    at: 0.0,
    node: null,
    step: "00",
    title: "Data Accretion",
    role: "Raw, unorganized research signal swirling before structure emerges.",
    cam: [0, 0, 11],
    look: [0, 0, 0],
    side: "left",
  },
  {
    at: 0.15,
    node: "planner",
    step: "01",
    title: "The Planner",
    role: "Decomposes complex research queries into structured, atomic sub-tasks and logical execution branches.",
    cam: [0, 8.5, 11], // high-angle, pulled back to leave room for the tooltip
    look: [0, 4, 0],
    side: "left",
  },
  {
    at: 0.3,
    node: "retriever",
    step: "02",
    title: "Local Retriever",
    role: "Executes semantic searches across Phase 15 vector databases to extract highly relevant local context.",
    cam: [5.4, 4.2, 7], // linear slide along the primary edge
    look: [-2, 2, -2],
    side: "right",
  },
  {
    at: 0.45,
    node: "critic",
    step: "03",
    title: "The Critic",
    role: "A strict, multi-layered temporal grader that ruthlessly evaluates data freshness and rejects hallucinated chunks.",
    cam: [0, 0.8, 9.5], // abrupt stop, focus on the diamond node
    look: [0, 0, 0],
    side: "left",
  },
  {
    at: 0.6,
    node: "rewriter",
    step: "04",
    title: "Rewriter",
    role: "Dynamically re-engineers query syntax to bridge semantic gaps when initial retrieval fails.",
    cam: [9.5, 3.8, 7], // orbit sweep
    look: [2, 1, -1],
    side: "right",
  },
  {
    at: 0.75,
    node: "paper_discovery",
    step: "05",
    title: "S2 API Fetcher",
    role: "Asynchronously hits Semantic Scholar and parses open-access arXiv PDFs on-the-fly.",
    cam: [-3.5, 0.6, 13.5], // rapid zoom out + drop to the peripheral cluster
    look: [-3, -2, 2],
    side: "left",
  },
  {
    at: 0.9,
    node: "web_fallback",
    step: "06",
    title: "Tavily Web Fallback",
    role: "Triggers live-web scraping execution paths as a resilient fallback mechanism.",
    cam: [11.5, -1, 8.5], // horizontal pan to the tethered external node
    look: [3, -2, 2],
    side: "right",
  },
  {
    at: 1.0,
    node: "vault",
    step: "07",
    title: "Obsidian Vault Gate",
    role: "Compiles verified data and writes zero-hallucination, heavily cited Markdown straight to local storage.",
    cam: [0, -2.2, 14.5], // smooth tilt up + pull back to the exit terminal
    look: [0, -5, 0],
    side: "left",
  },
];

// node id → its scroll-narrative description (for the 3D-anchored tooltip).
export const TOOLTIP_BY_NODE: Partial<Record<NodeId, string>> = Object.fromEntries(
  STORY.filter((s) => s.node).map((s) => [s.node as NodeId, s.role]),
);

// Ordered ids of the 7 spotlighted story nodes.
export const STORY_NODE_ORDER: NodeId[] = STORY.filter((s) => s.node).map(
  (s) => s.node as NodeId,
);

/**
 * How many story nodes have "emerged" from the core at this scroll offset.
 * A node emerges slightly before the camera fully arrives on it.
 */
export function revealedCount(progress: number): number {
  let c = 0;
  for (const s of STORY) {
    if (s.node && progress >= s.at - 0.07) c++;
  }
  return c;
}

/** The story node nearest the current scroll offset (null near the hero). */
export function storyNodeAt(progress: number): NodeId | null {
  let best = STORY[0];
  let bestD = Infinity;
  for (const s of STORY) {
    const d = Math.abs(s.at - progress);
    if (d < bestD) {
      bestD = d;
      best = s;
    }
  }
  return best.node;
}

// The locked overview the camera flies to once a query is submitted.
export const DASHBOARD_VIEW = {
  cam: [0, 0.5, 23] as [number, number, number],
  look: [0, -0.8, 0] as [number, number, number],
};

// Total virtual scroll pages for <ScrollControls>.
export const SCROLL_PAGES = 6;
