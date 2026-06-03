/**
 * Wire types — these mirror EXACTLY what `src/agent/stream.py` emits over SSE.
 * Each SSE message has an `event:` name (the `type`) and a JSON `data:` body.
 *
 * Keep this file in lock-step with the backend. If you add a field there, add
 * it here.
 */

// The agent graph's node names, verbatim from `src/agent/graph.py`.
// `vault` is a synthetic UI-only node representing the auto-save step (the
// backend signals it via the `saved` / `save_skipped` events, not a node).
export type NodeId =
  | "planner"
  | "retriever"
  | "critic"
  | "rewriter"
  | "paper_discovery"
  | "web_fallback"
  | "reflector"
  | "generator"
  | "llm_reasoning"
  | "no_answer"
  | "vault";

export type NodeStatus = "idle" | "processing" | "success" | "warn" | "error";

// ---- Event payloads -------------------------------------------------------

export interface RunStartedEvent {
  type: "run_started";
  run_id: string;
  question: string;
  ts: number;
}

export interface NodeCompleteEvent {
  type: "node_complete";
  node: Exclude<NodeId, "vault">;
  summary: NodeSummary;
  ts: number;
}

// Per-node summary is a union-ish bag; all fields optional because each node
// fills only its own slice (see `_summarize_node_update` in stream.py).
export interface NodeSummary {
  // planner
  is_complex?: boolean;
  sub_questions?: string[];
  reasoning?: string;
  // retriever
  total_chunks?: number;
  per_subq_counts?: Record<string, number>;
  graph_expanded?: number;
  // critic
  confidence?: string; // "high" | "medium" | "low"
  score?: number; // 0..1 numeric grade
  grades?: Record<string, number> | unknown;
  chunks_in?: number;
  chunks_kept?: number;
  // rewriter
  rewrite_attempt?: number;
  rewritten_count?: number;
  // web_fallback
  web_chunks_added?: number;
  providers_used?: string[];
  // generator
  answer_chars?: number;
  n_sources?: number;
  // reflector
  attempt?: number;
  gaps_found?: boolean;
  follow_ups?: string[];
  gaps?: string[];
  // no_answer
  reason?: string;
}

export interface Source {
  tag: string; // "[S1]" etc.
  citation: string;
  signal: string; // "paper:*" | "web:*" | "local"
  preview: string;
  rrf_score?: number | null;
  score?: number | null;
}

export interface FinalEvent {
  type: "final";
  run_id: string;
  answer: string;
  sources: Source[];
  sub_questions: string[];
  is_complex: boolean;
  confidence: string;
  score: number;
  rewrite_attempts: number;
  web_used: boolean;
  reflection_attempts: number;
  error: string | null;
  ts: number;
}

export interface SavedEvent {
  type: "saved";
  path: string;
  ts: number;
}

export interface SaveSkippedEvent {
  type: "save_skipped";
  reason: string;
  ts: number;
}

export interface ErrorEvent {
  type: "error";
  error: string;
  ts?: number;
}

export type AgentEvent =
  | RunStartedEvent
  | NodeCompleteEvent
  | FinalEvent
  | SavedEvent
  | SaveSkippedEvent
  | ErrorEvent;

// The terminal events that mean "the stream is genuinely done" — used to
// proactively close the EventSource so it doesn't auto-reconnect and re-run
// the whole agent.
export const TERMINAL_EVENTS = ["saved", "save_skipped", "error"] as const;
