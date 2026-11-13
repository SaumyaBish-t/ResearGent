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
  | "provenance_check"
  | "novelty_check"
  | "human_review_gate"
  | "critic"
  | "rewriter"
  | "paper_discovery"
  | "web_fallback"
  | "reflector"
  | "generator"
  | "verifier"
  | "originality_check"
  | "memory_keeper"
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
  diagnostics?: Array<{
    question?: string;
    chunks_in?: number;
    grades?: Record<string, number>;
    sources?: Record<string, number>;
    reasoning?: string;
  }>;
  // rewriter
  rewrite_attempt?: number;
  rewritten_count?: number;
  // web_fallback
  web_chunks_added?: number;
  providers_used?: string[];
  // provenance_check
  clean?: number;
  flagged?: number;
  retracted_dropped?: number;
  blocked_sources?: Array<{
    citation?: string;
    doi?: string;
    arxiv_id?: string;
    reason?: string;
  }>;
  // originality_check
  sentences_checked?: number;
  flags_found?: number;
  duration_ms?: number;
  embedding_batches?: number;
  skipped?: string;
  uncited_flags?: Array<{
    source_tag?: string;
    status?: string;
    draft_sentence?: string;
  }>;
  // novelty_check (Phase 24)
  /** "novel" | "improvement" | "not_novel" | "inconclusive" */
  verdict?: string;
  is_novel?: boolean;
  overall_similarity?: number;
  approach_similarity?: number;
  result_similarity?: number;
  /** True when the Human Review Gate should pause the run. */
  requires_review?: boolean;
  matched_metrics?: NoveltyMetricMatch[];
  compared_papers?: number;
  /** Papers dropped by the novelty screen (near-duplicates of the user's work). */
  dropped_sources?: number;
  // human_review_gate (Phase 24)
  /** "approved" | "rejected" | "auto_approved" */
  decision?: string;
  similarity?: number;
  threshold?: number;
  approved?: boolean;
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

/**
 * One row of the novelty comparison: the user's own metric against the
 * same metric reported by a prior paper.
 *
 * These rows are the point of the Novelty Check — "your BLEU-4 38.1 vs
 * their 34.8 (better)" is what the user needs to see, not just a
 * similarity score.
 */
export interface NoveltyMetricMatch {
  /** The user's metric name (e.g. "BLEU-4"). */
  name?: string;
  /** The user's value. */
  value?: number | string;
  /** The prior paper's metric name (may differ in formatting). */
  paper_name?: string;
  /** The prior paper's value. */
  paper_value?: number | string;
  /** "better" | "worse" | "equal" */
  direction?: string;
  /** Absolute difference (user - paper). */
  delta?: number;
  source?: string;
}

/**
 * The full novelty report attached to a run (Phase 24).
 *
 * Mirrors `novelty_report` in `src/agent/state.py`.
 */
export interface NoveltyReport {
  verdict?: string;
  is_novel?: boolean;
  overall_similarity?: number;
  approach_similarity?: number;
  result_similarity?: number;
  requires_review?: boolean;
  rationale?: string;
  compared_papers?: number;
  matched_metrics?: NoveltyMetricMatch[];
  user_approach?: string;
  user_metrics?: Array<{ name?: string; value?: number | string; unit?: string }>;
}

export interface Source {
  tag: string; // "S1" etc.
  citation: string;
  signal: string; // "paper:*" | "web:*" | "local"
  preview: string;
  rrf_score?: number | null;
  score?: number | null;
  // Paper metadata (optional, from HydratedChunk)
  doc_title?: string;
  url?: string;
  source_file?: string;
  authors?: string[];
  year?: number | null;
  venue?: string;
  arxiv_id?: string;
  // Claim verification verdict (set by Verifier node)
  verdict?: "supports" | "contradicts" | "unclear";
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
  provenance_dropped_count?: number;
  provenance_flagged_count?: number;
  // Phase 24 — Novelty Check + Human Review Gate
  novelty_report?: NoveltyReport | null;
  novelty_review_approved?: boolean;
  novelty_review_decision?: string | null;
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

/**
 * Phase 24 — emitted when the Human Review Gate pauses the graph.
 *
 * The graph has STOPPED and is waiting for `POST /api/novelty/review` with
 * this `run_id`. The pause DETAIL (similarity, papers) rides on the
 * preceding `novelty_check` node_complete summary, because `interrupt()`
 * raises before the gate node writes any state.
 */
export interface ReviewRequiredEvent {
  type: "review_required";
  run_id: string;
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
  | ReviewRequiredEvent
  | ErrorEvent;

// ── Review pipeline events ──────────────────────────────────────────────
export interface ReviewPlannedEvent {
  type: "review_planned";
  run_id: string;
  title: string;
  sections: string[];
  ts: number;
}
export interface ReviewProgressEvent {
  type: "review_progress";
  run_id?: string;
  stage: string;
  message: string;
  section_index?: number;
  section_count?: number;
  ts: number;
}
export interface ReviewSectionDoneEvent {
  type: "review_section_done";
  run_id: string;
  section_index: number;
  heading: string;
  n_sources: number;
  body_len: number;
  ts: number;
}
export interface ReviewCompleteEvent {
  type: "review_complete";
  run_id: string;
  review_id?: string;
  title: string;
  markdown: string;
  sections: string[];
  n_sources: number;
  duration_ms: number;
  ts: number;
}
export interface ReviewErrorEvent {
  type: "review_error";
  run_id?: string;
  error: string;
  ts: number;
}
export type ReviewEvent =
  | ReviewProgressEvent
  | ReviewPlannedEvent
  | ReviewSectionDoneEvent
  | ReviewCompleteEvent
  | ReviewErrorEvent;

// The terminal events that mean "the stream is genuinely done" — used to
// proactively close the EventSource so it doesn't auto-reconnect and re-run
// the whole agent.
export const TERMINAL_EVENTS = ["saved", "save_skipped", "error"] as const;
