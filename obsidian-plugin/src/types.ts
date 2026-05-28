// Mirrors the JSON shapes emitted by the Python backend's stream_agent.
// Keep in sync with src/agent/stream.py — any change there should update this.

export type StreamEventType =
  | "run_started"
  | "node_complete"
  | "final"
  | "error";

export interface RunStartedEvent {
  type: "run_started";
  run_id: string;
  question: string;
  ts: number;
}

export interface NodeCompleteEvent {
  type: "node_complete";
  node:
    | "planner"
    | "retriever"
    | "critic"
    | "rewriter"
    | "web_fallback"
    | "generator"
    | "reflector"
    | "no_answer"
    | "paper_discovery"
    | "llm_reasoning";
  summary: Record<string, unknown>;
  ts: number;
}

export interface SourcePayload {
  tag: string;          // "S1", "S2", ...
  citation: string;     // "file.md p.0" | "arxiv:..." | "https://..."
  signal: string;       // "BOTH" | "dense" | "bm25" | "web:tavily" | "paper:arxiv" | ...
  rrf_score: number | null;
  score: number | null;
  preview: string;
}

export interface FinalEvent {
  type: "final";
  run_id: string;
  answer: string;
  sources: SourcePayload[];
  sub_questions: string[];
  is_complex: boolean;
  confidence: string;
  rewrite_attempts: number;
  web_used: boolean;
  reflection_attempts: number;
  error: string | null;
  ts: number;
}

export interface ErrorEvent {
  type: "error";
  error: string;
  ts: number;
}

export type StreamEvent =
  | RunStartedEvent
  | NodeCompleteEvent
  | FinalEvent
  | ErrorEvent;

export interface ResearGentSettings {
  backendUrl: string;          // e.g. "http://127.0.0.1:8000"
  defaultK: number;            // top-k for retrieval
  outputFolder: string;        // subfolder for saved notes
  includeActiveNote: boolean;  // include current note as context for "Research this note"
  autoOpenSavedNote: boolean;  // open the new note immediately after saving
}

export const DEFAULT_SETTINGS: ResearGentSettings = {
  backendUrl: "http://127.0.0.1:8000",
  defaultK: 8,
  outputFolder: "ResearGent",
  includeActiveNote: true,
  autoOpenSavedNote: true,
};
