/**
 * useAgentStore — the central nervous system bridging the SSE stream, the 2D
 * React overlay, and the Three.js canvas.
 *
 * Phase 2 of the build. The store:
 *   - opens an EventSource against the FastAPI `/api/research` SSE endpoint,
 *   - reduces each named event into flat, render-friendly state,
 *   - tracks the active node + the edge data is currently flowing along so the
 *     3D scene can pulse nodes and shoot particles along edges.
 *
 * IMPORTANT — EventSource auto-reconnect: when the backend generator finishes
 * it CLOSES the stream, which a native EventSource interprets as a dropped
 * connection and silently reconnects → re-running the entire agent. We defend
 * against this by closing the EventSource ourselves the instant a terminal
 * event (`saved` / `save_skipped` / `error`) arrives.
 */

import { create } from "zustand";
import { NODES, edgeExists } from "./graph-config";
import type {
  AgentEvent,
  FinalEvent,
  NodeCompleteEvent,
  NodeId,
  NodeStatus,
  RunStartedEvent,
  SavedEvent,
  SaveSkippedEvent,
  Source,
} from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";

export interface LogEntry {
  id: number;
  ts: number;
  node: string;
  message: string;
  level: "info" | "warn" | "error" | "success";
}

interface AgentState {
  // ---- input / lifecycle ----
  query: string;
  running: boolean;
  finished: boolean;
  runId: string | null;

  // ---- structural UI state (scrollytelling ↔ dashboard) ----
  // `hasQueried` flips the whole experience from the scroll-driven intro
  // narrative to the live dashboard map view. `scrollProgress` (0..1) is
  // written by the in-canvas ScrollReporter and read by the HTML narrative
  // overlay — kept here so the 2D and 3D layers share one source of truth.
  hasQueried: boolean;
  scrollProgress: number;

  // ---- live graph state (consumed by the 3D scene) ----
  currentActiveNode: NodeId | null;
  previousNode: NodeId | null;
  activeEdge: { from: NodeId; to: NodeId } | null;
  nodeStatuses: Record<NodeId, NodeStatus>;

  // ---- scores + results ----
  confidenceScore: number | null; // numeric 0..1 (critic_score)
  confidenceLabel: string; // high | medium | low
  logs: LogEntry[];
  sources: Source[];
  finalOutput: FinalEvent | null;
  savedPath: string | null;
  saveSkippedReason: string | null;
  error: string | null;

  // ---- actions ----
  setQuery: (q: string) => void;
  setScrollProgress: (p: number) => void;
  startRun: (q?: string) => void;
  stopRun: () => void;
  reset: () => void;
}

function idleStatuses(): Record<NodeId, NodeStatus> {
  return NODES.reduce(
    (acc, n) => {
      acc[n.id] = "idle";
      return acc;
    },
    {} as Record<NodeId, NodeStatus>,
  );
}

// Module-scoped handle so we can close across renders / re-runs.
let es: EventSource | null = null;
let logSeq = 0;

function freshRunState(): Partial<AgentState> {
  return {
    running: true,
    finished: false,
    runId: null,
    currentActiveNode: null,
    previousNode: null,
    activeEdge: null,
    nodeStatuses: idleStatuses(),
    confidenceScore: null,
    confidenceLabel: "",
    logs: [],
    sources: [],
    finalOutput: null,
    savedPath: null,
    saveSkippedReason: null,
    error: null,
  };
}

export const useAgentStore = create<AgentState>((set, get) => {
  const log = (
    node: string,
    message: string,
    level: LogEntry["level"] = "info",
  ) =>
    set((s) => ({
      logs: [
        ...s.logs,
        { id: logSeq++, ts: Date.now(), node, message, level },
      ],
    }));

  const closeStream = () => {
    if (es) {
      es.close();
      es = null;
    }
  };

  const handleEvent = (evt: AgentEvent) => {
    switch (evt.type) {
      case "run_started": {
        const e = evt as RunStartedEvent;
        set({ runId: e.run_id });
        log("system", `run ${e.run_id} started`, "info");
        break;
      }

      case "node_complete": {
        const e = evt as NodeCompleteEvent;
        const node = e.node as NodeId;
        const prev = get().currentActiveNode;

        set((s) => {
          const statuses = { ...s.nodeStatuses };
          // The node that WAS active is now done.
          if (prev && prev !== node) statuses[prev] = "success";
          // Critic with non-high confidence is a soft "rejection" → amber flash.
          const isCriticReject =
            node === "critic" &&
            e.summary.confidence != null &&
            e.summary.confidence !== "high";
          statuses[node] =
            node === "no_answer"
              ? "error"
              : isCriticReject
                ? "warn"
                : "processing";

          const activeEdge =
            prev && prev !== node && edgeExists(prev, node)
              ? { from: prev, to: node }
              : null;

          const patch: Partial<AgentState> = {
            nodeStatuses: statuses,
            previousNode: prev,
            currentActiveNode: node,
            activeEdge,
          };

          if (node === "critic") {
            if (typeof e.summary.score === "number")
              patch.confidenceScore = e.summary.score;
            if (e.summary.confidence) patch.confidenceLabel = e.summary.confidence;
          }
          return patch;
        });

        log(node, summarizeNode(node, e.summary), nodeLevel(node, e.summary));
        break;
      }

      case "final": {
        const e = evt as FinalEvent;
        set((s) => {
          const statuses = { ...s.nodeStatuses };
          if (s.currentActiveNode) statuses[s.currentActiveNode] = "success";
          statuses.generator = e.error ? "error" : "success";
          return {
            nodeStatuses: statuses,
            finalOutput: e,
            sources: e.sources || [],
            confidenceScore: typeof e.score === "number" ? e.score : s.confidenceScore,
            confidenceLabel: e.confidence || s.confidenceLabel,
            activeEdge: null,
          };
        });
        log(
          "generator",
          `final answer · ${e.sources?.length ?? 0} sources · ${e.confidence}`,
          "success",
        );
        break;
      }

      case "saved": {
        const e = evt as SavedEvent;
        set((s) => ({
          savedPath: e.path,
          running: false,
          finished: true,
          nodeStatuses: { ...s.nodeStatuses, vault: "success" },
        }));
        log("vault", `saved → ${e.path}`, "success");
        closeStream();
        break;
      }

      case "save_skipped": {
        const e = evt as SaveSkippedEvent;
        set((s) => ({
          saveSkippedReason: e.reason,
          running: false,
          finished: true,
          nodeStatuses: { ...s.nodeStatuses, vault: "warn" },
        }));
        log("vault", `save skipped — ${e.reason}`, "warn");
        closeStream();
        break;
      }

      case "error": {
        set((s) => ({
          error: evt.error,
          running: false,
          finished: true,
          nodeStatuses: s.currentActiveNode
            ? { ...s.nodeStatuses, [s.currentActiveNode]: "error" }
            : s.nodeStatuses,
        }));
        log("system", evt.error, "error");
        closeStream();
        break;
      }
    }
  };

  return {
    query: "",
    running: false,
    finished: false,
    runId: null,
    hasQueried: false,
    scrollProgress: 0,
    currentActiveNode: null,
    previousNode: null,
    activeEdge: null,
    nodeStatuses: idleStatuses(),
    confidenceScore: null,
    confidenceLabel: "",
    logs: [],
    sources: [],
    finalOutput: null,
    savedPath: null,
    saveSkippedReason: null,
    error: null,

    setQuery: (q) => set({ query: q }),

    // Quantized to ~1% steps so 60fps frame writes don't thrash React; the
    // narrative overlay only needs threshold-crossing resolution.
    setScrollProgress: (p) => {
      const q = Math.round(Math.min(1, Math.max(0, p)) * 100) / 100;
      if (q !== get().scrollProgress) set({ scrollProgress: q });
    },

    startRun: (q) => {
      const question = (q ?? get().query).trim();
      if (!question || get().running) return;

      closeStream(); // tear down any previous stream first
      // `hasQueried: true` is the switch from intro narrative → live dashboard.
      set({ ...freshRunState(), query: question, hasQueried: true });

      const url = `${API_BASE}/api/research?q=${encodeURIComponent(question)}`;
      es = new EventSource(url);

      // Named events — the backend tags each message with `event: <type>`, so
      // the default `onmessage` never fires. We must listen per type.
      const names: AgentEvent["type"][] = [
        "run_started",
        "node_complete",
        "final",
        "saved",
        "save_skipped",
        "error",
      ];
      for (const name of names) {
        es.addEventListener(name, (ev) => {
          try {
            const data = JSON.parse((ev as MessageEvent).data);
            handleEvent({ ...data, type: name } as AgentEvent);
          } catch (err) {
            log("system", `bad event payload for ${name}: ${String(err)}`, "error");
          }
        });
      }

      es.onerror = () => {
        // If we've already finished, the server simply closed the stream —
        // ignore. Otherwise it's a genuine connection problem.
        if (get().finished) return;
        set({
          error: "connection to backend lost (is `researgent serve` running?)",
          running: false,
          finished: true,
        });
        log("system", "EventSource connection error", "error");
        closeStream();
      };
    },

    stopRun: () => {
      closeStream();
      set({ running: false, finished: true });
    },

    reset: () => {
      closeStream();
      // Returns the user to the scrollytelling intro (hasQueried: false).
      set({
        ...freshRunState(),
        running: false,
        finished: false,
        hasQueried: false,
        scrollProgress: 0,
        query: get().query,
      });
    },
  };
});

// ---- log formatting helpers ----------------------------------------------

function summarizeNode(node: NodeId, s: NodeCompleteEvent["summary"]): string {
  switch (node) {
    case "planner":
      return s.is_complex
        ? `complex → ${s.sub_questions?.length ?? 0} sub-questions`
        : "simple query";
    case "retriever":
      return `retrieved ${s.total_chunks ?? 0} chunks${s.graph_expanded ? ` (+${s.graph_expanded} graph)` : ""}`;
    case "critic":
      return `confidence=${s.confidence ?? "?"}${typeof s.score === "number" ? ` score=${s.score.toFixed(2)}` : ""} kept ${s.chunks_kept ?? "?"}/${s.chunks_in ?? "?"}`;
    case "rewriter":
      return `rewrite attempt ${s.rewrite_attempt ?? "?"} · ${s.rewritten_count ?? 0} queries`;
    case "paper_discovery":
      return "discovering papers (arXiv + Semantic Scholar)";
    case "web_fallback":
      return `+${s.web_chunks_added ?? 0} web chunks${s.providers_used?.length ? ` via ${s.providers_used.join(", ")}` : ""}`;
    case "generator":
      return `drafted ${s.answer_chars ?? 0} chars · ${s.n_sources ?? 0} sources`;
    case "reflector":
      return s.gaps_found
        ? `gaps found → ${s.follow_ups?.length ?? 0} follow-ups`
        : "no gaps · accept";
    case "no_answer":
      return `no answer — ${s.reason ?? "unknown"}`;
    default:
      return node;
  }
}

function nodeLevel(
  node: NodeId,
  s: NodeCompleteEvent["summary"],
): LogEntry["level"] {
  if (node === "no_answer") return "error";
  if (node === "critic" && s.confidence && s.confidence !== "high") return "warn";
  if (node === "reflector" && s.gaps_found) return "warn";
  return "info";
}
