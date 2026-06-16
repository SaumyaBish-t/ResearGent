/**
 * useAgentStore — the central nervous system bridging the SSE stream, the 2D
 * React overlay, and the Three.js canvas.
 *
 * Owns:
 *   - SSE stream lifecycle for /api/research
 *   - Live node/edge state for the 3D scene
 *   - Auth state (current user, usage snapshot)
 *   - Thread + turn history (sidebar + follow-ups)
 *   - Paywall state (when quota hits 402)
 *
 * IMPORTANT — EventSource auto-reconnect: when the backend generator finishes
 * it CLOSES the stream, which a native EventSource interprets as a dropped
 * connection and silently reconnects → re-running the entire agent. We defend
 * against this by closing the EventSource ourselves the instant a terminal
 * event (`saved` / `save_skipped` / `error`) arrives.
 *
 * IMPORTANT — credentials: every backend call (EventSource + fetch) MUST send
 * the session cookie. EventSource needs `withCredentials: true`; fetch needs
 * `credentials: "include"`. Without these the user appears un-authed.
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
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://127.0.0.1:8000";

export interface LogEntry {
  id: number;
  ts: number;
  node: string;
  message: string;
  level: "info" | "warn" | "error" | "success";
}

export interface CurrentUser {
  id: string;
  email: string;
  name: string | null;
  picture: string | null;
  is_admin: boolean;
}

export interface UsageSnapshot {
  is_subscribed: boolean;
  is_admin: boolean;
  threads_used_this_month: number;
  threads_limit: number;        // -1 = unlimited
  turns_limit_per_thread: number; // -1 = unlimited
}

export interface ThreadSummary {
  id: string;
  title: string;
  created_at: string;
}

export interface ThreadTurn {
  turn_index: number;
  question: string;
  answer: string;
  confidence: string | null;
  score: number | null;
  sources: Source[];
  run_id: string | null;
  created_at: string;
}

export interface PaywallState {
  open: boolean;
  reason: "thread_cap" | "turn_cap" | null;
  used?: number;
  limit?: number;
}

interface AgentState {
  // ---- auth ----
  authReady: boolean;       // /auth/me fetch resolved (success or 401)
  user: CurrentUser | null;
  usage: UsageSnapshot | null;

  // ---- threads / history ----
  threads: ThreadSummary[];
  currentThreadId: string | null;
  // Full turn list for the open thread — drives the chat-style ConversationModal.
  // The latest in-flight run appears here ONLY after its final event fires.
  threadTurns: ThreadTurn[];

  // ---- paywall ----
  paywall: PaywallState;

  // ---- input / lifecycle ----
  query: string;
  running: boolean;
  finished: boolean;
  runId: string | null;

  // ---- structural UI state (scrollytelling ↔ dashboard) ----
  hasQueried: boolean;
  scrollProgress: number;

  // ---- live graph state (consumed by the 3D scene) ----
  currentActiveNode: NodeId | null;
  previousNode: NodeId | null;
  activeEdge: { from: NodeId; to: NodeId } | null;
  nodeStatuses: Record<NodeId, NodeStatus>;

  // ---- scores + results ----
  confidenceScore: number | null;
  confidenceLabel: string;
  logs: LogEntry[];
  sources: Source[];
  finalOutput: FinalEvent | null;
  savedPath: string | null;
  saveSkippedReason: string | null;
  error: string | null;

  // ---- actions ----
  bootstrap: () => Promise<void>;
  signIn: () => void;
  signOut: () => Promise<void>;
  refreshUsage: () => Promise<void>;
  refreshThreads: () => Promise<void>;
  openThread: (id: string) => Promise<void>;

  setQuery: (q: string) => void;
  setScrollProgress: (p: number) => void;
  startRun: (q?: string) => Promise<void>;
  stopRun: () => void;
  reset: () => void;

  closePaywall: () => void;
  subscribe: () => Promise<void>;
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

// All fetch() calls to the API MUST share these options so the session cookie
// rides along (without `credentials: "include"` the browser drops it on cross
// origin, even for same-origin-looking dev URLs).
const fetchOpts: RequestInit = {
  credentials: "include",
  headers: { "Content-Type": "application/json" },
};

// Lazy-loads Razorpay Checkout JS on first subscribe click.
let razorpayLoaded = false;
function loadRazorpay(): Promise<void> {
  if (razorpayLoaded) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "https://checkout.razorpay.com/v1/checkout.js";
    s.async = true;
    s.onload = () => {
      razorpayLoaded = true;
      resolve();
    };
    s.onerror = () => reject(new Error("Razorpay script failed to load."));
    document.head.appendChild(s);
  });
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
        const e = evt as RunStartedEvent & { thread_id?: string; turn_index?: number };
        set({
          runId: e.run_id,
          currentThreadId: e.thread_id ?? get().currentThreadId,
        });
        log("system", `run ${e.run_id} started`, "info");
        break;
      }

      case "node_complete": {
        const e = evt as NodeCompleteEvent;
        const node = e.node as NodeId;
        const prev = get().currentActiveNode;

        set((s) => {
          const statuses = { ...s.nodeStatuses };
          if (prev && prev !== node) statuses[prev] = "success";
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
        const e = evt as FinalEvent & { thread_id?: string; turn_index?: number };
        set((s) => {
          const statuses = { ...s.nodeStatuses };
          if (s.currentActiveNode) statuses[s.currentActiveNode] = "success";
          statuses.generator = e.error ? "error" : "success";

          // Append the just-finished turn to the live thread conversation so
          // the modal renders the full chat without an extra fetch round-trip.
          const newTurn: ThreadTurn = {
            turn_index: typeof e.turn_index === "number"
              ? e.turn_index
              : s.threadTurns.length,
            question: s.query,
            answer: e.answer || "",
            confidence: e.confidence || null,
            score: typeof e.score === "number" ? e.score : null,
            sources: e.sources || [],
            run_id: e.run_id || null,
            created_at: new Date().toISOString(),
          };
          // Replace if a turn at the same index already exists (e.g. retry),
          // otherwise append. Keeps the list monotonic.
          const existing = s.threadTurns.filter((t) => t.turn_index !== newTurn.turn_index);
          const nextTurns = [...existing, newTurn].sort(
            (a, b) => a.turn_index - b.turn_index,
          );

          return {
            nodeStatuses: statuses,
            finalOutput: e,
            sources: e.sources || [],
            confidenceScore: typeof e.score === "number" ? e.score : s.confidenceScore,
            confidenceLabel: e.confidence || s.confidenceLabel,
            activeEdge: null,
            threadTurns: nextTurns,
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
        // Quota counter + sidebar both moved — refresh.
        void get().refreshUsage();
        void get().refreshThreads();
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
        void get().refreshUsage();
        void get().refreshThreads();
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
    // ---- initial state ----
    authReady: false,
    user: null,
    usage: null,
    threads: [],
    currentThreadId: null,
    threadTurns: [],
    paywall: { open: false, reason: null },

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

    // ---- auth ----
    bootstrap: async () => {
      try {
        const res = await fetch(`${API_BASE}/auth/me`, fetchOpts);
        if (res.ok) {
          const user: CurrentUser = await res.json();
          set({ user });
          // Once authed, hydrate usage + threads in parallel.
          void get().refreshUsage();
          void get().refreshThreads();
        } else {
          set({ user: null });
        }
      } catch {
        set({ user: null });
      } finally {
        set({ authReady: true });
      }
    },

    signIn: () => {
      // Full-page nav — Google's consent screen can't be iframed.
      window.location.href = `${API_BASE}/auth/google`;
    },

    signOut: async () => {
      try {
        await fetch(`${API_BASE}/auth/logout`, { ...fetchOpts, method: "POST" });
      } catch {
        /* ignore — clear local state regardless */
      }
      set({
        user: null,
        usage: null,
        threads: [],
        currentThreadId: null,
        threadTurns: [],
        ...freshRunState(),
        running: false,
        finished: false,
        hasQueried: false,
        scrollProgress: 0,
        query: "",
      });
    },

    refreshUsage: async () => {
      try {
        const res = await fetch(`${API_BASE}/api/usage`, fetchOpts);
        if (res.ok) set({ usage: await res.json() });
      } catch {
        /* non-fatal */
      }
    },

    refreshThreads: async () => {
      try {
        const res = await fetch(`${API_BASE}/api/threads`, fetchOpts);
        if (res.ok) {
          const data = await res.json();
          set({ threads: data.threads || [] });
        }
      } catch {
        /* non-fatal */
      }
    },

    openThread: async (id) => {
      closeStream();
      try {
        const res = await fetch(`${API_BASE}/api/threads/${id}`, fetchOpts);
        if (!res.ok) return;
        const data = await res.json();
        const rawTurns: any[] = data.turns || [];
        const turns: ThreadTurn[] = rawTurns.map((t) => ({
          turn_index: t.turn_index,
          question: t.question,
          answer: t.answer || "",
          confidence: t.confidence || null,
          score: typeof t.score === "number" ? t.score : null,
          sources: (t.sources as Source[]) || [],
          run_id: t.run_id || null,
          created_at: t.created_at,
        }));
        const last = turns[turns.length - 1];
        set({
          ...freshRunState(),
          running: false,
          finished: true,
          hasQueried: true,
          currentThreadId: id,
          threadTurns: turns,
          query: last?.question || "",
          finalOutput: last
            ? ({
                run_id: last.run_id || "",
                answer: last.answer || "",
                confidence: last.confidence || "",
                score: last.score ?? 0,
                sources: last.sources || [],
                rewrite_attempts: 0,
                reflection_attempts: 0,
                web_used: false,
              } as unknown as FinalEvent)
            : null,
          sources: last?.sources || [],
          confidenceScore: typeof last?.score === "number" ? last.score : null,
          confidenceLabel: last?.confidence || "",
          logs: turns.map((t) => ({
            id: logSeq++,
            ts: new Date(t.created_at).getTime(),
            node: "history",
            message: `Q${t.turn_index + 1}: ${t.question.slice(0, 80)}`,
            level: "info" as const,
          })),
          // Mark every node "success" for the dashboard snapshot.
          nodeStatuses: NODES.reduce(
            (acc, n) => {
              acc[n.id] = "success";
              return acc;
            },
            {} as Record<NodeId, NodeStatus>,
          ),
        });
      } catch {
        /* non-fatal */
      }
    },

    setQuery: (q) => set({ query: q }),

    setScrollProgress: (p) => {
      const q = Math.round(Math.min(1, Math.max(0, p)) * 100) / 100;
      if (q !== get().scrollProgress) set({ scrollProgress: q });
    },

    startRun: async (q) => {
      const question = (q ?? get().query).trim();
      if (!question || get().running) return;

      // Pre-flight: refresh usage so the paywall trigger has fresh numbers.
      // (The server will reject too, but we'd rather not flash the dashboard
      // for half a second before bouncing back.)
      const u = get().usage;
      const onFollowUp = !!get().currentThreadId && get().finished;
      if (u && !u.is_admin && !u.is_subscribed) {
        if (!onFollowUp && u.threads_limit !== -1 && u.threads_used_this_month >= u.threads_limit) {
          set({
            paywall: {
              open: true,
              reason: "thread_cap",
              used: u.threads_used_this_month,
              limit: u.threads_limit,
            },
          });
          return;
        }
      }

      closeStream();
      // New thread → clear prior conversation; follow-up → keep it.
      set((s) => ({
        ...freshRunState(),
        query: question,
        hasQueried: true,
        currentThreadId: onFollowUp ? s.currentThreadId : null,
        threadTurns: onFollowUp ? s.threadTurns : [],
      }));

      const params = new URLSearchParams({ q: question });
      if (onFollowUp && get().currentThreadId) {
        params.set("thread_id", get().currentThreadId as string);
      }
      const url = `${API_BASE}/api/research?${params.toString()}`;
      es = new EventSource(url, { withCredentials: true });

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

      es.onerror = async () => {
        if (get().finished) return;
        // EventSource doesn't surface HTTP status. Probe /api/usage — if the
        // user is over quota, that explains the failure → paywall. Otherwise
        // it's a real connection problem.
        closeStream();
        try {
          await get().refreshUsage();
          const usage = get().usage;
          const isFollowUp = !!get().currentThreadId && get().hasQueried;
          if (usage && !usage.is_admin && !usage.is_subscribed) {
            if (!isFollowUp && usage.threads_limit !== -1 && usage.threads_used_this_month >= usage.threads_limit) {
              set({
                running: false,
                finished: true,
                paywall: {
                  open: true,
                  reason: "thread_cap",
                  used: usage.threads_used_this_month,
                  limit: usage.threads_limit,
                },
              });
              return;
            }
            if (isFollowUp && usage.turns_limit_per_thread !== -1) {
              // Best-effort: turn cap surfacing without a per-thread count.
              set({
                running: false,
                finished: true,
                paywall: { open: true, reason: "turn_cap", limit: usage.turns_limit_per_thread },
              });
              return;
            }
          }
        } catch {
          /* fall through */
        }
        set({
          error: "connection to backend lost (is `researgent serve` running?)",
          running: false,
          finished: true,
        });
        log("system", "EventSource connection error", "error");
      };
    },

    stopRun: () => {
      closeStream();
      set({ running: false, finished: true });
    },

    reset: () => {
      closeStream();
      set({
        ...freshRunState(),
        running: false,
        finished: false,
        hasQueried: false,
        scrollProgress: 0,
        currentThreadId: null,
        threadTurns: [],
        query: get().query,
      });
    },

    closePaywall: () => set({ paywall: { open: false, reason: null } }),

    subscribe: async () => {
      try {
        // 1. Backend creates a one-time ORDER (not a subscription).
        const res = await fetch(`${API_BASE}/billing/checkout`, {
          ...fetchOpts,
          method: "POST",
        });
        if (!res.ok) {
          const txt = await res.text();
          log("billing", `checkout failed: ${txt}`, "error");
          return;
        }
        const cfg = await res.json();

        // 2. Open Razorpay Checkout against that order.
        await loadRazorpay();
        // @ts-expect-error — Razorpay is loaded globally by the script tag above.
        const rz = new window.Razorpay({
          key: cfg.key_id,
          order_id: cfg.order_id,
          amount: cfg.amount,           // paise
          currency: cfg.currency,
          name: "ResearGent",
          description: `Lifetime unlock · ₹${cfg.price_inr}`,
          prefill: { name: cfg.customer.name, email: cfg.customer.email },
          theme: { color: "#22d3ee" },
          // 3. On success, verify the signature server-side. The verify call
          // is the source of truth that flips is_subscribed — never trust the
          // client-side success event alone.
          handler: async (resp: any) => {
            try {
              const v = await fetch(`${API_BASE}/billing/verify`, {
                ...fetchOpts,
                method: "POST",
                body: JSON.stringify({
                  razorpay_order_id: resp.razorpay_order_id,
                  razorpay_payment_id: resp.razorpay_payment_id,
                  razorpay_signature: resp.razorpay_signature,
                }),
              });
              if (!v.ok) {
                const txt = await v.text();
                log("billing", `verify failed: ${txt}`, "error");
                return;
              }
              await get().refreshUsage();
              set({ paywall: { open: false, reason: null } });
              log("billing", "lifetime unlock activated", "success");
            } catch (e: any) {
              log("billing", `verify error: ${e?.message || e}`, "error");
            }
          },
        });
        rz.open();
      } catch (e: any) {
        log("billing", `subscribe error: ${e?.message || e}`, "error");
      }
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
