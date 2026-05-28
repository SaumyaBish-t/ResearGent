// SSE client for the ResearGent /api/research endpoint.
//
// Why fetch+ReadableStream and not EventSource?
//   1. EventSource doesn't support custom headers, POST bodies, or robust
//      cancellation — features we'll likely want later.
//   2. Obsidian's Electron context fully supports streaming fetch, so we
//      pay nothing for the upgrade.
//   3. Manual SSE parsing lets us cleanly handle the typed event names
//      ("run_started", "node_complete", "final", "error") that sse-starlette
//      emits via the `event:` field.
//
// Cancellation
// ------------
// `streamResearch()` returns an AbortController. Call `.abort()` to stop
// the stream cleanly — used when the user starts a new query before the
// previous one finishes.

import type { StreamEvent } from "./types";

export function streamResearch(
  backendUrl: string,
  question: string,
  options: {
    k?: number;
    context?: string;
    onEvent: (e: StreamEvent) => void;
    onDone?: () => void;
    onError?: (msg: string) => void;
  },
): AbortController {
  const ctrl = new AbortController();
  const params = new URLSearchParams({
    q: question,
    k: String(options.k ?? 8),
  });
  if (options.context) params.set("context", options.context);

  const url = `${backendUrl.replace(/\/+$/, "")}/api/research?${params.toString()}`;

  (async () => {
    try {
      const resp = await fetch(url, {
        method: "GET",
        headers: { Accept: "text/event-stream" },
        signal: ctrl.signal,
      });
      if (!resp.ok || !resp.body) {
        options.onError?.(`HTTP ${resp.status} ${resp.statusText}`);
        return;
      }

      // SSE wire format is plain text, line-delimited, with each event being:
      //   event: <type>
      //   data: <json>
      //   <blank line>
      // We buffer across chunks because TextDecoder can split mid-line.
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        // Split on the SSE record separator: blank line.
        let sepIdx: number;
        while ((sepIdx = buf.indexOf("\n\n")) !== -1) {
          const block = buf.slice(0, sepIdx);
          buf = buf.slice(sepIdx + 2);
          handleBlock(block, options.onEvent);
        }
      }
      options.onDone?.();
    } catch (err: unknown) {
      if ((err as { name?: string })?.name === "AbortError") return;
      options.onError?.((err as Error)?.message || String(err));
    }
  })();

  return ctrl;
}

function handleBlock(block: string, onEvent: (e: StreamEvent) => void): void {
  let eventName: string | undefined;
  let data: string | undefined;
  for (const line of block.split("\n")) {
    if (line.startsWith(":")) continue; // SSE comment / keepalive
    const idx = line.indexOf(":");
    if (idx === -1) continue;
    const field = line.slice(0, idx).trim();
    // Per spec, the value is everything after the FIRST colon, with a
    // single optional leading space stripped.
    const valRaw = line.slice(idx + 1);
    const value = valRaw.startsWith(" ") ? valRaw.slice(1) : valRaw;
    if (field === "event") eventName = value;
    else if (field === "data") data = (data ?? "") + value;
  }
  if (!data) return;
  try {
    const parsed = JSON.parse(data) as StreamEvent;
    // Trust the embedded `type` field if event name was missing.
    onEvent(parsed);
    void eventName; // silence unused warning when not needed
  } catch {
    // Drop malformed payloads silently — the next event recovers naturally.
  }
}

// Light wrapper around fetch for non-streaming status checks (used by
// the settings tab "Test connection" button).
export async function pingBackend(backendUrl: string): Promise<{ ok: boolean; detail: string }> {
  try {
    const resp = await fetch(`${backendUrl.replace(/\/+$/, "")}/api/status`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!resp.ok) return { ok: false, detail: `HTTP ${resp.status}` };
    const data = await resp.json();
    const routing = data?.routing as Record<string, { provider?: string }>;
    if (!routing) return { ok: true, detail: "responding, no routing info" };
    const parts: string[] = [];
    for (const [tier, r] of Object.entries(routing)) {
      if (r.provider) parts.push(`${tier}=${r.provider}`);
    }
    return { ok: true, detail: parts.join("  ") };
  } catch (err) {
    return { ok: false, detail: (err as Error)?.message || String(err) };
  }
}
