"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useAgentStore, API_BASE } from "@/lib/store";

/**
 * Literature review result modal.
 * Opens when reviewMarkdown is populated. Shows the full structured review
 * with markdown rendering. Esc to dismiss.
 */
export default function ReviewModal() {
  const reviewMarkdown = useAgentStore((s) => s.reviewMarkdown);
  const reviewTitle = useAgentStore((s) => s.reviewTitle);
  const reviewRunning = useAgentStore((s) => s.reviewRunning);
  const reviewSections = useAgentStore((s) => s.reviewSections);
  const reviewTrace = useAgentStore((s) => s.reviewTrace);
  const currentReviewId = useAgentStore((s) => s.currentReviewId);
  const [dismissed, setDismissed] = useState(false);
  const [pdfDownloading, setPdfDownloading] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);

  const md = reviewMarkdown;

  const downloadPdf = async () => {
    if (!currentReviewId || pdfDownloading) return;
    setPdfDownloading(true);
    setPdfError(null);
    try {
      const response = await fetch(`${API_BASE}/api/reviews/${currentReviewId}/pdf`, {
        credentials: "include",
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `Download failed (${response.status})`);
      }
      if (!response.headers.get("content-type")?.includes("application/pdf")) {
        throw new Error("The server did not return a PDF file.");
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `literature-review-${currentReviewId}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) {
      setPdfError(error instanceof Error ? error.message : "Could not download the PDF.");
    } finally {
      setPdfDownloading(false);
    }
  };

  // Re-open when new review arrives, starts running, or review ID changes
  useEffect(() => {
    if (md || reviewRunning) setDismissed(false);
  }, [md, reviewRunning, currentReviewId]);

  // Esc to close
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setDismissed(true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const open = !!md && !dismissed;
  if (!open && !reviewRunning) return null;

  return (
    <AnimatePresence mode="wait">
      {(open || reviewRunning) && (
        <motion.div
          key="review-modal"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="pointer-events-auto absolute inset-0 z-40 flex items-center justify-center bg-black/70 p-6 backdrop-blur-xl"
          onClick={() => setDismissed(true)}
        >
          <motion.div
            initial={{ scale: 0.96, y: 20, opacity: 0 }}
            animate={{ scale: 1, y: 0, opacity: 1 }}
            exit={{ scale: 0.97, opacity: 0 }}
            transition={{ type: "spring", stiffness: 220, damping: 24 }}
            style={{ background: "rgba(10, 13, 20, 0.96)" }}
            className="glass relative flex max-h-[88vh] w-[min(1000px,96vw)] flex-col overflow-hidden rounded-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Accent rail */}
            <div className="h-px w-full bg-gradient-to-r from-transparent via-accent/60 to-transparent" />

            {/* Header */}
            <header className="flex items-center justify-between gap-4 px-7 pt-5 pb-3 shrink-0">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      reviewRunning ? "bg-accent animate-pulse" : "bg-good"
                    }`}
                  />
                  <span className="font-mono text-[11.5px] uppercase tracking-[0.32em] text-ink-dim">
                    {reviewRunning ? "generating review…" : "literature review"}
                  </span>
                </div>
                {reviewTitle && (
                  <h2 className="mt-1.5 font-mono text-sm text-ink/80 truncate max-w-[600px]">
                    {reviewTitle}
                  </h2>
                )}
                {reviewSections.length > 0 && reviewRunning && (
                  <p className="mt-0.5 font-mono text-[10px] text-ink-mute/70">
                    {reviewSections.length} sections planned
                  </p>
                )}
              </div>

              <div className="flex items-center gap-3">
                {currentReviewId && !reviewRunning && (
                  <button
                    onClick={downloadPdf}
                    disabled={pdfDownloading}
                    className="rounded-lg border border-accent/30 bg-accent/10 px-3.5 py-1.5 font-mono text-[10px] uppercase tracking-widest text-accent transition hover:bg-accent/20 hover:border-accent/50 disabled:cursor-wait disabled:opacity-60"
                  >
                    {pdfDownloading ? "Preparing PDF…" : "Download PDF"}
                  </button>
                )}
                <button
                  onClick={() => setDismissed(true)}
                  className="rounded-lg px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-widest text-ink-dim transition hover:text-ink hover:bg-white/[0.04]"
                >
                  esc
                </button>
              </div>
            </header>
            {pdfError && (
              <p role="alert" className="px-7 pb-2 font-mono text-xs text-red-400">
                PDF download failed: {pdfError}
              </p>
            )}

            {/* Body — loading or markdown */}
            {reviewRunning && !md ? (
              <div className="flex-1 overflow-y-auto px-7 py-8">
                <div className="mx-auto flex max-w-2xl flex-col gap-5">
                  <div className="flex items-center gap-3 border-b border-line pb-4">
                    <div className="flex gap-1">
                      <span className="h-1.5 w-1.5 rounded-full bg-accent animate-bounce [animation-delay:0ms]" />
                      <span className="h-1.5 w-1.5 rounded-full bg-accent animate-bounce [animation-delay:150ms]" />
                      <span className="h-1.5 w-1.5 rounded-full bg-accent animate-bounce [animation-delay:300ms]" />
                    </div>
                    <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-ink-mute">
                      Review activity
                    </span>
                  </div>
                  {reviewTrace.length === 0 ? (
                    <p className="font-mono text-xs text-ink-dim">Starting the review…</p>
                  ) : (
                    <ol className="flex flex-col gap-3" aria-live="polite">
                      {reviewTrace.map((step, index) => (
                        <li key={`${index}-${step}`} className={`flex gap-3 font-mono text-xs ${index === reviewTrace.length - 1 ? "text-ink" : "text-ink-dim"}`}>
                          <span className="mt-0.5 text-accent">{index === reviewTrace.length - 1 ? "›" : "✓"}</span>
                          <span>{step}</span>
                        </li>
                      ))}
                    </ol>
                  )}
                </div>
              </div>
            ) : (
              <div className="flex-1 overflow-y-auto px-7 py-4">
                <div className="prose prose-invert prose-sm max-w-none
                  prose-headings:font-mono prose-headings:text-ink/90 prose-headings:tracking-tight
                  prose-h1:text-lg prose-h2:text-base prose-h3:text-sm
                  prose-p:text-ink/80 prose-p:leading-relaxed
                  prose-code:font-mono prose-code:text-accent/90 prose-code:bg-white/[0.04] prose-code:rounded prose-code:px-1
                  prose-pre:bg-white/[0.03] prose-pre:border prose-pre:border-line
                  prose-a:text-accent/80 prose-a:no-underline hover:prose-a:underline
                  prose-blockquote:border-accent/30 prose-blockquote:text-ink-dim
                  prose-strong:text-ink">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {md}
                  </ReactMarkdown>
                </div>
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
