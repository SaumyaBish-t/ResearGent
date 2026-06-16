"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useAgentStore } from "@/lib/store";

/**
 * Shown when /api/research returns 402 — or when the pre-flight check in the
 * store catches the cap client-side. Razorpay Checkout opens via the store's
 * `subscribe()` action; the webhook flips entitlement and the store polls
 * `/api/usage` until is_subscribed flips true.
 */
export default function PaywallModal() {
  const paywall = useAgentStore((s) => s.paywall);
  const closePaywall = useAgentStore((s) => s.closePaywall);
  const subscribe = useAgentStore((s) => s.subscribe);

  const title =
    paywall.reason === "turn_cap"
      ? "Follow-up limit reached"
      : "Monthly research limit reached";

  const body =
    paywall.reason === "turn_cap"
      ? "Free threads cap at 3 questions each. Unlock lifetime access for unlimited follow-ups across every thread."
      : "Free accounts get 3 researches per month. Unlock lifetime access — pay once, unlimited researches forever.";

  return (
    <AnimatePresence>
      {paywall.open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="pointer-events-auto absolute inset-0 z-30 flex items-center justify-center bg-black/70 backdrop-blur-md"
          onClick={closePaywall}
        >
          <motion.div
            initial={{ scale: 0.96, y: 20, opacity: 0 }}
            animate={{ scale: 1, y: 0, opacity: 1 }}
            exit={{ scale: 0.97, opacity: 0 }}
            transition={{ type: "spring", stiffness: 220, damping: 24 }}
            className="glass relative w-[min(480px,94vw)] overflow-hidden rounded-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="h-px w-full bg-gradient-to-r from-transparent via-accent/60 to-transparent" />

            <div className="p-7">
              <div className="mb-3 flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-warn" />
                <span className="font-mono text-[10px] uppercase tracking-[0.32em] text-ink-dim">
                  upgrade
                </span>
              </div>

              <h2 className="text-[20px] font-semibold tracking-tight text-ink">
                {title}
              </h2>

              <p className="mt-2 text-[13.5px] leading-relaxed text-ink-dim">
                {body}
              </p>

              {typeof paywall.used === "number" && typeof paywall.limit === "number" && (
                <div className="mt-4 flex items-center justify-between rounded-lg border border-line bg-white/[0.02] px-3 py-2 font-mono text-[11px]">
                  <span className="uppercase tracking-widest text-ink-mute">
                    used this month
                  </span>
                  <span className="tabular-nums text-ink">
                    {paywall.used} / {paywall.limit}
                  </span>
                </div>
              )}

              <div className="mt-6 rounded-xl border border-line bg-white/[0.02] p-4">
                <div className="flex items-baseline justify-between">
                  <span className="font-mono text-[10px] uppercase tracking-[0.28em] text-ink-mute">
                    lifetime · one-time
                  </span>
                  <div>
                    <span className="text-[24px] font-semibold tracking-tight text-ink">
                      ₹499
                    </span>
                    <span className="ml-1 text-[12px] text-ink-mute">once</span>
                  </div>
                </div>
                <ul className="mt-3 space-y-1.5 text-[12.5px] text-ink-dim">
                  <li>· Unlimited researches forever</li>
                  <li>· Unlimited follow-ups per thread</li>
                  <li>· Priority on rate-limited providers</li>
                </ul>
              </div>

              <div className="mt-6 flex items-center gap-3">
                <button
                  onClick={subscribe}
                  className="flex-1 rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-[#0a0d14] transition hover:bg-cyan-300"
                >
                  Unlock with Razorpay
                </button>
                <button
                  onClick={closePaywall}
                  className="rounded-xl border border-line px-4 py-2.5 text-sm text-ink-dim transition hover:border-accent/40 hover:text-accent"
                >
                  Later
                </button>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
