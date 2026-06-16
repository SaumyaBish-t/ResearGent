"use client";

import { motion } from "framer-motion";
import { useAgentStore } from "@/lib/store";

/**
 * Full-bleed sign-in gate shown when `authReady && !user`. Single CTA:
 * Sign in with Google. The 3D scene still renders behind it (the user sees
 * the particle core through the dimmed overlay).
 */
export default function SignInLanding() {
  const signIn = useAgentStore((s) => s.signIn);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="pointer-events-auto absolute inset-0 z-20 flex items-center justify-center bg-black/45 backdrop-blur-sm"
    >
      <div className="glass w-[min(440px,92vw)] rounded-2xl p-8 text-center">
        <div className="mb-2 flex items-center justify-center gap-2 font-mono text-[10px] uppercase tracking-[0.42em] text-ink-mute">
          <span className="h-px w-6 bg-line" />
          corrective-rag · multi-agent
          <span className="h-px w-6 bg-line" />
        </div>

        <h1 className="bg-gradient-to-b from-white via-white to-slate-400 bg-clip-text text-[40px] font-semibold tracking-tight text-transparent">
          ResearGent
        </h1>
        <p className="mx-auto mt-2 max-w-sm text-[13px] leading-relaxed text-ink-dim">
          A living map of agents that retrieve, critique, and rewrite their way
          to a grounded answer.
        </p>

        <button
          onClick={signIn}
          className="mx-auto mt-7 flex items-center gap-3 rounded-xl bg-white px-5 py-3 text-sm font-semibold text-[#0a0d14] transition hover:bg-slate-100"
        >
          <GoogleGlyph />
          Sign in with Google
        </button>

        <p className="mt-6 font-mono text-[10px] uppercase tracking-[0.28em] text-ink-mute">
          free · 3 researches / month
        </p>
      </div>
    </motion.div>
  );
}

function GoogleGlyph() {
  return (
    <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden>
      <path
        fill="#FFC107"
        d="M43.6 20.5H42V20H24v8h11.3C33.9 32.5 29.4 35.5 24 35.5c-6.4 0-11.5-5.1-11.5-11.5S17.6 12.5 24 12.5c2.9 0 5.6 1.1 7.6 2.9l5.7-5.7C33.7 6.4 29.1 4.5 24 4.5 13.2 4.5 4.5 13.2 4.5 24S13.2 43.5 24 43.5c10.6 0 19.5-7.7 19.5-19.5 0-1.2-.1-2.3-.4-3.5z"
      />
      <path
        fill="#FF3D00"
        d="M6.3 14.7l6.6 4.8C14.6 16.1 19 13 24 13c2.9 0 5.6 1.1 7.6 2.9l5.7-5.7C33.7 6.9 29.1 5 24 5 16.3 5 9.7 9 6.3 14.7z"
      />
      <path
        fill="#4CAF50"
        d="M24 43c5 0 9.5-1.9 12.9-5l-6-5c-1.9 1.4-4.3 2.3-6.9 2.3-5.3 0-9.8-3-11.4-7.6l-6.6 5.1C9.4 39 16.1 43 24 43z"
      />
      <path
        fill="#1976D2"
        d="M43.6 20.5H42V20H24v8h11.3c-.9 2.4-2.5 4.4-4.5 5.8l6 5c4.3-3.9 7.2-9.7 7.2-16.3 0-1.2-.1-2.3-.4-3.5z"
      />
    </svg>
  );
}
