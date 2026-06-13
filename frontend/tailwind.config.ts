import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Cyber-physical: deep space void + razor-sharp bioluminescent accents.
        base: "#050609",
        panel: "rgba(10, 13, 20, 0.72)",
        edge: "#1b2230",
        line: "rgba(148, 163, 184, 0.10)",
        accent: "#22d3ee", // electric cyan — retrieval / active data flow
        violet: "#a855f7", // neon violet — processing
        orange: "#fb923c", // critic
        good: "#34d399",   // emerald — success / vault
        warn: "#fbbf24",   // amber — low/medium confidence
        bad: "#f87171",    // red — error / no-answer
        ink: {
          DEFAULT: "#e6edf6",
          dim: "#8a93a6",
          mute: "#5b6478",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      letterSpacing: {
        micro: "0.32em",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(34,211,238,0.18), 0 0 24px rgba(34,211,238,0.16)",
        soft: "0 20px 60px -20px rgba(0,0,0,0.75)",
      },
      keyframes: {
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        "pulse-ring": {
          "0%": { transform: "scale(1)", opacity: "0.6" },
          "100%": { transform: "scale(2.2)", opacity: "0" },
        },
      },
      animation: {
        "fade-in": "fade-in 0.4s ease-out",
        "pulse-ring": "pulse-ring 1.6s ease-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
