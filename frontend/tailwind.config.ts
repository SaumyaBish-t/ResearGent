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
        base: "#040409",
        panel: "rgba(8, 10, 18, 0.55)",
        edge: "#1b2230",
        accent: "#22d3ee", // electric cyan — retrieval / active data flow
        violet: "#a855f7", // neon violet — processing
        orange: "#fb923c", // critic
        good: "#34d399", // emerald — success / vault
        warn: "#fbbf24", // amber — low/medium confidence
        bad: "#f87171", // red — error / no-answer
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      keyframes: {
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
      },
      animation: {
        "fade-in": "fade-in 0.4s ease-out",
      },
    },
  },
  plugins: [],
};

export default config;
