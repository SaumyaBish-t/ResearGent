# ResearGent — Frontend (3D Agentic Network)

A Next.js (App Router) + React Three Fiber interface that visualizes the
ResearGent LangGraph pipeline **live** as it runs. Nodes pulse when active,
edges shoot particles as data flows between agents, and the final cited answer
slides up in a modal.

## Two phases

The whole experience pivots on one store flag, `hasQueried`:

- **Intro (`false`)** — a 3D scrollytelling narrative. A *data-accretion vortex*
  of particles swirls, then magnetically snaps into the 7 agent nodes as you
  scroll. The camera is choreographed along a 7-stop path (`<ScrollControls>`),
  and glassmorphic cards crossfade in beside each spotlighted node.
- **Dashboard (`true`)** — submitting a query detaches the camera from scroll
  and damps it out to a locked "System Dashboard Map View". Nodes light up and
  edges fire particles live from the SSE stream. The search bar springs from
  screen-center down to a docked terminal bar.

## Architecture

```
frontend/
├── app/
│   ├── layout.tsx          root layout
│   ├── globals.css         tailwind + glassmorphism utilities (#050507 void)
│   └── page.tsx            mounts <Scene/> (WebGL) + <Overlay/> (HTML)
├── components/
│   ├── Scene.tsx           R3F <Canvas>; switches scroll-intro ↔ dashboard
│   ├── Particles.tsx       data-accretion vortex → magnetic node snap
│   ├── CameraRig.tsx       ScrollCamera + DashboardCamera + ScrollReporter
│   ├── AgentNode.tsx       one 3D node (matte; pulses/spins by status)
│   ├── Edges.tsx           static edges + flowing data particles
│   ├── Overlay.tsx         HTML overlay container (gates by hasQueried)
│   ├── NarrativeOverlay.tsx scroll-driven glass cards + progress rail
│   ├── SearchBar.tsx       center↔docked morph (layoutId="searchBar")
│   ├── LogPanel.tsx        left slide-in execution trace
│   └── ResultModal.tsx     final markdown answer + sources
└── lib/
    ├── types.ts            wire types — mirror src/agent/stream.py exactly
    ├── graph-config.ts     nodes/edges + the 7-stop STORY camera keyframes
    └── store.ts            Zustand store — opens SSE, reduces events  ← brain
```

`lib/store.ts` is the **central nervous system**: it opens an `EventSource`
against the backend's `/api/research` SSE endpoint, reduces each event into
flat state, and drives both the 3D scene and the 2D overlay. It exposes the two
structural UI flags `hasQueried` and `scrollProgress` on top of the live agent
state. The SSE/event-reduction logic was **not** touched by the scrollytelling
work — only additive UI state.

> Respects `prefers-reduced-motion`: the vortex slows, camera damping tightens,
> and overlay/scroll-hint animations are disabled.

## Prerequisites

The Python backend must be running and reachable:

```bash
# from the repo root
researgent serve            # → http://localhost:8000
```

The backend now sends CORS headers for `http://localhost:3000` by default
(configurable via `CORS_ALLOW_ORIGINS` in the repo-root `.env`).

## Run

```bash
cd frontend
cp .env.local.example .env.local      # points at http://localhost:8000
npm install
npm run dev                            # → http://localhost:3000
```

Open http://localhost:3000, type a question, hit **research**, and watch the
network light up.

## Event contract

The store consumes exactly these SSE events (named events, parsed via
`addEventListener`):

| event          | drives                                              |
| -------------- | --------------------------------------------------- |
| `run_started`  | run id, resets graph                                |
| `node_complete`| node status → success, next node → processing/pulse |
| `final`        | answer + sources → result modal                     |
| `saved`        | vault node → success, modal footer                  |
| `save_skipped` | vault node → warn, modal footer                     |
| `error`        | active node → red, error modal                      |

> The store closes the `EventSource` on the first terminal event
> (`saved`/`save_skipped`/`error`) to stop the browser from auto-reconnecting
> and accidentally re-running the whole agent.

## Optional polish

- **Bloom**: add `@react-three/postprocessing` + `postprocessing` and wrap the
  scene in an `<EffectComposer><Bloom/></EffectComposer>` for true glow.
