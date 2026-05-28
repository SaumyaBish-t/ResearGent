# ResearGent — Obsidian Plugin

Sidebar agent that runs the full ResearGent pipeline (plan → retrieve → critique → rewrite → web cascade → paper discovery → reflect) **inside Obsidian**, streams every node live, and saves answers back to your vault as notes with `[[wikilink]]` citations.

## Architecture

```
   Obsidian Sidebar Panel  ◄──── live SSE stream ────  ResearGent FastAPI
   (this plugin)                                       (`researgent serve`)
        │
        ▼
   Vault filesystem (write answer as note)
```

The plugin **does NOT bundle an LLM** — it calls the ResearGent backend you already run locally. That means: same models, same cascade, same observability log as your CLI runs.

## What it adds inside Obsidian

| Feature | Where |
|---|---|
| Sidebar view with question input, live trace, answer, sources | Right sidebar (toggleable) |
| `Open ResearGent sidebar` | Command palette + ribbon icon |
| `Research the current note` | Command palette — passes the note body as context |
| `Research the current selection` | Command palette — highlight text first |
| `Save answer as note` | Button inside the sidebar — writes to `<OutputFolder>/<date>/<slug>.md` with wikilink citations |
| Backend connection test | Settings tab → "Ping" button |

## Install

### Prerequisites

1. **ResearGent backend running.** From the repo root:
   ```powershell
   uv run researgent serve
   ```
   It listens on `http://127.0.0.1:8000` by default.

2. **Node + npm** to build the plugin. ([nodejs.org](https://nodejs.org))

### Build the plugin

```powershell
cd obsidian-plugin
npm install
npm run build
```

This produces `main.js` next to `manifest.json` and `styles.css`.

### Drop into your vault

Copy these three files into `<YourVault>/.obsidian/plugins/researgent/`:

```
manifest.json
main.js
styles.css
```

(Create the folder if it doesn't exist.)

### Enable

1. Open Obsidian → **Settings → Community plugins**
2. If you've never installed a community plugin: turn off **Restricted Mode**
3. Find **ResearGent** in the installed list → **Enable**
4. Click the **book-open** ribbon icon on the left to open the sidebar
5. (Optional) Go to **Settings → ResearGent → Ping** to verify the backend connection

## Use

- **Quick question:** click the ribbon icon, type into the sidebar, press `Cmd/Ctrl+Enter`.
- **Research the open note:** `Cmd/Ctrl+P` → "ResearGent: Research the current note". The note's body is appended as context so the agent grounds its answer in what *you've* already written.
- **Research a highlight:** select text in any editor → command palette → "ResearGent: Research the current selection".
- **Save the answer:** click **Save answer as note**. The note lands at `<vault>/<OutputFolder>/<YYYY-MM-DD>/<slug>.md` with:
  - YAML frontmatter (run_id, confidence, sources count, tags)
  - The answer with `[Sn]` citations
  - A `## Sources` section where vault-resident citations become `[[wikilinks]]` (Obsidian backlinks auto-resolve them)
  - A `## Provenance` block with the agent's run metadata

## Settings

| Setting | What it controls |
|---|---|
| **Backend URL** | Where ResearGent's FastAPI lives. Change if you ran `researgent serve --port 9000`. |
| **Default top-k** | Total chunk budget the retriever shares across sub-questions. |
| **Output folder** | Subfolder for saved answers (default `ResearGent`). A date subfolder is added automatically. |
| **Use active note as context** | If on, "Research the current note" passes the body to the agent. Turn off for pure question answering. |
| **Auto-open saved notes** | After saving, immediately open the new note. |

## Why a plugin and not just the web UI?

The web UI (`http://localhost:8000`) works fine in a browser. The plugin adds:

- **Native vault writes** — answers land in your vault using Obsidian's own API. No URI scheme, no permission prompts.
- **Live wikilinks** — citations to vault notes become real Obsidian backlinks. The graph view shows research notes as nodes in your knowledge graph.
- **Active-note context** — research a note without copy-pasting.
- **Selection-based queries** — highlight in any editor, run the command.
- **Theme-aware UI** — adapts to your Obsidian theme automatically.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Ping" fails | Backend not running. Run `uv run researgent serve` in another terminal. |
| Sidebar opens empty | Disable & re-enable the plugin. Check developer console (Ctrl+Shift+I) for errors. |
| `main.js` missing after build | `cd obsidian-plugin && npm install` first, then `npm run build`. |
| Citations to vault notes don't become wikilinks | Make sure the note name matches a real note in your vault. Citations use exact name (case-insensitive). |
| Backend at a different host | Update **Settings → ResearGent → Backend URL** and click Ping. |
