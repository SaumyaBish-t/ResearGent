---
title: README — How this notes folder works
tags: [meta, getting-started]
---

# How this notes folder works

This folder is your **knowledge base**. ResearGent reads every `.md` file
in it (and any subfolder) and uses the contents as the AI's local memory.

## The only conventions you need

1. **One concept per file.** Keep notes small and focused. A note about
   *Mixture of Experts* should not also explain *Sparse Computation* —
   those are two separate notes that link to each other.

2. **Use `[[wikilinks]]` to connect notes.** When you mention a concept
   that has its own note, link to it:

       MoE pattern uses [[Sparse Computation]] to scale capacity.

   When the agent retrieves a note that contains a wikilink, it
   automatically pulls the linked note as additional context. That's the
   "AI brain" behavior — your notes form a graph, not a flat dump.

3. **Use `#tags` for cross-cutting categories.** Tags are for
   classifications that span many notes (`#ml`, `#read-later`,
   `#in-progress`, `#paper-summary`).

4. **Add YAML frontmatter at the top** of important notes:

       ---
       title: Mixture of Experts
       tags: [ml, architectures]
       ---

   The `title` overrides the filename for display + wikilink resolution.
   `tags:` here merge with inline `#tags` in the body.

## Folder structure — totally optional

Folders don't matter for retrieval. The wikilink graph is what matters.
Use folders if they help YOU navigate visually. A reasonable default:

    notes/
    ├── README.md              ← this file
    ├── _examples/             ← reference patterns (delete once you understand)
    ├── concepts/              ← atomic single-concept notes
    ├── projects/              ← project-specific knowledge
    ├── sources/               ← quotes / extracts from papers, books, articles
    └── ResearGent/            ← agent-written answers (auto-created)

Or just put every note flat in `notes/`. Both work identically.

## The compounding-knowledge loop

```
1. WRITE   →  Drop a thought into a new .md file in your editor.
              Link to existing concepts with [[wikilinks]] as they come up.

2. INGEST  →  `uv run researgent vault-ingest`  (re-run after meaningful edits)
              This re-indexes every note, rebuilds embeddings + BM25 + graph.

3. ASK     →  `uv run researgent serve`  and open the web UI,
              OR `uv run researgent research "..."`.
              The agent retrieves YOUR notes + walks YOUR wikilink graph.

4. SAVE    →  `--save-to-vault` (CLI) or "Save as Note" button (web UI)
              writes the answer back as a new note that LINKS to the notes
              it cited. Your graph gets denser.

5. REPEAT  →  Each answer makes the brain stronger. The notes you keep
              shape what the agent retrieves on future questions.
```

## What kinds of notes are most useful?

**Most useful** (high retrieval value):
- Conceptual definitions ("What is X? How does X work?")
- Project status / decisions ("We chose Y because Z")
- Lessons learned ("This DIDN'T work because ...")
- Comparisons ("X vs Y: trade-offs are ...")
- Source extracts (key passages from a paper, with your commentary)

**Less useful** (lower retrieval value):
- Daily journal entries with no factual content
- Meeting transcripts with no extraction
- Pure to-do lists (the agent doesn't act on tasks; use a task app)

## Conventions for the agent's saved notes

When the agent writes back, you get:
- `ResearGent/YYYY-MM-DD/<question-slug>.md`
- Frontmatter with `confidence`, `web_used`, `papers_used`, `run_id`
- `## Sources` section with `[[wikilink]]` references when the citation
  matches a real note in your folder (auto-becomes a backlink)
- `## Provenance` block with agent metadata

You can MOVE these into your main concept/project folders later if they
prove worth keeping long-term, or delete them. They're starting points.

## See also

- [[Atomic Note Example]] — the smallest useful note shape
- [[Project Note Example]] — how to capture project-specific knowledge
- [[Index Example]] — how to make a "table of contents" note (MOC)
- [[Daily Research Log Example]] — capturing in-flight thinking
