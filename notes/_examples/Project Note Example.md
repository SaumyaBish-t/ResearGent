---
title: Project Note Example
tags: [meta, example, projects]
status: in-progress
---

# Project Note Example

A **project note** anchors all knowledge tied to one concrete project.
Unlike atomic notes (which describe a concept in the abstract), this is
where you record:

- What you're building / researching
- What decisions you've made and why
- Open questions
- Links to the underlying concepts you're using

The agent treats this like any other note, but its DENSE WIKILINK
STRUCTURE makes it a great seed for graph expansion: ask anything about
this project and the agent walks out to all the concepts it depends on.

## Decisions

- We chose hybrid retrieval (see [[Atomic Note Example]]) over pure dense
  because exact-term queries failed in early tests.
- Embedder is `nomic-embed-text` via Ollama. Decision driven by latency +
  zero monthly cost.

## Open questions

- Should we add graph traversal at 2 hops, not 1? Unknown trade-off.
- What's the right confidence threshold for triggering web fallback?

## Related concepts

- [[Atomic Note Example]]
- [[Index Example]]
- [[Daily Research Log Example]]

## Status updates

(Add dated entries here as the project evolves. Each becomes searchable
context for "what did we conclude about X" questions later.)
