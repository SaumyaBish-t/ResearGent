---
title: Index Example
tags: [meta, example, moc]
---

# Index Example  (also called a "Map of Content" or MOC)

An **index note** is a hand-curated table of contents for a topic area.
Its only job is to LINK OUT to all the atomic notes in some domain so
they're easy for both you and the agent to discover.

## Why bother

The agent already walks wikilinks during retrieval, but those edges are
*per-note*. An index note creates a "hub" — many concepts converge
through it. When the agent retrieves an index, it picks up everything it
links to as related context. That's high-leverage.

## Pattern

```markdown
# Index: Retrieval Methods

## Foundations
- [[Cosine Similarity]]
- [[Dense Embeddings]]
- [[BM25]]

## Hybrid + fusion
- [[Reciprocal Rank Fusion]]
- [[Hybrid Retrieval]]

## Quality + correction
- [[Corrective RAG]]
- [[Self-RAG Reflection Tokens]]
- [[Re-ranking with Cross-Encoders]]

## Open questions
- [[Why does hybrid sometimes UNDER-perform dense?]]
```

## When to create one

- Once you've written 5+ atomic notes in the same domain
- When you find yourself searching for the same group of notes repeatedly
- When you want the agent to "see the whole map" of a topic in one chunk

## When NOT to create one

- For domains where you have only 1-2 notes — premature
- Just to look organized — folders work fine for navigation

## See also

- [[README]]
- [[Atomic Note Example]]
