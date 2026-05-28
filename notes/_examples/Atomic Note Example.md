---
title: Atomic Note Example
tags: [meta, example, atomic-notes]
---

# Atomic Note Example

An **atomic note** captures ONE concept. Self-contained, dense, linked.

## Why atomic

A note about *Reciprocal Rank Fusion* should not also explain
*BM25* or *cosine similarity* — those are their own notes. Each
concept gets its own file so:

1. The agent can retrieve exactly the chunk that matters
2. The wikilink graph reflects real conceptual relationships
3. You can reuse the same concept in many contexts without duplication

## What goes in one

```
## Definition
One paragraph. Plain language. The thing's core claim.

## Why it matters
A second paragraph. The problem it solves.

## See also
- [[Related Concept A]]
- [[Related Concept B]]
- [[Project Where You Used It]]
```

## What does NOT belong

- Multi-page explanations → split into multiple atomic notes that link
- Step-by-step tutorials → those are procedures, write a separate
  `[[How to do X]]` note that links to its concept dependencies
- Brainstorm dumps → fine to start there, but distill to atomic notes
  once you understand the concept

## See also

- [[README]]
- [[Index Example]]
- [[Project Note Example]]
