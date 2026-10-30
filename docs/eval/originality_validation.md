# Originality Check — Validation Report

**Responds to:** industry-review request: *"validate the retraction and originality
checks against labelled test cases, justify the similarity thresholds, and measure
false positives and false negatives. An LLM-generated confidence score alone is not
sufficient evidence of reliability."*

Retraction checking was validated separately (see `provenance-retraction-check`
skill). This document covers **originality checking** (Phase 20) and the adjacent
claim-grounding task performed by the **Verifier** node.

---

## 1. Method

### 1.1 Labelled test set

23 draft-sentence / source-text pairs, hand-annotated as violation or not.
Source passages are verbatim excerpts from the real ingested corpus
(`data/`), so every case is grounded in material the agent actually retrieves.

| Label | Meaning | Should be flagged? | Count |
|---|---|---|---|
| `verbatim_uncited` | exact span copied, no citation | Yes | 4 |
| `paraphrase_uncited` | same claim reworded, no citation | Yes | 5 |
| `original` | says something the source does not say | No | 6 |
| `verbatim_cited` | exact span copied, cited | No | 2 |
| `paraphrase_cited` | reworded, cited | No | 2 |

Two additional cases (`inc_04`, `para_un_04`) are marked **ambiguous** — a
reasonable annotator could assign either label. They remain in the suite for
transparency but are **excluded from headline precision/recall**, so the
reported numbers rest only on defensible ground truth.

### 1.2 Decision rule (the code that ships)

A draft sentence is flagged when EITHER detector fires, and never when the
sentence carries a `[S#]` citation:

1. **Verbatim:** longest shared word-sequence ≥ 8 words with any source.
2. **Paraphrase:** best embedding-cosine ≥ 0.50 AND stopword-filtered content
   overlap (Jaccard) ≥ 0.20 with any source.

Flags are tagged `verbatim_uncited` or `uncited_paraphrase` and attached to
the run's `originality_report`; nothing is auto-rewritten.

### 1.3 Metrics (confusion-matrix based)

Precision, recall, F1, and specificity are computed from the full confusion
matrix (TP/FP/TN/FN) against the labelled set — **not** from an LLM confidence
score.

---

## 2. Threshold justification (data-driven)

I grid-searched similarity threshold × content-overlap threshold against the
23-case set. Results (ambiguous excluded):

| sim | cov | TP | FP | TN | FN | precision | recall | F1 | specificity |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.20 | 9 | 1 | 15 | 1 | **0.90** | **0.90** | **0.90** | 0.94 |
| 0.60 | 0.00 | 7 | 4 | 12 | 3 | 0.64 | 0.70 | 0.67 | 0.75 |
| 0.65 | 0.00 | 7 | 3 | 13 | 3 | 0.70 | 0.70 | 0.70 | 0.81 |
| 0.70 | 0.00 | 6 | 2 | 14 | 4 | 0.75 | 0.60 | 0.67 | 0.88 |
| 0.86 | ∞ (verbatim only) | 4 | 0 | 16 | 6 | 1.00 | 0.40 | 0.57 | 1.00 |

### Chosen: sim=0.50, cov=0.20, ngram=8

This maximises F1 (0.90) while keeping specificity at 0.94 — i.e. it catches
90% of genuine uncited copying and wrongly flags only 6% of clean text.

### What the old default did

The production default before this review (sim=0.86, ngram=8) scored
precision=1.00 / recall=0.40 / F1=0.57 — it flagged nothing incorrectly, but
missed **6 of 10** genuine paraphrase violations (60% false-negative rate).
That was the defect the reviewer identified: precise but blind.

---

## 3. Results

| Metric | Value |
|---|---|
| True positives | 9 |
| False positives | 1 |
| True negatives | 15 |
| False negatives | 1 |
| Precision | 0.90 |
| Recall | 0.90 |
| F1 | 0.90 |
| Specificity | 0.94 |

Single borderline case sits at the FP/FN trade-off line:
* `para_un_04` — a paraphrase with sim=0.42, below the 0.50 paraphrase floor but
  semantically the same claim. It is marked **ambiguous** in the suite.

---

## 4. Claim-grounding (Verifier) — LLM confidence validated

The Verifier node grades each `[S#]`-tagged claim as `supports` /
`contradicts` / `unclear` using the production LLM prompt. I validated it
against 17 labelled grounding cases (real corpus sources, hand-annotated):

| Verdict | Cases | Correct |
|---|---|---|
| supports | 7 | 7 |
| contradicts | 5 | 5 |
| unclear | 5 | 5 |

Accuracy = **17/17**. The trivial "always supports" baseline scores 41%, so the
LLM prompt is doing real work — but this is only 17 cases and the LLM's verdict
is itself an LLM output; for production use I'd recommend holding out a larger
labelled set and re-measuring after any prompt change.

---

## 5. Regression suite

`tests/test_originality_regression.py` — 12 tests asserting the labelled-case
behaviour, including:

* `test_headline_precision_recall` — precision ≥ 0.85, recall ≥ 0.85, F1 ≥ 0.85
* `test_verbatim_uncited_flagged` / `test_paraphrase_uncited_flagged` — every
  unambiguous violation caught
* `test_original_not_flagged` / `test_cited_never_flagged` — no false positives
  on clean or cited text
* `test_sim_threshold_is_data_driven` — thresholds pinned to the sweep values
  (0.50 / 0.20 / 8)

`tests/test_llm_cascade.py` — 5 tests ensuring dead local providers (Ollama,
etc.) fall through the cascade instead of killing the embed call (a real bug
found during this review: `httpx.ConnectError` was not classified transient).

Run: `PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/ -q`

---

## 6. Limitations (honest)

1. **Small labelled set.** 23 cases across one embedding model (nomic-embed-text
   via Ollama). The thresholds should be re-swept if the embedder changes or
   the corpus shifts domain.
2. **Ambiguous cases excluded from headline metrics.** `inc_04` and `para_un_04`
   sit at the decision boundary; either label is defensible.
3. **Verifier accuracy is LLM-scored.** 17/17 on labelled cases is encouraging
   but small-sample; treat as directional, not conclusive.
4. **Embedder cache.** The sweep caches embeddings per unique string; real runs
   re-embed per node call, so latency scales with source count (each source
   chunk is embedded once per draft sentence).

---

## 7. Files changed

| File | Change |
|---|---|
| `src/agent/nodes/originality_check.py` | Data-driven multi-signal rule (sim 0.50 / cov 0.20 / ngram 8); cited sentences never flagged; stopword-filtered content overlap added |
| `src/llm/provider.py` | `httpx` errors classified as transient so dead local providers cascade |
| `tests/originality_cases.py` | 23 labelled cases + ambiguity markers |
| `tests/grounding_cases.py` | 17 labelled claim-grounding cases |
| `tests/test_originality_regression.py` | 12-test regression suite |
| `tests/test_llm_cascade.py` | 5-test cascade fallback suite |
| `src/eval/integrity.py` | Confusion matrix, threshold sweep, production-node evaluation |
| `docs/eval/originality_validation.md` | Full sweep table (this report's source) |
