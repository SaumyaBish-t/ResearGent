# Research Quality Benchmark (excluding provenance and originality)

This benchmark measures local paper retrieval quality and defines human checks
for answer accuracy, citation support, literature-review quality, and confidence
calibration. It contains 24 questions, eight from each indexed domain, with a
manually specified expected paper filename and reference answer points.

## Run the labeled retrieval benchmark

```powershell
.\.venv\Scripts\python.exe -m src.main eval-retrieval eval_suites/research_quality_v1.yaml
```

The retrieval command runs no generator or LLM judge. It reports:

- **Precision@5**: relevant retrieved chunks divided by five.
- **Recall@5**: expected papers found in the top five divided by expected papers.
- **MRR**: reciprocal rank of the first expected paper.

The expected paper labels are filename matches against the local corpus. These
metrics describe local retrieval only; they do not measure answer correctness.

## Current measured result

Run on 2026-11-13 using the local BM25 index:

| Domain | Questions | Precision@5 | Recall@5 | MRR | No expected paper in top 5 |
|---|---:|---:|---:|---:|---:|
| Agentic AI | 8 | 0.175 | 0.875 | 0.875 | 1 |
| Time series | 8 | 0.125 | 0.625 | 0.562 | 3 |
| Quantitative finance | 8 | 0.175 | 0.875 | 0.875 | 1 |
| **Overall** | **24** | **0.158** | **0.792** | **0.771** | **5** |

The expected paper appeared in the top five for 19 of 24 queries (79.2%). The
five misses were `agentic_rest_react`, `ts_crossformer`, `ts_dtw_lowrank`,
`ts_informer_pemfc`, and `quant_esg_risk`. Inspect the per-query records in
`data/eval/retrieval_runs.jsonl` for the actual ranked filenames.

### Important limitation

This is a BM25-only baseline, not a score for the complete dense+BM25 hybrid
retriever. The configured embedder is local Ollama at
`http://localhost:11434/v1`, and its port was unreachable during this run
(`TcpTestSucceeded: False`). The dense retrieval path consequently failed for
all 24 queries. Start Ollama with the configured `nomic-embed-text` model, set
`retrieval_mode: hybrid` at the top of the benchmark YAML, and rerun before
making claims about hybrid retrieval.

Precision@5 is low partly because this corpus is chunked: chunks from the same
expected paper occupy multiple top-five slots. Report both precision and recall
and inspect ranked results; do not present recall alone as retrieval quality.

## Answer and citation review

The YAML includes `reference_points` for each question. Run the full answer
suite after the embedding service is available:

```powershell
.\.venv\Scripts\python.exe -m src.main eval eval_suites/research_quality_v1.yaml --show-answers
```

That run records answer faithfulness, answer relevance, context precision,
confidence, citations, and latency in `data/eval/runs.jsonl`. The automatic
faithfulness and relevance metrics use an LLM judge, so have a human reviewer
independently grade the 24 generated answers against the reference points and
original cited papers. Record each factual claim as supported, contradicted,
or not established; separately mark whether each citation supports its
associated claim.

## Confidence calibration

After human grading, compare the system's confidence labels with those grades.
For each confidence bucket (low, medium, high), report the number of answers,
the fraction judged correct and properly cited, and the gap between the
confidence level and observed accuracy. A high-confidence answer that fails
human review counts as a calibration error. Do not use the critic's own score
as the correctness label.

## Literature review review rubric

For these three topics, have two reviewers independently rate each item from
1 (poor) to 5 (strong), then resolve disagreements:

- Corrective RAG and self-corrective retrieval for LLM agents.
- ProbSparse Informer methods for forecasting and degradation prediction.
- Reinforcement learning for optimal execution in limit order book simulators.

1. Coverage: does the review include the main relevant papers from the labeled
   set?
2. Synthesis: does it compare and connect findings instead of listing paper
   summaries?
3. Accuracy: are claims consistent with the cited paper text?
4. Citation support: does each citation substantiate its adjacent claim?
5. Balance: does it represent limitations, conflicting results, and uncertainty
   where the sources contain them?

Report the mean and range per rubric item, the number of reviews, reviewer
agreement, and the sources used. This human review complements the automated
retrieval metrics; it should not be replaced by the model confidence score.
