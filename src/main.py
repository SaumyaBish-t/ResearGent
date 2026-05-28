"""
ResearGent CLI entrypoint.

Phase 0 commands:
    researgent status         — show provider configuration & routing
    researgent smoke          — call every configured provider with a tiny prompt
    researgent ask "<text>"   — one-shot chat using the REASONING tier

Phases 1+ will add: ingest, retrieve, research, eval, serve.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.llm import ModelTier, chat, list_status

app = typer.Typer(
    help="ResearGent — Agentic Research Engine with Corrective RAG & Self-Reflection",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def status() -> None:
    """Show providers, tier routing, and cascade fallback chains."""
    info = list_status()

    # --- Providers table ---
    p_table = Table(title="Providers", show_header=True, header_style="bold cyan")
    p_table.add_column("Provider")
    p_table.add_column("Configured")
    p_table.add_column("Base URL", overflow="fold")
    p_table.add_column("reasoning / fast / tool / embed", overflow="fold")

    for name, data in info["providers"].items():
        m = data["models"]
        p_table.add_row(
            name,
            "[green]yes[/green]" if data["configured"] else "[red]no[/red]",
            data["base_url"],
            f"{m.get('reasoning')}  /  {m.get('fast')}  /  {m.get('tool')}  /  {m.get('embed')}",
        )
    console.print(p_table)

    # --- Routing + cascade table ---
    r_table = Table(title="Tier Routing (primary + cascade fallback)", header_style="bold cyan")
    r_table.add_column("Tier")
    r_table.add_column("Primary provider")
    r_table.add_column("Primary model", overflow="fold")
    r_table.add_column("Cascade chain (on failure)", overflow="fold")

    for tier, route in info["routing"].items():
        chain = info["cascade"].get(tier, [])
        if "error" in route:
            r_table.add_row(tier, "[red]error[/red]", route["error"], "")
        else:
            cascade_str = " -> ".join(chain) if chain else "-"
            r_table.add_row(tier, route["provider"], route["model"], cascade_str)
    console.print(r_table)


@app.command()
def smoke() -> None:
    """
    Ping each chat tier with a one-line prompt. Confirms wiring + credentials.
    Skips EMBED (use `retrieve` or `ingest` to exercise that path).
    """
    prompt = [
        {"role": "system", "content": "Reply in EXACTLY one short sentence."},
        {"role": "user", "content": "Say hello and name yourself."},
    ]
    for tier in (ModelTier.REASONING, ModelTier.FAST, ModelTier.TOOL):
        console.print(f"\n[bold cyan]>> Testing tier: {tier.value}[/bold cyan]")
        try:
            out = chat(prompt, tier=tier, max_tokens=80)
            console.print(Panel(out.strip(), border_style="green"))
        except Exception as e:
            console.print(Panel(f"[red]{type(e).__name__}: {e}[/red]", border_style="red"))


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question"),
    tier: str = typer.Option("reasoning", help="reasoning | fast"),
) -> None:
    """One-shot chat — no retrieval yet (that's Phase 1)."""
    t = ModelTier(tier)
    messages = [
        {"role": "system", "content": "You are a helpful research assistant. Be concise."},
        {"role": "user", "content": question},
    ]
    out = chat(messages, tier=t)
    console.print(Panel(out.strip(), title=f"answer (tier={tier})", border_style="cyan"))


@app.command(name="status-json")
def status_json() -> None:
    """Machine-readable status (handy for CI / debugging)."""
    print(json.dumps(list_status(), indent=2))


@app.command()
def stats(
    last: int = typer.Option(0, help="Only consider the last N calls (0 = all)"),
) -> None:
    """
    Show aggregated LLM call statistics from the observability log.

    Counts, success rate, latency, and token usage broken down by tier and
    by (tier, provider) pair. Use this to spot slow providers, broken keys,
    or cascade fallbacks happening silently.
    """
    from src.llm.observability import load_records, summarize

    records = load_records(limit=last or None)
    if not records:
        console.print(
            "[yellow]No LLM calls logged yet.[/yellow]  "
            "Run `researgent smoke` or `rag-ask` first."
        )
        return

    summary = summarize(records)

    # ---- Headline numbers ----
    t = summary["tokens"]
    console.print(
        f"\n[bold]Total calls:[/bold] {summary['total_calls']}    "
        f"[bold]Tokens:[/bold] {t['input']:,} in / {t['output']:,} out / {t['total']:,} total    "
        f"[bold]Cascade fallbacks used:[/bold] {summary['cascade_used']}"
    )

    # ---- By tier ----
    t_table = Table(title="By tier", header_style="bold cyan")
    for col in ("Tier", "Calls", "OK%", "Avg ms", "p95 ms", "Tok in", "Tok out"):
        t_table.add_column(col)
    for tier_name, v in sorted(summary["by_tier"].items()):
        t_table.add_row(
            tier_name,
            str(v["count"]),
            f"{int(v['success_rate'] * 100)}%",
            str(v["avg_ms"]),
            str(v["p95_ms"]),
            f"{v['in_tok']:,}",
            f"{v['out_tok']:,}",
        )
    console.print(t_table)

    # ---- By tier × provider ----
    p_table = Table(title="By tier x provider", header_style="bold cyan")
    for col in ("Tier/Provider", "Calls", "OK%", "Avg ms", "p95 ms"):
        p_table.add_column(col)
    for pair, v in sorted(summary["by_tier_provider"].items()):
        p_table.add_row(
            pair,
            str(v["count"]),
            f"{int(v['success_rate'] * 100)}%",
            str(v["avg_ms"]),
            str(v["p95_ms"]),
        )
    console.print(p_table)


# ---------------------------------------------------------------------------
# Phase 1 commands: ingest / rag-ask / retrieve / store
# ---------------------------------------------------------------------------


@app.command(name="vault-ingest")
def vault_ingest(
    path: str = typer.Argument(
        "",
        help="Path to a folder of .md notes. Defaults to NOTES_FOLDER_PATH "
             "(or OBSIDIAN_VAULT_PATH, or ./notes) from .env if blank.",
    ),
) -> None:
    """
    Ingest a folder of markdown notes as the local knowledge corpus.

    Works with ANY tool that edits .md files — VS Code, Obsidian, Logseq,
    Foam, vim — not specific to Obsidian. We just parse the standard
    `[[wikilink]]` + `#tag` conventions plus optional YAML frontmatter.

    Walks every .md file recursively, chunks on heading boundaries, embeds
    into the vector store. Re-running is safe (idempotent by content hash).
    """
    from pathlib import Path as _P
    from src.config import settings
    from src.ingest import ingest_vault

    notes_path = path or settings.resolve_notes_folder()
    if not notes_path:
        console.print(
            "[red]No notes folder configured.[/red]\n"
            "  Either pass a path explicitly, OR set [cyan]NOTES_FOLDER_PATH[/cyan] in .env,\n"
            "  OR put your notes in the [cyan]./notes[/cyan] folder (created by default in this repo)."
        )
        raise typer.Exit(code=1)

    p = _P(notes_path)
    if not p.exists():
        console.print(f"[red]Notes folder not found:[/red] {p}")
        raise typer.Exit(code=1)

    console.print(f"[dim]ingesting from:[/dim] [cyan]{p.resolve()}[/cyan]\n")
    results = ingest_vault(p)
    ok = sum(1 for r in results if "error" not in r)
    total_chunks = sum(r.get("chunks_inserted", 0) for r in results)
    total_tags = len({t for r in results for t in (r.get("tags") or [])})
    total_links = sum(len(r.get("wikilinks") or []) for r in results)
    console.print(
        f"\n[bold green]Done.[/bold green] {ok}/{len(results)} notes ingested, "
        f"{total_chunks} chunks, {total_tags} unique tags, {total_links} wikilinks."
    )


@app.command()
def ingest(
    path: str = typer.Argument(
        "data/papers",
        help="PDF file or directory containing PDFs. Default: data/papers",
    ),
) -> None:
    """
    Ingest a PDF (or every PDF in a directory) into the vector store.

    Re-running on the same file is safe — chunks are replaced by content hash.
    """
    # Import inside the command so the heavy ChromaDB/PyMuPDF deps don't slow
    # down `researgent --help` and the lightweight Phase 0 commands.
    from src.ingest import ingest_directory, ingest_file

    p = Path(path)
    if not p.exists():
        console.print(f"[red]Path not found:[/red] {p}")
        raise typer.Exit(code=1)

    if p.is_file():
        result = ingest_file(p)
        console.print(Panel(json.dumps(result, indent=2), title="ingested", border_style="green"))
        return

    results = ingest_directory(p)
    ok = sum(1 for r in results if "error" not in r)
    total_chunks = sum(r.get("chunks_inserted", 0) for r in results)
    console.print(
        f"\n[bold green]Done.[/bold green] {ok}/{len(results)} files ingested, "
        f"{total_chunks} chunks total."
    )


@app.command(name="rag-ask")
def rag_ask(
    question: str = typer.Argument(..., help="Question to ask the indexed corpus"),
    k: int = typer.Option(5, help="Number of chunks to retrieve"),
    mode: str = typer.Option("hybrid", help="hybrid | naive — retrieval strategy"),
) -> None:
    """
    Retrieve top-k chunks and answer with citations.

    `--mode hybrid` (default) uses dense + BM25 + RRF — wins on exact-term
    queries. `--mode naive` is the dense-only baseline from Phase 1.
    """
    if mode == "hybrid":
        from src.rag import hybrid_rag
        result = hybrid_rag(question, k=k)
    elif mode == "naive":
        from src.rag import naive_rag
        result = naive_rag(question, k=k)
    else:
        console.print(f"[red]Unknown mode: {mode}[/red] (use: hybrid | naive)")
        raise typer.Exit(code=1)

    console.print(Panel(result.formatted(), title=f"answer (mode={mode}, k={k})", border_style="cyan"))


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Query to retrieve chunks for"),
    k: int = typer.Option(5, help="Number of chunks to retrieve"),
    mode: str = typer.Option("hybrid", help="hybrid | naive | bm25"),
) -> None:
    """Show raw retrieved chunks (no LLM call). Useful for debugging retrieval."""
    if mode == "hybrid":
        from src.retrieval import hybrid_retrieve
        chunks = hybrid_retrieve(query, k=k)
        if not chunks:
            console.print("[yellow]No chunks retrieved (corpus empty?).[/yellow]")
            return
        for i, c in enumerate(chunks, start=1):
            preview = c.text.strip().replace("\n", " ")
            preview = preview[:300] + ("..." if len(preview) > 300 else "")
            ranks = []
            if c.dense_rank: ranks.append(f"dense#{c.dense_rank}")
            if c.bm25_rank: ranks.append(f"bm25#{c.bm25_rank}")
            title = f"[S{i}] {c.citation}   signal={c.signal}  rrf={c.rrf_score:.4f}  ({', '.join(ranks)})"
            console.print(Panel(preview, title=title, border_style="cyan"))
    elif mode == "naive":
        from src.retrieval import naive_retrieve
        chunks = naive_retrieve(query, k=k)
        if not chunks:
            console.print("[yellow]No chunks retrieved (corpus empty?).[/yellow]")
            return
        for i, c in enumerate(chunks, start=1):
            preview = c.text.strip().replace("\n", " ")
            preview = preview[:300] + ("..." if len(preview) > 300 else "")
            title = f"[S{i}] {c.citation}   cos={c.score:.3f}"
            console.print(Panel(preview, title=title, border_style="cyan"))
    elif mode == "bm25":
        from src.retrieval import bm25 as bm25_idx
        hits = bm25_idx.search(query, k=k)
        if not hits:
            console.print("[yellow]No BM25 hits (index empty or no term overlap).[/yellow]")
            return
        for i, h in enumerate(hits, start=1):
            preview = h.text.strip().replace("\n", " ")
            preview = preview[:300] + ("..." if len(preview) > 300 else "")
            m = h.metadata
            cit = f"{m.get('source_file','?')} p.{m.get('page_number','?')}"
            title = f"[S{i}] {cit}   bm25={h.score:.3f}"
            console.print(Panel(preview, title=title, border_style="cyan"))
    else:
        console.print(f"[red]Unknown mode: {mode}[/red] (use: hybrid | naive | bm25)")
        raise typer.Exit(code=1)


@app.command()
def research(
    question: str = typer.Argument(..., help="Question to research"),
    k: int = typer.Option(8, help="Total chunks budget across all sub-questions"),
    run_id: str = typer.Option("", help="Stable id for checkpoint replay; auto if blank"),
    no_checkpoint: bool = typer.Option(False, help="Skip SQLite checkpointing (for tests)"),
    save_to_vault: bool = typer.Option(
        False,
        "--save-to-vault",
        help="Write the answer back as a markdown note in your Obsidian vault (Phase 8). "
             "Requires OBSIDIAN_VAULT_PATH set in .env.",
    ),
) -> None:
    """
    Full agentic research — plan / retrieve / critique / rewrite / web /
    paper-discovery / generate / reflect.

    Decomposes complex queries into sub-questions, retrieves with hybrid
    (dense + BM25 + RRF), grades chunks with a fast Critic, rewrites and
    retries on low confidence, falls through arXiv+Semantic Scholar then
    Tavily/Serper/DuckDuckGo web cascade, generates a structured answer
    with grounded [S<n>] citations, and the Reflector audits the draft.

    With --save-to-vault, the answer + sources are written as a new
    markdown note in your Obsidian vault — frontmatter, wikilinks,
    `#researgent` tag included.
    """
    from src.agent import run_agent

    result = run_agent(
        question,
        k=k,
        run_id=run_id or None,
        use_checkpointer=not no_checkpoint,
    )
    title = f"agent (k={k}, run_id={result.run_id})"
    if result.error:
        title += f"  [{result.error}]"
    console.print(Panel(result.formatted(), title=title, border_style="cyan"))

    if save_to_vault:
        from pathlib import Path as _P
        from urllib.parse import quote
        import subprocess, sys
        from src.agent.vault_writer import write_run_to_vault
        from src.config import settings

        notes_folder = settings.resolve_notes_folder()
        if not notes_folder:
            console.print(
                "[red]--save-to-vault needs a notes folder.[/red]  "
                "Set [cyan]NOTES_FOLDER_PATH[/cyan] in .env, or put .md files in [cyan]./notes[/cyan]."
            )
            raise typer.Exit(code=1)
        try:
            note_path = write_run_to_vault(
                vault_path=notes_folder,
                output_subfolder=settings.obsidian_output_folder,
                question=question,
                answer=result.answer,
                sources=result.sources,
                sub_questions=result.sub_questions,
                is_complex=result.is_complex,
                confidence=result.confidence,
                rewrite_attempts=result.rewrite_attempts,
                web_used=result.web_used,
                papers_used=result.papers_used,
                reflection_attempts=result.reflection_attempts,
                run_id=result.run_id,
            )
            console.print(
                f"\n[green]saved to vault:[/green] [cyan]{note_path}[/cyan]"
            )

            # Build the obsidian:// URI and ask the OS to open it. Obsidian
            # registers this scheme on install; if the user runs Obsidian on
            # the same folder, the new note pops open in it. Harmless no-op
            # if Obsidian isn't installed.
            vault_root = _P(notes_folder).resolve()
            try:
                rel = _P(note_path).resolve().relative_to(vault_root)
                vault_name = vault_root.name
                uri = (
                    f"obsidian://open?vault={quote(vault_name)}"
                    f"&file={quote(str(rel).replace(chr(92), '/'))}"
                )
                # OS-specific URI launcher — no extra deps.
                if sys.platform == "win32":
                    subprocess.Popen(["cmd", "/c", "start", "", uri], shell=False)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", uri])
                else:
                    subprocess.Popen(["xdg-open", uri])
                console.print(f"[dim]opening in Obsidian: {uri}[/dim]")
            except Exception:
                # URI open is a nicety, not a critical path — never fail
                # the command because the OS launcher returned non-zero.
                pass
        except Exception as e:
            console.print(f"[red]vault write failed:[/red] {type(e).__name__}: {e}")


@app.command()
def bench(
    query: str = typer.Argument(..., help="Query to benchmark"),
    k: int = typer.Option(5, help="Top-k for both retrievers"),
) -> None:
    """
    Side-by-side: naive (dense-only) vs hybrid (dense+BM25+RRF) for one query.

    Shows which chunks are unique to each retriever, which are found by both
    (the strongest signal), and the rank shifts caused by fusion. Use this on
    your hardest queries to see exactly when hybrid pays off.
    """
    from src.retrieval import naive_retrieve, hybrid_retrieve
    from src.retrieval import bm25 as bm25_idx

    naive = naive_retrieve(query, k=k)
    hybrid = hybrid_retrieve(query, k=k)
    bm25_hits = bm25_idx.search(query, k=k)

    if not naive and not hybrid:
        console.print("[yellow]No results — is the corpus ingested?[/yellow]")
        return

    def _key(source_file: str, chunk_index: int) -> str:
        return f"{source_file}::{chunk_index}"

    naive_keys = {_key(c.source_file, c.chunk_index) for c in naive}
    bm25_keys = {_key(h.metadata.get("source_file","?"), int(h.metadata.get("chunk_index", -1))) for h in bm25_hits}
    hybrid_keys = {_key(c.source_file, c.chunk_index) for c in hybrid}

    # ---- Summary table ----
    t = Table(title=f"Retrieval comparison  -  query: {query!r}", header_style="bold cyan")
    t.add_column("Rank")
    t.add_column("Naive (dense top-k)", overflow="fold")
    t.add_column("BM25 (lexical top-k)", overflow="fold")
    t.add_column("Hybrid (RRF top-k)", overflow="fold")

    for i in range(k):
        n_cell = ""
        if i < len(naive):
            c = naive[i]
            n_cell = f"{c.citation}\ncos={c.score:.3f}"
        b_cell = ""
        if i < len(bm25_hits):
            h = bm25_hits[i]
            m = h.metadata
            b_cell = f"{m.get('source_file','?')} p.{m.get('page_number','?')}\nbm25={h.score:.2f}"
        h_cell = ""
        if i < len(hybrid):
            c = hybrid[i]
            h_cell = f"{c.citation}\nrrf={c.rrf_score:.4f}  [{c.signal}]"
        t.add_row(str(i + 1), n_cell, b_cell, h_cell)
    console.print(t)

    # ---- Overlap stats ----
    both = naive_keys & bm25_keys
    only_naive = naive_keys - bm25_keys
    only_bm25 = bm25_keys - naive_keys
    hybrid_from_naive = hybrid_keys & naive_keys
    hybrid_from_bm25 = hybrid_keys & bm25_keys
    hybrid_from_both = hybrid_keys & both

    console.print(
        f"\n[bold]Overlap:[/bold] "
        f"both={len(both)}  only-dense={len(only_naive)}  only-bm25={len(only_bm25)}  "
        f"|  hybrid pulled {len(hybrid_from_naive)}/{k} from dense, "
        f"{len(hybrid_from_bm25)}/{k} from bm25, "
        f"{len(hybrid_from_both)}/{k} ranked by both"
    )


@app.command()
def doctor() -> None:
    """
    Health check — verifies Ollama is reachable + tests the EMBED tier with
    one short string. Use this to debug slow / hanging ingest before you
    process a 100-page PDF.
    """
    import time as _t
    import httpx
    from src.config import settings
    from src.llm import ModelTier, embed

    # ---- Ollama reachability ----
    console.print("[bold cyan]Ollama check:[/bold cyan]")
    try:
        url = settings.ollama_base_url.rstrip("/v1").rstrip("/") + "/api/tags"
        r = httpx.get(url, timeout=3.0)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            console.print(f"  [green]reachable[/green]  models pulled: {', '.join(models) or '(none)'}")
            # Ollama lists models with a tag (`nomic-embed-text:latest`) but
            # accepts them with or without. Normalize both sides before diffing.
            def _bare(name: str) -> str:
                return name.split(":", 1)[0]
            pulled_bare = {_bare(m) for m in models}
            need = [settings.ollama_model_reasoning, settings.ollama_model_embed]
            missing = [m for m in need if _bare(m) not in pulled_bare]
            if missing:
                console.print(f"  [yellow]missing:[/yellow] {missing}  ->  ollama pull {' '.join(missing)}")
        else:
            console.print(f"  [red]HTTP {r.status_code}[/red]")
    except Exception as e:
        console.print(f"  [red]unreachable[/red]: {type(e).__name__}: {e}")
        console.print(f"  [dim]is Ollama running?  https://ollama.com[/dim]")

    # ---- EMBED tier smoke ----
    console.print("\n[bold cyan]EMBED tier check:[/bold cyan]")
    t0 = _t.perf_counter()
    try:
        vec = embed(["hello world"], tier=ModelTier.EMBED)
        dur = int((_t.perf_counter() - t0) * 1000)
        console.print(f"  [green]OK[/green]  {dur} ms  dim={len(vec[0])}")
    except Exception as e:
        dur = int((_t.perf_counter() - t0) * 1000)
        console.print(f"  [red]FAIL[/red]  {dur} ms  {type(e).__name__}: {e}")


@app.command()
def discover(
    topic: str = typer.Argument(..., help="Topic / query to search papers for"),
    max_results: int = typer.Option(5, help="How many top papers to return"),
) -> None:
    """
    Phase 7 — Standalone paper discovery (arXiv + Semantic Scholar).

    Useful for browsing what the agent's paper_discovery node would pull
    for a given query, without running the full agent loop. Also handy as
    a "find me papers about X" utility on its own.
    """
    from src.retrieval import discover_papers

    console.print(f"[cyan]searching arXiv + Semantic Scholar for:[/cyan] {topic}\n")
    papers = discover_papers(topic, max_results=max_results)
    if not papers:
        console.print("[yellow]No papers found.[/yellow]")
        return

    t = Table(title=f"Top {len(papers)} papers", header_style="bold cyan")
    t.add_column("Score", justify="right")
    t.add_column("Year")
    t.add_column("Title", overflow="fold")
    t.add_column("Citation", overflow="fold")
    t.add_column("Src")
    t.add_column("Cites", justify="right")
    for p in papers:
        t.add_row(
            f"{p.score:.2f}",
            str(p.year or "?"),
            p.title or "(no title)",
            p.citation,
            p.source,
            str(p.citations or ""),
        )
    console.print(t)
    console.print(
        f"\n[dim]Tip:[/dim] download a paper's PDF + drop into data/papers/, "
        f"then [cyan]researgent ingest[/cyan] to add it to your permanent corpus."
    )


@app.command()
def eval(
    suite_path: str = typer.Argument(..., help="Path to a YAML eval suite (see eval_suites/sample.yaml)"),
    skip_metrics: bool = typer.Option(False, help="Run agent but skip RAGAS scoring (faster smoke)"),
    show_answers: bool = typer.Option(False, help="Print each answer inline (verbose)"),
) -> None:
    """
    Phase 6a — Run an evaluation suite, compute RAGAS-style metrics, persist.

    Suite YAML format:
      name: my-suite
      queries:
        - id: q1
          question: "What is X?"
          tags: [definition]
    """
    from pathlib import Path as _P
    from src.eval import EvalSuite, run_suite
    from src.eval.runner import summarize_results

    p = _P(suite_path)
    if not p.exists():
        console.print(f"[red]Suite not found: {p}[/red]")
        raise typer.Exit(code=1)

    suite = EvalSuite.from_yaml(p)
    console.print(f"[bold cyan]Running suite:[/bold cyan] {suite.name}  "
                  f"({len(suite.queries)} queries)\n")

    def _on_complete(i: int, total: int, r) -> None:
        sc = r.scores
        line = (f"[{i}/{total}] [cyan]{r.query_id}[/cyan]  "
                f"faith={sc['faithfulness']:.2f}  rel={sc['answer_relevancy']:.2f}  "
                f"prec={sc['context_precision']:.2f}  "
                f"overall=[bold]{sc['overall']:.2f}[/bold]  "
                f"({r.latency_ms}ms)")
        console.print(line)
        if show_answers:
            console.print(Panel(r.answer.strip(), border_style="dim"))

    results = run_suite(suite, on_query_complete=_on_complete, skip_metrics=skip_metrics)
    summary = summarize_results(results)

    # ---- Summary table ----
    t = Table(title=f"Suite '{suite.name}' summary", header_style="bold cyan")
    for col in ("Metric", "Mean", "Min", "Max"):
        t.add_column(col)
    for metric in ("faithfulness", "answer_relevancy", "context_precision", "overall"):
        s = summary[metric]
        t.add_row(metric, f"{s['mean']:.3f}", f"{s['min']:.3f}", f"{s['max']:.3f}")
    console.print()
    console.print(t)
    console.print(
        f"\n[bold]Mean latency:[/bold] {summary['mean_latency_ms']} ms    "
        f"[bold]Reflection-triggered:[/bold] {summary['n_reflections_triggered']}/{summary['n']}    "
        f"[bold]Web-fallback used:[/bold] {summary['n_web_used']}/{summary['n']}"
    )
    console.print(f"\nResults appended to [cyan]data/eval/runs.jsonl[/cyan]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address (use 0.0.0.0 for LAN)"),
    port: int = typer.Option(8000, help="HTTP port"),
    reload: bool = typer.Option(False, help="Auto-reload on code change (dev mode)"),
) -> None:
    """
    Phase 6b/6c — Launch FastAPI + web UI. Streams agent runs live via SSE.

    Open http://localhost:8000 in your browser. The UI shows live per-node
    progress, the final answer with clickable citations, and a sources panel.
    """
    import uvicorn

    console.print(
        f"[bold green]ResearGent UI:[/bold green] http://{host}:{port}\n"
        f"[dim]API docs:[/dim] http://{host}:{port}/docs"
    )
    uvicorn.run(
        "src.api.app:create_app",
        host=host, port=port, reload=reload, factory=True,
        log_level="info",
    )


@app.command()
def store(
    action: str = typer.Argument("info", help="info | reset"),
) -> None:
    """Inspect or reset the vector store. `reset` drops the current collection."""
    from src.store import list_collections, reset_papers_collection

    if action == "info":
        cols = list_collections()
        if not cols:
            console.print("[yellow]No collections yet. Run `ingest` to create one.[/yellow]")
            return
        t = Table(title="Chroma Collections", header_style="bold cyan")
        t.add_column("Name", overflow="fold")
        t.add_column("Chunks")
        for c in cols:
            t.add_row(c["name"], str(c["count"]))
        console.print(t)
    elif action == "reset":
        name = reset_papers_collection()
        console.print(f"[yellow]Dropped collection:[/yellow] {name}")
    else:
        console.print(f"[red]Unknown action: {action}[/red] (use: info | reset)")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
