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
    """Show which providers are configured and how tiers are routed."""
    info = list_status()

    # --- Providers table ---
    p_table = Table(title="Providers", show_header=True, header_style="bold cyan")
    p_table.add_column("Provider")
    p_table.add_column("Configured")
    p_table.add_column("Base URL", overflow="fold")
    p_table.add_column("Models (reasoning / fast / embed)", overflow="fold")

    for name, data in info["providers"].items():
        models = data["models"]
        p_table.add_row(
            name,
            "[green]yes[/green]" if data["configured"] else "[red]no[/red]",
            data["base_url"],
            f"{models['reasoning']}  /  {models['fast']}  /  {models['embed']}",
        )
    console.print(p_table)

    # --- Routing table ---
    r_table = Table(title="Tier Routing", show_header=True, header_style="bold cyan")
    r_table.add_column("Tier")
    r_table.add_column("Provider")
    r_table.add_column("Model")

    for tier, route in info["routing"].items():
        if "error" in route:
            r_table.add_row(tier, "[red]error[/red]", route["error"])
        else:
            r_table.add_row(tier, route["provider"], route["model"])
    console.print(r_table)


@app.command()
def smoke() -> None:
    """
    Ping each tier with a one-line prompt. Confirms wiring + credentials end-to-end.
    """
    prompt = [
        {"role": "system", "content": "Reply in EXACTLY one short sentence."},
        {"role": "user", "content": "Say hello and name yourself."},
    ]
    for tier in (ModelTier.REASONING, ModelTier.FAST):
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


# ---------------------------------------------------------------------------
# Phase 1 commands: ingest / rag-ask / retrieve / store
# ---------------------------------------------------------------------------


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
) -> None:
    """
    Phase 1 naive RAG — retrieve top-k chunks and answer with citations.

    This is the baseline we'll beat in every later phase.
    """
    from src.rag import naive_rag

    result = naive_rag(question, k=k)
    console.print(Panel(result.formatted(), title=f"answer (k={k})", border_style="cyan"))


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Query to retrieve chunks for"),
    k: int = typer.Option(5, help="Number of chunks to retrieve"),
) -> None:
    """Show raw retrieved chunks (debugging — no LLM call)."""
    from src.retrieval import naive_retrieve

    chunks = naive_retrieve(query, k=k)
    if not chunks:
        console.print("[yellow]No chunks retrieved (corpus empty?). Run `ingest` first.[/yellow]")
        return

    for i, c in enumerate(chunks, start=1):
        preview = c.text.strip().replace("\n", " ")
        preview = preview[:300] + ("..." if len(preview) > 300 else "")
        title = f"[S{i}] {c.citation}   score={c.score:.3f}"
        console.print(Panel(preview, title=title, border_style="cyan"))


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
