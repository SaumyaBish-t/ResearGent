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


if __name__ == "__main__":
    app()
