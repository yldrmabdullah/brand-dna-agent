"""CLI entrypoint.

`brand-dna run --config configs/brands/cos.yaml` is the production interface.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from brand_dna.core.config import get_settings, load_brand_config
from brand_dna.core.observability import configure_logging
from brand_dna.core.orchestrator import run_brand

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Brand DNA Agent — autonomous brand intelligence for fashion brands. "
        "Produces a Brand DNA dossier (PDF + JSON manifest) from a brand's "
        "web presence."
    ),
)
console = Console()


@app.command()
def run(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to brand YAML config (e.g. configs/brands/cos.yaml).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    log_format: str = typer.Option(
        None,
        "--log-format",
        help="Override log format: 'json' (default) or 'console'.",
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        help="Override log level: DEBUG | INFO | WARNING | ERROR.",
    ),
) -> None:
    """Run the agent end-to-end for a brand. Produces dossier PDF + JSON."""
    settings = get_settings()
    configure_logging(
        level=log_level or settings.log_level,
        fmt=log_format or settings.log_format,
    )

    try:
        brand_config = load_brand_config(config)
    except Exception as exc:
        console.print(f"[red]Failed to load config:[/red] {exc}")
        raise typer.Exit(code=2)

    console.print(
        Panel.fit(
            f"[bold]{brand_config.name}[/bold]\n"
            f"[dim]{brand_config.url}[/dim]\n"
            f"social: {brand_config.social or '(none)'}",
            title="Brand DNA · Run",
            border_style="cyan",
        )
    )

    if not settings.openrouter_api_key:
        console.print(
            "[yellow]Warning:[/yellow] OPENROUTER_API_KEY not set. "
            "LLM-driven sections will fail and the dossier will degrade. "
            "Set it in .env and re-run for a complete dossier."
        )

    try:
        dossier = asyncio.run(run_brand(brand_config))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130)
    except Exception as exc:
        console.print(f"[red]Run failed:[/red] {exc}")
        raise

    _print_summary(dossier)


@app.command()
def validate(
    config: Path = typer.Argument(
        ...,
        help="Brand YAML config to validate.",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
) -> None:
    """Validate a brand YAML against the schema. Useful before scheduled runs."""
    try:
        bc = load_brand_config(config)
    except Exception as exc:
        console.print(f"[red]Invalid:[/red] {exc}")
        raise typer.Exit(code=2)
    console.print(f"[green]OK[/green] · {bc.name} · {bc.url}")


@app.command()
def show(
    run_dir: Path = typer.Argument(
        ...,
        help="Path to a run output dir, e.g. outputs/cos/<run_id>/",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Pretty-print a completed run's summary (without re-rendering)."""
    manifest = run_dir / "brand_dna.json"
    if not manifest.exists():
        console.print(f"[red]No brand_dna.json in {run_dir}[/red]")
        raise typer.Exit(code=2)
    data = json.loads(manifest.read_text())
    console.print(Panel.fit(
        f"[bold]{data['brand_name']}[/bold]\n"
        f"{data['one_line_positioning']}",
        title="Brand DNA · Summary",
        border_style="cyan",
    ))
    tbl = Table(title="Telemetry", show_header=False)
    rm = data.get("run_metadata", {})
    for k in (
        "run_id",
        "total_duration_s",
        "images_after_filter",
        "pages_crawled",
        "llm_tokens_in",
        "llm_tokens_out",
        "estimated_cost_usd",
    ):
        tbl.add_row(k, str(rm.get(k)))
    console.print(tbl)


def _print_summary(dossier) -> None:
    console.rule("[bold green]Run complete")
    console.print(Panel.fit(
        f"[bold]{dossier.brand_name}[/bold]\n"
        f"[italic]{dossier.one_line_positioning}[/italic]",
        title="Brand DNA",
        border_style="green",
    ))

    palette_tbl = Table(title="Top Colors", show_header=True, header_style="bold")
    palette_tbl.add_column("Hex")
    palette_tbl.add_column("%", justify="right")
    palette_tbl.add_column("Pantone (~)")
    palette_tbl.add_column("Descriptor")
    for c in dossier.color_palette.entries[:6]:
        palette_tbl.add_row(
            c.hex,
            f"{c.percentage:.0f}%",
            c.nearest_pantone or "—",
            c.descriptor or "—",
        )
    console.print(palette_tbl)

    cluster_tbl = Table(title="Aesthetic Clusters", show_header=True)
    cluster_tbl.add_column("#", justify="right")
    cluster_tbl.add_column("Label")
    cluster_tbl.add_column("Size", justify="right")
    cluster_tbl.add_column("Description")
    for c in dossier.aesthetic_clusters:
        cluster_tbl.add_row(
            str(c.cluster_id + 1),
            c.label,
            str(c.size),
            (c.description or "")[:60] + ("…" if len(c.description or "") > 60 else ""),
        )
    console.print(cluster_tbl)

    rm = dossier.run_metadata
    telemetry = Table(title="Run Telemetry", show_header=False)
    telemetry.add_row("Duration", f"{rm.total_duration_s:.1f}s")
    telemetry.add_row("Pages crawled", str(rm.pages_crawled))
    telemetry.add_row("Images after filter", str(rm.images_after_filter))
    telemetry.add_row("LLM tokens (in/out)", f"{rm.llm_tokens_in} / {rm.llm_tokens_out}")
    telemetry.add_row("LLM cost (est.)", f"${rm.estimated_cost_usd:.4f}")
    console.print(telemetry)

    console.print(
        f"\n[green]Outputs:[/green] {dossier.run_metadata.run_id}\n"
        f"  • PDF      → outputs/{dossier.brand_name.lower().replace(' ', '-')}/{dossier.run_metadata.run_id}/brand_dna.pdf\n"
        f"  • Manifest → outputs/{dossier.brand_name.lower().replace(' ', '-')}/{dossier.run_metadata.run_id}/brand_dna.json"
    )


if __name__ == "__main__":
    app()
