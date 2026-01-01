"""Synkro CLI - Generate training data from the command line."""

import typer
from pathlib import Path
from typing import Optional

app = typer.Typer(
    name="synkro",
    help="Generate training datasets from documents.",
    no_args_is_help=True,
)


@app.command()
def generate(
    source: str = typer.Argument(
        ...,
        help="Policy text, file path (.pdf, .docx, .txt, .md), or URL",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path (auto-generated if not specified)",
    ),
    traces: int = typer.Option(
        20,
        "--traces", "-n",
        help="Number of traces to generate",
    ),
    format: str = typer.Option(
        "sft",
        "--format", "-f",
        help="Output format: sft or qa",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model", "-m",
        help="Model for generation (e.g., gpt-4o-mini, claude-3-5-sonnet, gemini-2.5-flash)",
    ),
):
    """
    Generate training data from a policy document.

    Examples:

        synkro generate policy.pdf

        synkro generate "All expenses over $50 need approval" --traces 50

        synkro generate handbook.docx -o training.jsonl -n 100
    """
    import synkro
    from synkro import Policy

    # Determine if source is text, file, or URL
    source_path = Path(source)
    
    if source_path.exists():
        # It's a file
        policy = Policy.from_file(source_path)
    elif source.startswith(("http://", "https://")):
        # It's a URL
        policy = Policy.from_url(source)
    else:
        # Treat as raw text
        policy = Policy(text=source)

    # Generate
    dataset = synkro.generate(
        policy,
        traces=traces,
        generation_model=model,
    )

    # Save
    if output:
        dataset.save(output, format=format)
    else:
        dataset.save(format=format)


@app.command()
def demo():
    """
    Run a quick demo with a built-in example policy.
    """
    import synkro
    from synkro.examples import EXPENSE_POLICY
    from rich.console import Console
    
    console = Console()
    console.print("\n[cyan]Running demo with built-in expense policy...[/cyan]\n")
    
    dataset = synkro.generate(EXPENSE_POLICY, traces=5)
    dataset.save("demo_output.jsonl")
    
    console.print("\n[green]Demo complete![/green]")
    console.print("[dim]Check demo_output.jsonl for the generated training data.[/dim]\n")


@app.command()
def version():
    """Show version information."""
    import synkro
    from rich.console import Console
    
    console = Console()
    console.print(f"[cyan]synkro[/cyan] v{synkro.__version__}")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()

