"""Command-line helpers for agentic translation workspaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from agentic_language_translation_tool.extractors import inspect_input
from agentic_language_translation_tool.workspace import (
    WorkspaceError,
    init_workspace,
    resume_summary,
    validate_workspace,
)

app = typer.Typer(help="Workspace-first helper toolkit for agentic translation.")
console = Console()


@app.command("inspect")
def inspect_command(input_path: Annotated[Path, typer.Argument(exists=True)]) -> None:
    """Inspect a source document and report parser support."""
    console.print(json.dumps(inspect_input(input_path), ensure_ascii=False, indent=2))


@app.command("init-workspace")
def init_workspace_command(
    input_path: Annotated[Path, typer.Argument(exists=True)],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    source_language: Annotated[str, typer.Option("--source-language", "-s")],
    target_language: Annotated[str, typer.Option("--target-language", "-t")],
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Create a resumable translation workspace."""
    try:
        manifest = init_workspace(
            input_path,
            workspace,
            source_language=source_language,
            target_language=target_language,
            force=force,
        )
    except WorkspaceError as error:
        raise typer.BadParameter(str(error)) from error
    console.print(f"Workspace ready: {manifest.workspace_path}")
    console.print(f"Next: {manifest.state.next_command}")


@app.command("resume")
def resume_command(workspace: Annotated[Path, typer.Argument(exists=True)]) -> None:
    """Print and refresh the resume summary for a workspace."""
    errors = validate_workspace(workspace)
    if errors:
        for error in errors:
            console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1)
    console.print(resume_summary(workspace))


@app.command("validate-workspace")
def validate_workspace_command(workspace: Annotated[Path, typer.Argument(exists=True)]) -> None:
    """Validate workspace consistency."""
    errors = validate_workspace(workspace)
    if errors:
        for error in errors:
            console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1)
    console.print("[green]Workspace is valid.[/green]")


def main() -> None:
    """Run the CLI application."""
    app()


if __name__ == "__main__":
    main()
