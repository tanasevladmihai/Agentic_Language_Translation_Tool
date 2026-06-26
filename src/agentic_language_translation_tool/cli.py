"""Command-line helpers for agentic translation workspaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from agentic_language_translation_tool.extractors import inspect_input
from agentic_language_translation_tool.workflows import (
    apply_translations,
    apply_verification,
    plan_corrections,
    rebuild_document,
    validate_job,
)
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


@app.command("apply-translations")
def apply_translations_command(
    workspace: Annotated[Path, typer.Argument(exists=True)],
    translations: Annotated[Path, typer.Option("--translations", "-t", exists=True)],
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Apply agent-produced translation JSONL to a workspace."""
    try:
        manifest = apply_translations(workspace, translations, force=force)
    except WorkspaceError as error:
        raise typer.BadParameter(str(error)) from error
    console.print(f"Translations applied. Stage: {manifest.state.stage.value}")
    console.print(f"Next: {manifest.state.next_command}")


@app.command("apply-verification")
def apply_verification_command(
    workspace: Annotated[Path, typer.Argument(exists=True)],
    results: Annotated[Path, typer.Option("--results", "-r", exists=True)],
) -> None:
    """Apply verifier findings JSONL to a workspace."""
    try:
        manifest = apply_verification(workspace, results)
    except WorkspaceError as error:
        raise typer.BadParameter(str(error)) from error
    console.print(f"Verification applied. Stage: {manifest.state.stage.value}")
    console.print(f"Next: {manifest.state.next_command}")


@app.command("plan-corrections")
def plan_corrections_command(workspace: Annotated[Path, typer.Argument(exists=True)]) -> None:
    """Generate correction batches for unresolved verifier findings."""
    try:
        manifest = plan_corrections(workspace)
    except WorkspaceError as error:
        raise typer.BadParameter(str(error)) from error
    console.print(f"Correction planning complete. Stage: {manifest.state.stage.value}")
    console.print(f"Next: {manifest.state.next_command}")


@app.command("validate")
def validate_command(workspace: Annotated[Path, typer.Argument(exists=True)]) -> None:
    """Run deterministic QA checks and write QA reports."""
    try:
        report = validate_job(workspace)
    except WorkspaceError as error:
        raise typer.BadParameter(str(error)) from error
    status = "passed" if report.passed else "failed"
    console.print(f"QA {status} with {report.issue_count} issue(s).")


@app.command("rebuild")
def rebuild_command(
    workspace: Annotated[Path, typer.Argument(exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Rebuild a translated TXT or Markdown document."""
    try:
        rebuilt = rebuild_document(workspace, output, force=force)
    except WorkspaceError as error:
        raise typer.BadParameter(str(error)) from error
    console.print(f"Rebuilt document: {rebuilt}")


def main() -> None:
    """Run the CLI application."""
    app()


if __name__ == "__main__":
    main()
