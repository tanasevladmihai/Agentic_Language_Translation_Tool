from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_language_translation_tool.cli import app
from agentic_language_translation_tool.io import write_jsonl
from agentic_language_translation_tool.models import (
    BatchPurpose,
    BatchStatus,
    JobStage,
    TranslationRecord,
)
from agentic_language_translation_tool.workflows import apply_translations
from agentic_language_translation_tool.workspace import (
    WorkspaceError,
    extract_to_workspace,
    init_workspace,
    plan_translation_batches,
    plan_verification_batches,
    read_segments,
    resume_summary,
)


def test_extract_to_workspace_creates_no_batch_files(tmp_path: Path) -> None:
    source = _source(tmp_path)
    workspace = tmp_path / "workspace"

    manifest = extract_to_workspace(
        source,
        workspace,
        source_language="en",
        target_language="de",
    )

    assert manifest.state.stage == JobStage.EXTRACTED
    assert manifest.state.next_command == f"altt plan-batches {workspace.resolve()}"
    assert (workspace / "job.json").exists()
    assert (workspace / "segments.jsonl").exists()
    assert (workspace / "structure.json").exists()
    assert not list((workspace / "translation_batches").glob("*.md"))
    assert not list((workspace / "verification_batches").glob("*.md"))


def test_plan_translation_batches_is_idempotent_and_force_refreshes(tmp_path: Path) -> None:
    workspace = _extracted_workspace(tmp_path)

    first = plan_translation_batches(workspace, max_segments=1)
    second = plan_translation_batches(workspace, max_segments=1)

    assert first.state.stage == JobStage.TRANSLATION_PLANNED
    second_translation_batches = [
        batch for batch in second.batches if batch.purpose == BatchPurpose.TRANSLATION
    ]
    assert len(second_translation_batches) == 2
    assert len(list((workspace / "translation_batches").glob("*.md"))) == 2

    forced = plan_translation_batches(workspace, max_segments=10, force=True)

    forced_translation_batches = [
        batch for batch in forced.batches if batch.purpose == BatchPurpose.TRANSLATION
    ]
    assert len(forced_translation_batches) == 1
    assert len(list((workspace / "translation_batches").glob("*.md"))) == 1


def test_plan_batches_refuses_untracked_batch_files_without_force(tmp_path: Path) -> None:
    workspace = _extracted_workspace(tmp_path)
    stray = workspace / "translation_batches" / "translation_0001.md"
    stray.write_text("manual file", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="batch files already exist"):
        plan_translation_batches(workspace)


def test_plan_verification_batches_uses_current_translations(tmp_path: Path) -> None:
    workspace = _extracted_workspace(tmp_path)
    plan_translation_batches(workspace, max_segments=1)
    segments = read_segments(workspace)
    translations = tmp_path / "translations.jsonl"
    write_jsonl(
        translations,
        [
            TranslationRecord(
                segment_id=segment.segment_id,
                translated_text=f"DE {segment.source_text}",
            )
            for segment in segments
        ],
    )
    manifest_after_translation = apply_translations(workspace, translations)

    manifest = plan_verification_batches(workspace, max_segments=10)
    prompt = (workspace / "verification_batches" / "verification_0001.md").read_text(
        encoding="utf-8"
    )

    assert manifest_after_translation.state.stage == JobStage.PARTIALLY_TRANSLATED
    assert manifest_after_translation.state.next_command == f"altt plan-verification {workspace}"
    assert manifest.state.stage == JobStage.VERIFICATION_PLANNED
    assert "DE First paragraph." in prompt
    assert "<TRANSLATION_HERE>" not in prompt
    assert any(
        batch.status == BatchStatus.PENDING
        for batch in manifest.batches
        if batch.purpose == BatchPurpose.VERIFICATION
    )


def test_init_workspace_still_runs_full_convenience_flow(tmp_path: Path) -> None:
    source = _source(tmp_path)
    workspace = tmp_path / "workspace"

    manifest = init_workspace(source, workspace, source_language="en", target_language="de")

    assert manifest.state.stage == JobStage.TRANSLATION_PLANNED
    assert list((workspace / "translation_batches").glob("*.md"))
    assert list((workspace / "verification_batches").glob("*.md"))


def test_resume_summary_for_explicit_stages(tmp_path: Path) -> None:
    workspace = _extracted_workspace(tmp_path)
    extracted_summary = resume_summary(workspace)
    assert "altt plan-batches" in extracted_summary

    plan_translation_batches(workspace)
    planned_summary = resume_summary(workspace)
    assert "translation_0001" in planned_summary

    segments = read_segments(workspace)
    translations = tmp_path / "translations.jsonl"
    write_jsonl(
        translations,
        [
            TranslationRecord(segment_id=segment.segment_id, translated_text=f"DE {index}")
            for index, segment in enumerate(segments, start=1)
        ],
    )
    apply_translations(workspace, translations)
    translated_summary = resume_summary(workspace)

    assert "altt plan-verification" in translated_summary


def test_cli_exposes_explicit_workflow_commands() -> None:
    runner = CliRunner()

    root = runner.invoke(app, ["--help"])
    extract = runner.invoke(app, ["extract", "--help"])
    plan_batches = runner.invoke(app, ["plan-batches", "--help"])
    plan_verification = runner.invoke(app, ["plan-verification", "--help"])

    assert root.exit_code == 0
    assert "extract" in root.output
    assert extract.exit_code == 0
    assert "--workspace" in extract.output
    assert plan_batches.exit_code == 0
    assert "--max-segments" in plan_batches.output
    assert plan_verification.exit_code == 0
    assert "--max-segments" in plan_verification.output


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source.txt"
    source.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")
    return source


def _extracted_workspace(tmp_path: Path) -> Path:
    source = _source(tmp_path)
    workspace = tmp_path / "workspace"
    extract_to_workspace(source, workspace, source_language="en", target_language="de")
    return workspace
