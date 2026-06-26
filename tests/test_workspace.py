from pathlib import Path

import pytest

from agentic_language_translation_tool.models import JobStage
from agentic_language_translation_tool.workspace import (
    WorkspaceError,
    init_workspace,
    resume_summary,
    validate_workspace,
)


def test_init_workspace_creates_resumable_protocol_files(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    workspace = tmp_path / "workspace"
    source.write_text("Hello world.\n\nApple is a company.", encoding="utf-8")

    manifest = init_workspace(
        source,
        workspace,
        source_language="en",
        target_language="de",
    )

    assert manifest.state.stage == JobStage.TRANSLATION_PLANNED
    assert (workspace / "job.json").exists()
    assert (workspace / "resume.md").exists()
    assert (workspace / "segments.jsonl").exists()
    assert (workspace / "translation_batches" / "translation_0001.md").exists()
    assert (workspace / "verification_batches" / "verification_0001.md").exists()
    assert validate_workspace(workspace) == []


def test_init_workspace_is_idempotent_for_existing_job(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    workspace = tmp_path / "workspace"
    source.write_text("Hello world.", encoding="utf-8")

    first = init_workspace(source, workspace, source_language="en", target_language="fr")
    second = init_workspace(source, workspace, source_language="en", target_language="fr")

    assert second.job_id == first.job_id
    assert validate_workspace(workspace) == []


def test_init_workspace_refuses_non_empty_unmanaged_directory(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    workspace = tmp_path / "workspace"
    source.write_text("Hello world.", encoding="utf-8")
    workspace.mkdir()
    (workspace / "stray.txt").write_text("stray", encoding="utf-8")

    with pytest.raises(WorkspaceError):
        init_workspace(source, workspace, source_language="en", target_language="fr")


def test_resume_summary_lists_next_action(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    workspace = tmp_path / "workspace"
    source.write_text("Hello world.", encoding="utf-8")
    init_workspace(source, workspace, source_language="en", target_language="fr")

    summary = resume_summary(workspace)

    assert "Pick Up Where You Left Off" in summary
    assert "translation_0001" in summary
    assert "Files To Inspect First" in summary
