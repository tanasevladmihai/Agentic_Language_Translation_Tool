"""Workspace creation, validation, and resume protocol."""

from __future__ import annotations

import shutil
from pathlib import Path

from agentic_language_translation_tool import __version__
from agentic_language_translation_tool.extractors import extract_document
from agentic_language_translation_tool.glossary import (
    GLOSSARY_FILE,
    enrich_segments_with_glossary,
    load_glossary,
)
from agentic_language_translation_tool.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    read_json_model,
    read_jsonl_model,
    stable_id,
    write_jsonl,
)
from agentic_language_translation_tool.models import (
    Batch,
    BatchPurpose,
    BatchStatus,
    GlossaryMetadata,
    JobManifest,
    JobStage,
    JobState,
    Segment,
)
from agentic_language_translation_tool.tasks import create_batches

JOB_FILE = "job.json"
RESUME_FILE = "resume.md"
SEGMENTS_FILE = "segments.jsonl"
STRUCTURE_FILE = "structure.json"


class WorkspaceError(RuntimeError):
    """Raised when a workspace is invalid or unsafe to update."""


def init_workspace(
    source: Path,
    workspace: Path,
    *,
    source_language: str,
    target_language: str,
    glossary: Path | None = None,
    force: bool = False,
) -> JobManifest:
    """Create or reuse a full convenience translation workspace."""
    if (workspace / JOB_FILE).exists() and not force:
        manifest = load_manifest(workspace)
        write_resume(workspace, manifest)
        return manifest
    manifest = extract_to_workspace(
        source,
        workspace,
        source_language=source_language,
        target_language=target_language,
        glossary=glossary,
        force=force,
    )
    manifest = plan_translation_batches(workspace, force=True)
    manifest = plan_verification_batches(workspace, force=True)
    manifest.state = _state_from_batches(JobStage.TRANSLATION_PLANNED, manifest.batches, workspace)
    save_manifest(workspace, manifest)
    write_resume(workspace, manifest)
    return manifest


def extract_to_workspace(
    source: Path,
    workspace: Path,
    *,
    source_language: str,
    target_language: str,
    glossary: Path | None = None,
    force: bool = False,
) -> JobManifest:
    """Extract a source document into a resumable workspace without planning batches."""
    source = source.resolve()
    workspace = workspace.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if (workspace / JOB_FILE).exists() and not force:
        raise WorkspaceError(f"workspace already contains extracted work: {workspace}")
    if workspace.exists() and any(workspace.iterdir()) and not force:
        raise WorkspaceError(f"workspace already exists and is not empty: {workspace}")

    _ensure_workspace_dirs(workspace)

    source_copy = workspace / "source" / source.name
    if not source_copy.exists() or force:
        shutil.copy2(source, source_copy)

    extraction = extract_document(source)
    segments = extraction.segments
    glossary_metadata: GlossaryMetadata | None = None
    if glossary is not None:
        normalized_glossary = load_glossary(glossary)
        atomic_write_json(workspace / GLOSSARY_FILE, normalized_glossary)
        segments = enrich_segments_with_glossary(segments, normalized_glossary)
        glossary_metadata = GlossaryMetadata(
            source_file=normalized_glossary.source_file or glossary.name,
            source_checksum=normalized_glossary.source_checksum or file_sha256(glossary),
            applied_at=normalized_glossary.applied_at,
        )
    write_jsonl(workspace / SEGMENTS_FILE, segments)
    atomic_write_json(workspace / STRUCTURE_FILE, extraction.structure)

    manifest = JobManifest(
        job_id=f"job_{stable_id(str(source), file_sha256(source))}",
        source_path=str(source),
        workspace_path=str(workspace),
        source_language=source_language,
        target_language=target_language,
        tool_version=__version__,
        source_checksum=file_sha256(source),
        state=_state_from_batches(JobStage.EXTRACTED, [], workspace),
        batches=[],
        glossary=glossary_metadata,
    )
    save_manifest(workspace, manifest)
    write_resume(workspace, manifest)
    return manifest


def plan_translation_batches(
    workspace: Path,
    *,
    max_segments: int = 10,
    force: bool = False,
) -> JobManifest:
    """Plan translation task batches for an extracted workspace."""
    manifest = load_manifest(workspace)
    segments = read_segments(workspace)
    existing = [batch for batch in manifest.batches if batch.purpose == BatchPurpose.TRANSLATION]
    if existing and not force:
        write_resume(workspace, manifest)
        return manifest
    _ensure_no_untracked_batch_files(workspace, "translation_batches", "translation_", force=force)
    if force:
        _remove_purpose_batches(workspace, manifest, BatchPurpose.TRANSLATION)
    batches = create_batches(
        segments,
        purpose=BatchPurpose.TRANSLATION,
        output_dir=workspace / "translation_batches",
        max_segments=max_segments,
    )
    translated_ids = {segment.segment_id for segment in segments if segment.translated_text}
    for batch in batches:
        if set(batch.segment_ids) <= translated_ids:
            batch.status = BatchStatus.TRANSLATED
    manifest.batches.extend(batches)
    manifest.state = _state_from_batches(JobStage.TRANSLATION_PLANNED, manifest.batches, workspace)
    save_manifest(workspace, manifest)
    write_resume(workspace, manifest)
    return manifest


def plan_verification_batches(
    workspace: Path,
    *,
    max_segments: int = 10,
    force: bool = False,
) -> JobManifest:
    """Plan verification task batches for a workspace's current segments."""
    manifest = load_manifest(workspace)
    segments = read_segments(workspace)
    existing = [batch for batch in manifest.batches if batch.purpose == BatchPurpose.VERIFICATION]
    if existing and not force:
        write_resume(workspace, manifest)
        return manifest
    _ensure_no_untracked_batch_files(
        workspace,
        "verification_batches",
        "verification_",
        force=force,
    )
    if force:
        _remove_purpose_batches(workspace, manifest, BatchPurpose.VERIFICATION)
    batches = create_batches(
        segments,
        purpose=BatchPurpose.VERIFICATION,
        output_dir=workspace / "verification_batches",
        max_segments=max_segments,
    )
    manifest.batches.extend(batches)
    manifest.state = _state_from_batches(JobStage.VERIFICATION_PLANNED, manifest.batches, workspace)
    save_manifest(workspace, manifest)
    write_resume(workspace, manifest)
    return manifest


def load_manifest(workspace: Path) -> JobManifest:
    """Load a workspace manifest."""
    return read_json_model(workspace / JOB_FILE, JobManifest)


def save_manifest(workspace: Path, manifest: JobManifest) -> None:
    """Persist a manifest and refresh its timestamp."""
    manifest.touch()
    atomic_write_json(workspace / JOB_FILE, manifest)


def _ensure_workspace_dirs(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    for directory in [
        "source",
        "translation_batches",
        "translations",
        "verification_batches",
        "verification_results",
        "correction_batches",
        "output",
    ]:
        (workspace / directory).mkdir(parents=True, exist_ok=True)


def _ensure_no_untracked_batch_files(
    workspace: Path,
    directory: str,
    prefix: str,
    *,
    force: bool,
) -> None:
    if force:
        return
    existing_files = sorted((workspace / directory).glob(f"{prefix}*.md"))
    if existing_files:
        relative = ", ".join(str(path.relative_to(workspace)) for path in existing_files)
        raise WorkspaceError(f"batch files already exist; pass --force to replace: {relative}")


def _remove_purpose_batches(
    workspace: Path,
    manifest: JobManifest,
    purpose: BatchPurpose,
) -> None:
    remaining: list[Batch] = []
    for batch in manifest.batches:
        if batch.purpose == purpose:
            batch_file = workspace / batch.file
            if batch_file.exists():
                batch_file.unlink()
        else:
            remaining.append(batch)
    manifest.batches = remaining


def validate_workspace(workspace: Path) -> list[str]:
    """Return workspace consistency errors."""
    errors: list[str] = []
    for required in [JOB_FILE, RESUME_FILE, SEGMENTS_FILE, STRUCTURE_FILE]:
        if not (workspace / required).exists():
            errors.append(f"missing {required}")
    if errors:
        return errors
    manifest = load_manifest(workspace)
    batch_files = [workspace / batch.file for batch in manifest.batches]
    for file in batch_files:
        if not file.exists():
            errors.append(f"missing batch file {file.relative_to(workspace)}")
    segment_ids = {segment.segment_id for segment in read_segments(workspace)}
    for batch in manifest.batches:
        missing = [segment_id for segment_id in batch.segment_ids if segment_id not in segment_ids]
        if missing:
            errors.append(
                f"batch {batch.batch_id} references missing segments: {', '.join(missing)}"
            )
    if manifest.glossary is not None and not (workspace / GLOSSARY_FILE).exists():
        errors.append(f"missing {GLOSSARY_FILE}")
    return errors


def read_segments(workspace: Path) -> list[Segment]:
    """Read workspace segments."""
    return read_jsonl_model(workspace / SEGMENTS_FILE, Segment)


def resume_summary(workspace: Path) -> str:
    """Build and refresh a human-readable resume summary."""
    manifest = load_manifest(workspace)
    manifest.state = _state_from_batches(manifest.state.stage, manifest.batches, workspace)
    save_manifest(workspace, manifest)
    return write_resume(workspace, manifest)


def write_resume(workspace: Path, manifest: JobManifest) -> str:
    """Write resume.md and return its content."""
    state = manifest.state
    findings = [
        finding
        for finding in manifest.findings
        if finding.severity.value in {"error", "blocker", "warning"}
    ]
    lines = [
        "# Pick Up Where You Left Off",
        "",
        f"- Workspace: `{manifest.workspace_path}`",
        f"- Job ID: `{manifest.job_id}`",
        f"- Current stage: `{state.stage.value}`",
        f"- Source language: `{manifest.source_language}`",
        f"- Target language: `{manifest.target_language}`",
        f"- Completed batches: {_format_list(state.completed_batches)}",
        f"- Pending batches: {_format_list(state.pending_batches)}",
        f"- Failed batches: {_format_list(state.failed_batches)}",
        f"- Blocked batches: {_format_list(state.blocked_batches)}",
        "",
        "## Next Action",
        "",
        f"`{state.next_command}`" if state.next_command else "Inspect pending batch files.",
        "",
        "## Files To Inspect First",
        "",
        "- `job.json`",
        "- `resume.md`",
        "- `segments.jsonl`",
        "- `glossary.json` if present",
        "- `translation_batches/`",
        "- `verification_batches/`",
        "",
        "## Verification Findings Requiring Correction",
        "",
    ]
    if findings:
        for finding in findings:
            lines.append(
                f"- `{finding.segment_id}` {finding.severity.value}/{finding.category.value}: "
                f"{finding.explanation}"
            )
    else:
        lines.append("- None recorded yet.")
    content = "\n".join(lines).rstrip() + "\n"
    atomic_write_text(workspace / RESUME_FILE, content)
    return content


def _state_from_batches(stage: JobStage, batches: list[Batch], workspace: Path) -> JobState:
    completed_statuses = {BatchStatus.TRANSLATED, BatchStatus.VERIFIED, BatchStatus.ACCEPTED}
    completed = [batch.batch_id for batch in batches if batch.status in completed_statuses]
    pending = [batch.batch_id for batch in batches if batch.status == BatchStatus.PENDING]
    failed = [batch.batch_id for batch in batches if batch.status == BatchStatus.NEEDS_CORRECTION]
    blocked = [batch.batch_id for batch in batches if batch.status == BatchStatus.BLOCKED]
    next_command = _next_command(stage, pending, workspace)
    return JobState(
        stage=stage,
        completed_batches=completed,
        pending_batches=pending,
        failed_batches=failed,
        blocked_batches=blocked,
        next_command=next_command,
    )


def _next_command(stage: JobStage, pending: list[str], workspace: Path) -> str:
    if pending:
        return f"Open the next pending batch `{pending[0]}` in `{workspace}`."
    if stage == JobStage.EXTRACTED:
        return f"altt plan-batches {workspace}"
    if stage == JobStage.TRANSLATION_PLANNED:
        return f"altt validate-workspace {workspace}"
    if stage == JobStage.PARTIALLY_TRANSLATED:
        return f"altt plan-verification {workspace}"
    if stage == JobStage.VERIFICATION_PLANNED:
        return f"altt validate {workspace}"
    if stage == JobStage.CORRECTIONS_PLANNED:
        return f"altt plan-corrections {workspace}"
    if stage == JobStage.VALIDATED:
        return f"altt rebuild {workspace} --output <path>"
    if stage == JobStage.BLOCKED:
        return f"Inspect {workspace / RESUME_FILE} and resolve blocker findings."
    return f"altt resume {workspace}"


def _format_list(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "none"
