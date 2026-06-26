"""Shared schemas for the workspace protocol."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class JobStage(StrEnum):
    """Top-level stage for a translation job."""

    INITIALIZED = "initialized"
    EXTRACTED = "extracted"
    TRANSLATION_PLANNED = "translation_planned"
    PARTIALLY_TRANSLATED = "partially_translated"
    VERIFICATION_PLANNED = "verification_planned"
    CORRECTIONS_PLANNED = "corrections_planned"
    VALIDATED = "validated"
    REBUILT = "rebuilt"
    BLOCKED = "blocked"


class BatchStatus(StrEnum):
    """Lifecycle state for an agent task batch."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    TRANSLATED = "translated"
    VERIFIED = "verified"
    NEEDS_CORRECTION = "needs_correction"
    ACCEPTED = "accepted"
    BLOCKED = "blocked"


class BatchPurpose(StrEnum):
    """Reason an agent task batch exists."""

    TRANSLATION = "translation"
    VERIFICATION = "verification"
    CORRECTION = "correction"


class FindingSeverity(StrEnum):
    """Impact of a verification finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKER = "blocker"


class FindingCategory(StrEnum):
    """Concept-aware verification finding categories."""

    HALLUCINATION = "hallucination"
    OMISSION = "omission"
    ENTITY_DAMAGE = "entity_damage"
    OVERLY_LITERAL = "overly_literal"
    TERMINOLOGY_DRIFT = "terminology_drift"
    TONE_OR_STYLE_DRIFT = "tone_or_style_drift"
    FORMATTING_DAMAGE = "formatting_damage"
    UNTRANSLATED_TEXT = "untranslated_text"


class WorkspaceModel(BaseModel):
    """Base model shared by protocol schemas."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Segment(WorkspaceModel):
    """A translatable unit extracted from a source document."""

    segment_id: str
    source_text: str
    translated_text: str | None = None
    format: str
    context: str = ""
    path: list[str] = Field(default_factory=list)
    style_tags: list[str] = Field(default_factory=list)
    placeholders: list[str] = Field(default_factory=list)
    protected_terms: list[str] = Field(default_factory=list)
    checksum: str
    notes: list[str] = Field(default_factory=list)

    @field_validator("segment_id", "source_text", "format", "checksum")
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        """Reject empty identifiers and content-critical fields."""
        if not value.strip():
            raise ValueError("value must not be empty")
        return value


class Batch(WorkspaceModel):
    """A batch of segment IDs prepared for an agent."""

    batch_id: str
    purpose: BatchPurpose
    segment_ids: list[str]
    token_estimate: int = Field(ge=0)
    status: BatchStatus = BatchStatus.PENDING
    file: str

    @field_validator("batch_id", "file")
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        """Reject empty batch identifiers and paths."""
        if not value.strip():
            raise ValueError("value must not be empty")
        return value


class VerificationFinding(WorkspaceModel):
    """A verifier agent or deterministic validator finding."""

    finding_id: str
    segment_id: str
    category: FindingCategory
    severity: FindingSeverity
    explanation: str
    evidence: str = ""
    correction_guidance: str = ""


class TranslationRecord(WorkspaceModel):
    """A translated segment row produced by an agent."""

    segment_id: str
    translated_text: str
    notes: list[str] = Field(default_factory=list)

    @field_validator("segment_id", "translated_text")
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        """Reject empty translation identifiers and text."""
        if not value.strip():
            raise ValueError("value must not be empty")
        return value


class QaIssue(WorkspaceModel):
    """A deterministic QA issue found in a workspace."""

    issue_id: str
    segment_id: str | None = None
    category: str
    severity: FindingSeverity
    explanation: str


class QaReport(WorkspaceModel):
    """Deterministic QA report for a workspace."""

    job_id: str
    workspace_path: str
    checked_at: datetime = Field(default_factory=utc_now)
    passed: bool
    issue_count: int
    issues: list[QaIssue] = Field(default_factory=list)


class JobState(WorkspaceModel):
    """Resumable state summary for a translation job."""

    stage: JobStage = JobStage.INITIALIZED
    completed_batches: list[str] = Field(default_factory=list)
    pending_batches: list[str] = Field(default_factory=list)
    failed_batches: list[str] = Field(default_factory=list)
    blocked_batches: list[str] = Field(default_factory=list)
    next_command: str = ""


class JobManifest(WorkspaceModel):
    """The top-level manifest stored as job.json."""

    job_id: str
    source_path: str
    workspace_path: str
    source_language: str
    target_language: str
    tool_version: str
    source_checksum: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    state: JobState = Field(default_factory=JobState)
    batches: list[Batch] = Field(default_factory=list)
    findings: list[VerificationFinding] = Field(default_factory=list)

    def touch(self) -> None:
        """Update the manifest timestamp."""
        self.updated_at = utc_now()
