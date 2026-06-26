import pytest
from pydantic import ValidationError

from agentic_language_translation_tool.models import (
    BatchStatus,
    FindingCategory,
    JobManifest,
    JobStage,
    Segment,
)


def test_enum_values_are_protocol_strings() -> None:
    assert JobStage.INITIALIZED.value == "initialized"
    assert BatchStatus.NEEDS_CORRECTION.value == "needs_correction"
    assert FindingCategory.OVERLY_LITERAL.value == "overly_literal"


def test_segment_rejects_empty_required_fields() -> None:
    with pytest.raises(ValidationError):
        Segment(segment_id="", source_text="Hello", format="txt", checksum="abc")


def test_manifest_defaults_to_initialized_state() -> None:
    manifest = JobManifest(
        job_id="job_123",
        source_path="source.txt",
        workspace_path="workspace",
        source_language="en",
        target_language="de",
        tool_version="0.1.0",
        source_checksum="abc",
    )

    assert manifest.state.stage == JobStage.INITIALIZED
    assert manifest.batches == []
    assert manifest.findings == []
