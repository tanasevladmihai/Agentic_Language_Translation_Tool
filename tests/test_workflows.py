from __future__ import annotations

from pathlib import Path

import pytest

from agentic_language_translation_tool.io import read_jsonl_model, write_jsonl
from agentic_language_translation_tool.models import (
    BatchPurpose,
    BatchStatus,
    FindingCategory,
    FindingSeverity,
    JobStage,
    TranslationRecord,
    VerificationFinding,
)
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
    read_segments,
    resume_summary,
)


def test_apply_translations_updates_segments_and_batch_status(tmp_path: Path) -> None:
    workspace = _txt_workspace(tmp_path)
    segments = read_segments(workspace)
    translations = tmp_path / "translations.jsonl"
    write_jsonl(
        translations,
        [
            TranslationRecord(
                segment_id=segment.segment_id,
                translated_text=f"DE: {segment.source_text}",
            )
            for segment in segments
        ],
    )

    manifest = apply_translations(workspace, translations)
    translated_segments = read_segments(workspace)

    assert all(segment.translated_text for segment in translated_segments)
    assert manifest.state.stage == JobStage.VERIFICATION_PLANNED
    assert all(
        batch.status == BatchStatus.TRANSLATED
        for batch in manifest.batches
        if batch.purpose == BatchPurpose.TRANSLATION
    )
    assert (workspace / "translations" / "translations.jsonl").exists()


def test_apply_translations_rejects_unknown_and_overwrite(tmp_path: Path) -> None:
    workspace = _txt_workspace(tmp_path)
    segment = read_segments(workspace)[0]
    translations = tmp_path / "translations.jsonl"
    write_jsonl(
        translations,
        [TranslationRecord(segment_id=segment.segment_id, translated_text="Bonjour")],
    )
    apply_translations(workspace, translations)

    with pytest.raises(WorkspaceError, match="already has translation"):
        apply_translations(workspace, translations)

    unknown = tmp_path / "unknown.jsonl"
    write_jsonl(unknown, [TranslationRecord(segment_id="seg_missing", translated_text="Bonjour")])
    with pytest.raises(WorkspaceError, match="unknown segment_id"):
        apply_translations(workspace, unknown)


def test_apply_verification_and_plan_corrections(tmp_path: Path) -> None:
    workspace = _translated_txt_workspace(tmp_path)
    segment = read_segments(workspace)[0]
    findings = tmp_path / "findings.jsonl"
    write_jsonl(
        findings,
        [
            VerificationFinding(
                finding_id="finding_1",
                segment_id=segment.segment_id,
                category=FindingCategory.OVERLY_LITERAL,
                severity=FindingSeverity.ERROR,
                explanation="The translation is too literal.",
                evidence="Awkward phrasing.",
                correction_guidance="Rewrite naturally.",
            )
        ],
    )

    manifest = apply_verification(workspace, findings)
    corrected_manifest = plan_corrections(workspace)

    assert manifest.state.stage == JobStage.CORRECTIONS_PLANNED
    assert any(batch.status == BatchStatus.NEEDS_CORRECTION for batch in manifest.batches)
    assert any(batch.purpose == BatchPurpose.CORRECTION for batch in corrected_manifest.batches)
    correction_files = list((workspace / "correction_batches").glob("*.md"))
    assert len(correction_files) == 1
    correction_prompt = correction_files[0].read_text(encoding="utf-8")
    assert "The translation is too literal." in correction_prompt
    assert "Rewrite naturally." in correction_prompt


def test_validate_job_detects_deterministic_qa_issues(tmp_path: Path) -> None:
    workspace = _txt_workspace(tmp_path)
    segments = read_segments(workspace)
    translations = tmp_path / "translations.jsonl"
    write_jsonl(
        translations,
        [
            TranslationRecord(segment_id=segments[0].segment_id, translated_text="Bonjour"),
            TranslationRecord(segment_id=segments[2].segment_id, translated_text="Dokumentation."),
        ],
    )
    apply_translations(workspace, translations)

    report = validate_job(workspace)

    assert not report.passed
    assert {issue.category for issue in report.issues} >= {
        "missing_translation",
        "placeholder_drift",
        "protected_term_damage",
    }
    assert (workspace / "qa_report.json").exists()
    assert (workspace / "qa_report.md").exists()


def test_validate_job_detects_markdown_link_damage_and_blockers(tmp_path: Path) -> None:
    workspace = _markdown_workspace(tmp_path)
    segments = read_segments(workspace)
    linked_segment = next(
        segment for segment in segments if "https://apple.com" in segment.source_text
    )
    translations = tmp_path / "translations.jsonl"
    write_jsonl(
        translations,
        [
            TranslationRecord(segment_id=segment.segment_id, translated_text="Titel")
            if segment.segment_id != linked_segment.segment_id
            else TranslationRecord(
                segment_id=segment.segment_id,
                translated_text="Apple Link entfernt",
            )
            for segment in segments
        ],
    )
    apply_translations(workspace, translations)
    findings = tmp_path / "blockers.jsonl"
    write_jsonl(
        findings,
        [
            VerificationFinding(
                finding_id="finding_blocker",
                segment_id=linked_segment.segment_id,
                category=FindingCategory.HALLUCINATION,
                severity=FindingSeverity.BLOCKER,
                explanation="Added unsupported claim.",
            )
        ],
    )
    apply_verification(workspace, findings)

    report = validate_job(workspace)

    assert not report.passed
    assert "markdown_link_damage" in {issue.category for issue in report.issues}
    assert "verification_hallucination" in {issue.category for issue in report.issues}


def test_rebuild_txt_and_markdown_outputs(tmp_path: Path) -> None:
    txt_workspace = _translated_txt_workspace(tmp_path / "txt")
    txt_output = tmp_path / "translated.txt"

    rebuild_document(txt_workspace, txt_output)

    assert (
        txt_output.read_text(encoding="utf-8")
        == "Hallo {name}.\n\nMittlerer Absatz.\n\nDokumentation: https://example.com.\n"
    )

    md_workspace = _markdown_workspace(tmp_path / "md")
    md_segments = read_segments(md_workspace)
    md_translations = tmp_path / "md_translations.jsonl"
    write_jsonl(
        md_translations,
        [
            TranslationRecord(segment_id=segment.segment_id, translated_text=f"DE {index}")
            for index, segment in enumerate(md_segments, start=1)
        ],
    )
    apply_translations(md_workspace, md_translations)
    md_output = tmp_path / "translated.md"

    rebuild_document(md_workspace, md_output)
    rebuilt = md_output.read_text(encoding="utf-8")

    assert "DE 1" in rebuilt
    assert "DE 2" in rebuilt
    assert "```python\nprint('x')\n```" in rebuilt


def test_full_mvp_flow_and_resume(tmp_path: Path) -> None:
    workspace = _translated_txt_workspace(tmp_path)
    findings = tmp_path / "empty_findings.jsonl"
    findings.write_text("", encoding="utf-8")

    apply_verification(workspace, findings)
    report = validate_job(workspace)
    output = tmp_path / "final.txt"
    rebuild_document(workspace, output)
    summary = resume_summary(workspace)

    assert report.passed
    assert output.exists()
    assert "Current stage: `rebuilt`" in summary


def test_rebuild_refuses_missing_translation_or_blocker(tmp_path: Path) -> None:
    workspace = _txt_workspace(tmp_path)
    with pytest.raises(WorkspaceError, match="missing translations"):
        rebuild_document(workspace, tmp_path / "out.txt")

    translated = _translated_txt_workspace(tmp_path / "blocked")
    segment = read_segments(translated)[0]
    findings = tmp_path / "blocker.jsonl"
    write_jsonl(
        findings,
        [
            VerificationFinding(
                finding_id="finding_blocker",
                segment_id=segment.segment_id,
                category=FindingCategory.HALLUCINATION,
                severity=FindingSeverity.BLOCKER,
                explanation="Unsupported extra meaning.",
            )
        ],
    )
    apply_verification(translated, findings)
    with pytest.raises(WorkspaceError, match="unresolved blocker"):
        rebuild_document(translated, tmp_path / "blocked.txt")


def test_verification_results_are_normalized(tmp_path: Path) -> None:
    workspace = _translated_txt_workspace(tmp_path)
    segment = read_segments(workspace)[0]
    findings = tmp_path / "findings.jsonl"
    finding = VerificationFinding(
        finding_id="finding_info",
        segment_id=segment.segment_id,
        category=FindingCategory.TONE_OR_STYLE_DRIFT,
        severity=FindingSeverity.INFO,
        explanation="Looks fine.",
    )
    write_jsonl(findings, [finding])

    apply_verification(workspace, findings)

    stored = read_jsonl_model(
        workspace / "verification_results" / "findings.jsonl",
        VerificationFinding,
    )
    assert stored == [finding]


def _txt_workspace(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.txt"
    workspace = tmp_path / "workspace"
    source.write_text(
        "Hello {name}.\n\nMiddle paragraph.\n\nRead docs at https://example.com.",
        encoding="utf-8",
    )
    init_workspace(source, workspace, source_language="en", target_language="de")
    return workspace


def _translated_txt_workspace(tmp_path: Path) -> Path:
    workspace = _txt_workspace(tmp_path)
    segments = read_segments(workspace)
    translations = tmp_path / "translations.jsonl"
    translated_text = {
        segments[0].segment_id: "Hallo {name}.",
        segments[1].segment_id: "Mittlerer Absatz.",
        segments[2].segment_id: "Dokumentation: https://example.com.",
    }
    write_jsonl(
        translations,
        [
            TranslationRecord(
                segment_id=segment.segment_id,
                translated_text=translated_text[segment.segment_id],
            )
            for segment in segments
        ],
    )
    apply_translations(workspace, translations)
    return workspace


def _markdown_workspace(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.md"
    workspace = tmp_path / "workspace"
    source.write_text(
        "# Title\n\n"
        "Visit [Apple](https://apple.com).\n\n"
        "```python\nprint('x')\n```\n",
        encoding="utf-8",
    )
    init_workspace(source, workspace, source_language="en", target_language="de")
    return workspace
