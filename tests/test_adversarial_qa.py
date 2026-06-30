from __future__ import annotations

from pathlib import Path

from agentic_language_translation_tool.adversarial_qa import (
    detect_adversarial_qa_issues,
    important_source_tokens,
    source_entities,
    suspicious_addition_terms,
)
from agentic_language_translation_tool.io import write_jsonl
from agentic_language_translation_tool.models import Segment, TranslationRecord
from agentic_language_translation_tool.workflows import apply_translations, validate_job
from agentic_language_translation_tool.workspace import init_workspace, read_segments


def test_adversarial_qa_detects_hallucinated_additions() -> None:
    segment = _segment("The app stores settings locally.")

    issues = detect_adversarial_qa_issues(
        segment,
        "The app stores settings locally and is officially recommended and guaranteed.",
    )

    assert "officially recommended" in suspicious_addition_terms(
        "This is officially recommended."
    )
    assert "possible_hallucination" in {issue.category for issue in issues}
    assert all(issue.severity.value == "warning" for issue in issues)


def test_adversarial_qa_detects_omissions_and_entity_damage() -> None:
    segment = _segment("CloudKit syncs Project Atlas records across devices for field teams.")

    issues = detect_adversarial_qa_issues(segment, "Synchronisiert.")
    categories = {issue.category for issue in issues}

    assert source_entities(segment.source_text) >= {"CloudKit", "Project Atlas"}
    assert "possible_omission" in categories
    assert "possible_entity_damage" in categories


def test_adversarial_qa_detects_overly_literal_translation() -> None:
    segment = _segment("Please review the customer support dashboard before deployment starts.")

    issues = detect_adversarial_qa_issues(
        segment,
        "Bitte review the customer support dashboard before deployment starts.",
    )

    assert "customer" in important_source_tokens(segment.source_text)
    assert "possible_overly_literal" in {issue.category for issue in issues}


def test_adversarial_qa_detects_tone_or_style_drift() -> None:
    segment = _segment("The update is available today.")

    issues = detect_adversarial_qa_issues(segment, "Buy the amazing update today!!!")

    assert "possible_tone_or_style_drift" in {issue.category for issue in issues}


def test_adversarial_qa_does_not_flag_placeholders_urls_or_protected_terms() -> None:
    segment = Segment(
        segment_id="seg_test",
        source_text="Use {name} with API_KEY at https://example.com.",
        translated_text=None,
        format="txt",
        placeholders=["{name}"],
        protected_terms=["API_KEY"],
        checksum="checksum",
    )

    issues = detect_adversarial_qa_issues(
        segment,
        "Nutze {name} mit API_KEY bei https://example.com.",
    )

    assert issues == []


def test_adversarial_warnings_do_not_fail_validate_job(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    workspace = tmp_path / "workspace"
    source.write_text("The app stores settings locally.", encoding="utf-8")
    init_workspace(source, workspace, source_language="en", target_language="de")
    segment = read_segments(workspace)[0]
    translations = tmp_path / "translations.jsonl"
    write_jsonl(
        translations,
        [
            TranslationRecord(
                segment_id=segment.segment_id,
                translated_text=(
                    "The app stores settings locally and is officially recommended."
                ),
            )
        ],
    )
    apply_translations(workspace, translations)

    report = validate_job(workspace)

    assert report.passed
    assert "possible_hallucination" in {issue.category for issue in report.issues}
    assert (workspace / "qa_report.md").read_text(encoding="utf-8").count(
        "possible_hallucination"
    ) == 1


def _segment(source_text: str) -> Segment:
    return Segment(
        segment_id="seg_test",
        source_text=source_text,
        translated_text=None,
        format="txt",
        checksum="checksum",
    )
