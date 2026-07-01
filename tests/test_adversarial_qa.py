from __future__ import annotations

from pathlib import Path

from agentic_language_translation_tool.adversarial_qa import (
    content_signal_tokens,
    detect_adversarial_qa_issues,
    extract_signal_set,
    protected_signal_tokens,
)
from agentic_language_translation_tool.io import write_jsonl
from agentic_language_translation_tool.models import Segment, TranslationRecord
from agentic_language_translation_tool.workflows import apply_translations, validate_job
from agentic_language_translation_tool.workspace import init_workspace, read_segments


def test_signal_extraction_is_language_agnostic() -> None:
    segment = _segment(
        'API_KEY v2.1 costs 15% less on 2026-06-30. Click "Export" at https://example.com.'
    )

    signals = extract_signal_set(segment.source_text, segment)

    assert "API_KEY" in signals.acronyms | signals.code_identifiers
    assert "v2.1" in signals.versions
    assert "15%" in signals.percentages
    assert "2026-06-30" in signals.dates
    assert '"Export"' in signals.quoted_labels
    assert "https://example.com." in signals.urls
    assert {"API_KEY", "v2.1", "15%", '"Export"'} <= protected_signal_tokens(
        segment.source_text,
        segment,
    )


def test_detects_missing_versions_numbers_and_quoted_labels() -> None:
    segment = _segment('Instalați Modul-X v2.1 și apăsați "Export" înainte de 2026-06-30.')

    issues = detect_adversarial_qa_issues(
        segment,
        "Instale le module et appuyez sur le bouton.",
        source_language="ro",
        target_language="fr",
    )
    categories = {issue.category for issue in issues}

    assert "possible_entity_damage" in categories
    assert "possible_omission" in categories


def test_detects_added_target_only_signals_as_possible_hallucination() -> None:
    segment = _segment("Sincronizați setările locale.")

    issues = detect_adversarial_qa_issues(
        segment,
        'Sincronizați setările locale cu API_X v9.4 și apăsați "Launch" pe 2027-01-01.',
        source_language="ro",
        target_language="ro",
    )

    assert "possible_hallucination" in {issue.category for issue in issues}


def test_cjk_content_tokens_do_not_require_spaces() -> None:
    tokens = content_signal_tokens("設定を保存してから同期します")

    assert {"設", "定", "保", "存"}.intersection(tokens)


def test_cyrillic_and_arabic_do_not_depend_on_latin_capitalization() -> None:
    cyrillic = _segment("Версия v3.0 синхронизирует модуль МОДУЛЬ7.")
    arabic = _segment("يحفظ الإصدار v3.0 الإعدادات بنسبة 15%.")

    cyrillic_issues = detect_adversarial_qa_issues(
        cyrillic,
        "Версия синхронизирует модуль.",
        source_language="ru",
        target_language="ru",
    )
    arabic_issues = detect_adversarial_qa_issues(
        arabic,
        "يحفظ الإصدار الإعدادات.",
        source_language="ar",
        target_language="ar",
    )

    assert "possible_entity_damage" in {issue.category for issue in cyrillic_issues}
    assert "possible_entity_damage" in {issue.category for issue in arabic_issues}


def test_overly_literal_warns_only_across_different_languages() -> None:
    segment = _segment("customer support dashboard deployment")

    same_language = detect_adversarial_qa_issues(
        segment,
        "customer support dashboard deployment",
        source_language="en",
        target_language="en",
    )
    different_language = detect_adversarial_qa_issues(
        segment,
        "customer support dashboard deployment listo",
        source_language="en",
        target_language="es",
    )

    assert "possible_overly_literal" not in {issue.category for issue in same_language}
    assert "possible_overly_literal" in {issue.category for issue in different_language}


def test_tone_style_drift_uses_language_neutral_punctuation() -> None:
    segment = _segment("Actualizarea este disponibilă azi.")

    issues = detect_adversarial_qa_issues(
        segment,
        "Actualizarea este disponibilă azi!!!!",
        source_language="ro",
        target_language="ro",
    )

    assert "possible_tone_or_style_drift" in {issue.category for issue in issues}


def test_no_english_promotional_terms_are_default_hallucination_trigger() -> None:
    segment = _segment("Actualizarea este disponibilă azi.")

    issues = detect_adversarial_qa_issues(
        segment,
        "Actualizarea este disponibilă azi și este amazing.",
        source_language="ro",
        target_language="ro",
    )

    assert "possible_hallucination" not in {issue.category for issue in issues}


def test_adversarial_warnings_do_not_fail_validate_job(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    workspace = tmp_path / "workspace"
    source.write_text("Sincronizați setările locale.", encoding="utf-8")
    init_workspace(source, workspace, source_language="ro", target_language="fr")
    segment = read_segments(workspace)[0]
    translations = tmp_path / "translations.jsonl"
    write_jsonl(
        translations,
        [
            TranslationRecord(
                segment_id=segment.segment_id,
                translated_text=(
                    'Synchronisez les paramètres locaux avec API_X v9.4 et "Launch".'
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
