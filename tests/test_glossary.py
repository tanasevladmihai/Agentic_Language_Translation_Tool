from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_language_translation_tool.glossary import (
    GlossaryError,
    enrich_segments_with_glossary,
    load_glossary,
)
from agentic_language_translation_tool.io import read_json_model, write_jsonl
from agentic_language_translation_tool.models import (
    FindingCategory,
    FindingSeverity,
    Glossary,
    Segment,
    TranslationRecord,
    VerificationFinding,
)
from agentic_language_translation_tool.workflows import (
    apply_glossary,
    apply_translations,
    apply_verification,
    plan_corrections,
    validate_job,
)
from agentic_language_translation_tool.workspace import init_workspace, read_segments


def test_load_glossary_csv_normalizes_booleans_and_semicolon_lists(tmp_path: Path) -> None:
    glossary_path = tmp_path / "glossary.csv"
    glossary_path.write_text(
        "source_term,required_translation,preferred_translation,forbidden_translations,"
        "case_sensitive,do_not_translate,context,notes\n"
        "Apple,,Apple,Apfel;Malus,false,true,technology company,brand rule\n",
        encoding="utf-8",
    )

    glossary = load_glossary(glossary_path)

    assert glossary.source_file == "glossary.csv"
    assert len(glossary.entries) == 1
    entry = glossary.entries[0]
    assert entry.source_term == "Apple"
    assert entry.forbidden_translations == ["Apfel", "Malus"]
    assert entry.do_not_translate is True
    assert entry.case_sensitive is False


def test_load_glossary_json_accepts_entries_object(tmp_path: Path) -> None:
    glossary_path = tmp_path / "glossary.json"
    glossary_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "source_term": "CloudKit",
                        "required_translation": "CloudKit",
                        "case_sensitive": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    glossary = load_glossary(glossary_path)

    assert glossary.entries[0].source_term == "CloudKit"
    assert glossary.entries[0].required_translation == "CloudKit"
    assert glossary.entries[0].case_sensitive is True


def test_load_glossary_rejects_missing_source_and_duplicates(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    missing.write_text("preferred_translation\nApple\n", encoding="utf-8")
    with pytest.raises(GlossaryError, match="source_term"):
        load_glossary(missing)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        json.dumps(
            [
                {"source_term": "Apple"},
                {"source_term": "apple"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(GlossaryError, match="duplicate"):
        load_glossary(duplicate)


def test_enrich_segments_with_glossary_is_idempotent(tmp_path: Path) -> None:
    glossary = _json_glossary(
        tmp_path,
        [
            {
                "source_term": "Apple",
                "do_not_translate": True,
                "context": "technology company",
            }
        ],
    )
    segments = [
        Segment(
            segment_id="seg_1",
            source_text="Apple released a device.",
            format="txt",
            checksum="abc",
        )
    ]

    enriched_once = enrich_segments_with_glossary(segments, glossary)
    enriched_twice = enrich_segments_with_glossary(enriched_once, glossary)

    assert enriched_twice[0].protected_terms == ["Apple"]
    assert len([note for note in enriched_twice[0].notes if note.startswith("Glossary:")]) == 1


def test_init_workspace_and_apply_glossary_update_files_and_prompts(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("Apple released CloudKit.", encoding="utf-8")
    glossary_path = tmp_path / "glossary.json"
    _write_glossary_entries(
        glossary_path,
        [
            {
                "source_term": "Apple",
                "do_not_translate": True,
                "context": "technology company",
            },
            {
                "source_term": "CloudKit",
                "required_translation": "CloudKit",
            },
        ],
    )
    workspace = tmp_path / "workspace"

    manifest = init_workspace(
        source,
        workspace,
        source_language="en",
        target_language="de",
        glossary=glossary_path,
    )
    apply_glossary(workspace, glossary_path)
    segments = read_segments(workspace)
    prompt = (workspace / "translation_batches" / "translation_0001.md").read_text(
        encoding="utf-8"
    )
    stored_glossary = read_json_model(workspace / "glossary.json", Glossary)

    assert manifest.glossary is not None
    assert stored_glossary.entries[0].source_term == "Apple"
    assert "Apple" in segments[0].protected_terms
    assert "CloudKit" in segments[0].protected_terms
    assert "Glossary Rules" in prompt
    assert "do not translate" in prompt
    assert "required: CloudKit" in prompt
    assert len([note for note in segments[0].notes if note.startswith("Glossary:")]) == 2


def test_apply_glossary_regenerates_verification_and_correction_prompts(tmp_path: Path) -> None:
    workspace = _workspace_with_source(tmp_path, "Apple released CloudKit.")
    glossary_path = tmp_path / "glossary.json"
    _write_glossary_entries(
        glossary_path,
        [{"source_term": "Apple", "do_not_translate": True}],
    )
    apply_glossary(workspace, glossary_path)
    segment = read_segments(workspace)[0]
    translations = tmp_path / "translations.jsonl"
    write_jsonl(
        translations,
        [TranslationRecord(segment_id=segment.segment_id, translated_text="Apfel veroffentlicht.")],
    )
    apply_translations(workspace, translations)
    findings = tmp_path / "findings.jsonl"
    write_jsonl(
        findings,
        [
            VerificationFinding(
                finding_id="finding_1",
                segment_id=segment.segment_id,
                category=FindingCategory.ENTITY_DAMAGE,
                severity=FindingSeverity.ERROR,
                explanation="Brand was translated literally.",
            )
        ],
    )
    apply_verification(workspace, findings)
    plan_corrections(workspace)

    verification_prompt = (workspace / "verification_batches" / "verification_0001.md").read_text(
        encoding="utf-8"
    )
    correction_prompt = next((workspace / "correction_batches").glob("*.md")).read_text(
        encoding="utf-8"
    )

    assert "Glossary Rules" in verification_prompt
    assert "Glossary Rules" in correction_prompt
    assert "do not translate" in correction_prompt


def test_validate_job_reports_glossary_terminology_issues(tmp_path: Path) -> None:
    workspace = _workspace_with_source(tmp_path, "Apple ships CloudKit for macOS.")
    glossary_path = tmp_path / "glossary.json"
    _write_glossary_entries(
        glossary_path,
        [
            {
                "source_term": "Apple",
                "do_not_translate": True,
                "forbidden_translations": ["Apfel"],
            },
            {
                "source_term": "CloudKit",
                "required_translation": "CloudKit",
            },
            {
                "source_term": "macOS",
                "preferred_translation": "macOS",
            },
        ],
    )
    apply_glossary(workspace, glossary_path)
    segment = read_segments(workspace)[0]
    translations = tmp_path / "translations.jsonl"
    write_jsonl(
        translations,
        [
            TranslationRecord(
                segment_id=segment.segment_id,
                translated_text="Apfel liefert KloudKit fur Mac OS.",
            )
        ],
    )
    apply_translations(workspace, translations)

    report = validate_job(workspace)
    categories = {issue.category for issue in report.issues}
    qa_markdown = (workspace / "qa_report.md").read_text(encoding="utf-8")

    assert not report.passed
    assert "do_not_translate_term_changed" in categories
    assert "required_translation_missing" in categories
    assert "preferred_translation_missing" in categories
    assert "forbidden_translation_used" in categories
    assert "terminology_drift" in categories
    assert "required_translation_missing" in qa_markdown


def _workspace_with_source(tmp_path: Path, text: str) -> Path:
    source = tmp_path / "source.txt"
    workspace = tmp_path / "workspace"
    source.write_text(text, encoding="utf-8")
    init_workspace(source, workspace, source_language="en", target_language="de")
    return workspace


def _json_glossary(tmp_path: Path, entries: list[dict[str, object]]) -> Glossary:
    path = tmp_path / "glossary.json"
    _write_glossary_entries(path, entries)
    return load_glossary(path)


def _write_glossary_entries(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
