"""Glossary parsing, matching, and segment enrichment."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from rapidfuzz import fuzz

from agentic_language_translation_tool.io import file_sha256
from agentic_language_translation_tool.models import Glossary, GlossaryEntry, Segment

GLOSSARY_FILE = "glossary.json"
FUZZY_DRIFT_THRESHOLD = 88


class GlossaryError(ValueError):
    """Raised when a glossary file cannot be parsed or validated."""


def load_glossary(path: Path) -> Glossary:
    """Load a CSV or JSON glossary and return the normalized model."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        entries = _load_csv_entries(path)
    elif suffix == ".json":
        entries = _load_json_entries(path)
    else:
        raise GlossaryError(f"unsupported glossary format: {path.suffix}")
    _reject_duplicate_source_terms(entries)
    return Glossary(
        entries=entries,
        source_file=path.name,
        source_checksum=file_sha256(path),
    )


def read_workspace_glossary(workspace: Path) -> Glossary | None:
    """Read the normalized workspace glossary if one exists."""
    glossary_path = workspace / GLOSSARY_FILE
    if not glossary_path.exists():
        return None
    raw = json.loads(glossary_path.read_text(encoding="utf-8"))
    return Glossary.model_validate(raw)


def enrich_segments_with_glossary(segments: list[Segment], glossary: Glossary) -> list[Segment]:
    """Apply matching glossary entries to segment protected terms and notes."""
    for segment in segments:
        matched_entries = find_glossary_entries(segment.source_text, glossary)
        if not matched_entries:
            continue
        protected = list(segment.protected_terms)
        notes = list(segment.notes)
        for entry in matched_entries:
            if entry.source_term not in protected:
                protected.append(entry.source_term)
            note = glossary_note(entry)
            if note not in notes:
                notes.append(note)
        segment.protected_terms = sorted(protected)
        segment.notes = sorted(notes)
    return segments


def find_glossary_entries(text: str, glossary: Glossary) -> list[GlossaryEntry]:
    """Return glossary entries whose source term appears in text."""
    return [entry for entry in glossary.entries if term_in_text(entry.source_term, text, entry)]


def term_in_text(term: str, text: str, entry: GlossaryEntry) -> bool:
    """Return whether a glossary term appears as a bounded phrase."""
    flags = 0 if entry.case_sensitive else re.IGNORECASE
    pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", flags)
    return bool(pattern.search(text))


def glossary_note(entry: GlossaryEntry) -> str:
    """Render a compact glossary note for segment task files."""
    parts = [f"Glossary: {entry.source_term}"]
    if entry.do_not_translate:
        parts.append("do not translate")
    if entry.required_translation:
        parts.append(f"required: {entry.required_translation}")
    if entry.preferred_translation:
        parts.append(f"preferred: {entry.preferred_translation}")
    if entry.forbidden_translations:
        parts.append(f"forbidden: {'; '.join(entry.forbidden_translations)}")
    if entry.context:
        parts.append(f"context: {entry.context}")
    if entry.notes:
        parts.append(f"notes: {entry.notes}")
    return " | ".join(parts)


def likely_terminology_drift(expected: str, translation: str) -> bool:
    """Detect conservative near-miss terminology drift."""
    if not expected or _contains_text(translation, expected, case_sensitive=False):
        return False
    for phrase in _candidate_phrases(translation, expected):
        score = fuzz.ratio(expected.casefold(), phrase.casefold())
        if FUZZY_DRIFT_THRESHOLD <= score < 100:
            return True
    return False


def _load_csv_entries(path: Path) -> list[GlossaryEntry]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "source_term" not in reader.fieldnames:
            raise GlossaryError("CSV glossary must include a source_term header")
        return [_entry_from_mapping(row) for row in reader if _row_has_content(row.values())]


def _load_json_entries(path: Path) -> list[GlossaryEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict) and isinstance(raw.get("entries"), list):
        items = raw["entries"]
    else:
        raise GlossaryError("JSON glossary must be a list or an object with entries")
    return [_entry_from_mapping(cast(dict[str, object], item)) for item in items]


def _entry_from_mapping(raw: dict[str, object]) -> GlossaryEntry:
    data: dict[str, object] = dict(raw)
    if "case_sensitive" in data:
        data["case_sensitive"] = _parse_bool(data["case_sensitive"])
    if "do_not_translate" in data:
        data["do_not_translate"] = _parse_bool(data["do_not_translate"])
    try:
        return GlossaryEntry.model_validate(data)
    except ValueError as error:
        raise GlossaryError(str(error)) from error


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().casefold()
    if text in {"", "0", "false", "f", "no", "n"}:
        return False
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    raise GlossaryError(f"invalid boolean value: {value}")


def _reject_duplicate_source_terms(entries: list[GlossaryEntry]) -> None:
    seen: set[str] = set()
    for entry in entries:
        key = entry.source_term.casefold()
        if key in seen:
            raise GlossaryError(f"duplicate glossary source_term: {entry.source_term}")
        seen.add(key)


def _row_has_content(values: Iterable[object]) -> bool:
    return any(str(value or "").strip() for value in values)


def _contains_text(text: str, needle: str, *, case_sensitive: bool) -> bool:
    if case_sensitive:
        return needle in text
    return needle.casefold() in text.casefold()


def _candidate_phrases(text: str, expected: str) -> list[str]:
    words = re.findall(r"[\w-]+", text, flags=re.UNICODE)
    expected_words = max(1, len(re.findall(r"[\w-]+", expected, flags=re.UNICODE)))
    candidates: list[str] = []
    for width in {expected_words, expected_words + 1}:
        if width <= 0:
            continue
        for index in range(0, max(0, len(words) - width + 1)):
            candidates.append(" ".join(words[index : index + width]))
    return candidates
