"""Conservative deterministic checks for adversarial translation QA."""

from __future__ import annotations

import re

from agentic_language_translation_tool.io import stable_id
from agentic_language_translation_tool.models import FindingSeverity, QaIssue, Segment

ENTITY_PATTERN = re.compile(
    r"""
    (?:"[^"\n]{2,}"|'[^'\n]{2,}')
    |\b[A-Z]{2,}(?:[-_./]?[A-Z0-9]+)*\b
    |\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z0-9]+){0,3}(?:\s+\d+(?:\.\d+)*)?\b
    |\bv?\d+(?:\.\d+){1,3}\b
    """,
    re.VERBOSE,
)
WORD_PATTERN = re.compile(r"\b[\w'-]{3,}\b", re.UNICODE)
PLACEHOLDER_OR_URL_PATTERN = re.compile(r"`[^`]+`|\{[^{}]+\}|https?://\S+")

STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "allows",
    "also",
    "and",
    "because",
    "before",
    "between",
    "from",
    "into",
    "only",
    "that",
    "their",
    "these",
    "this",
    "through",
    "using",
    "when",
    "where",
    "while",
    "with",
    "without",
}
COMMON_SENTENCE_STARTS = {
    "A",
    "An",
    "And",
    "Before",
    "For",
    "If",
    "In",
    "It",
    "Keep",
    "See",
    "The",
    "This",
    "Use",
    "When",
}
SUSPICIOUS_ADDITIONS = {
    "certified",
    "guaranteed",
    "officially recommended",
    "official recommendation",
    "proven",
    "best",
    "must-have",
    "ensures",
    "therefore",
    "because of this",
    "as a result",
}
PROMOTIONAL_TONE_TERMS = {
    "amazing",
    "best",
    "excellent",
    "exclusive",
    "guaranteed",
    "incredible",
    "must-have",
    "perfect",
    "revolutionary",
}
IMPERATIVE_STARTERS = {"buy", "choose", "discover", "get", "install", "order", "try", "use"}


def detect_adversarial_qa_issues(segment: Segment, translation: str) -> list[QaIssue]:
    """Return conservative warning-level adversarial QA risks."""
    issues: list[QaIssue] = []
    source = segment.source_text
    if not translation.strip():
        return []

    additions = suspicious_addition_terms(translation) - suspicious_addition_terms(source)
    if additions:
        issues.append(
            _issue(
                segment.segment_id,
                "possible_hallucination",
                f"Translation contains unsupported risk terms: {', '.join(sorted(additions))}.",
            )
        )

    missing_entities = _missing_source_entities(segment, translation)
    if missing_entities:
        issues.append(
            _issue(
                segment.segment_id,
                "possible_entity_damage",
                "Source entities may be missing or changed: "
                f"{', '.join(sorted(missing_entities))}.",
            )
        )

    if _has_possible_omission(segment, translation, missing_entities):
        issues.append(
            _issue(
                segment.segment_id,
                "possible_omission",
                "Translation is much shorter and appears to omit important source content.",
            )
        )

    if _looks_overly_literal(segment, translation):
        issues.append(
            _issue(
                segment.segment_id,
                "possible_overly_literal",
                "Translation preserves unusually much source wording or order.",
            )
        )

    if _has_tone_or_style_drift(source, translation):
        issues.append(
            _issue(
                segment.segment_id,
                "possible_tone_or_style_drift",
                "Translation appears to shift tone, emphasis, or call-to-action style.",
            )
        )
    return issues


def source_entities(text: str) -> set[str]:
    """Extract entity-like terms that often should survive translation."""
    entities: set[str] = set()
    for match in ENTITY_PATTERN.findall(_remove_placeholders_and_urls(text)):
        value = match.strip("\"'")
        if not value or value in COMMON_SENTENCE_STARTS:
            continue
        if len(value) < 2:
            continue
        entities.add(value)
    return entities


def suspicious_addition_terms(text: str) -> set[str]:
    """Return suspicious promotional, causal, or certainty terms in text."""
    folded = text.casefold()
    return {term for term in SUSPICIOUS_ADDITIONS if term in folded}


def important_source_tokens(text: str) -> set[str]:
    """Extract coarse important tokens for omission/literalism heuristics."""
    return set(_important_tokens_in_order(text))


def _important_tokens_in_order(text: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in WORD_PATTERN.findall(_remove_placeholders_and_urls(text)):
        normalized = token.strip("'").casefold()
        if len(normalized) < 5 or normalized in STOPWORDS:
            continue
        if normalized in seen:
            continue
        tokens.append(normalized)
        seen.add(normalized)
    return tokens


def _missing_source_entities(segment: Segment, translation: str) -> set[str]:
    protected = {term.casefold() for term in segment.protected_terms}
    placeholders = {placeholder.casefold() for placeholder in segment.placeholders}
    missing: set[str] = set()
    folded_translation = translation.casefold()
    for entity in source_entities(segment.source_text):
        folded_entity = entity.casefold()
        if folded_entity in protected or folded_entity in placeholders:
            continue
        if folded_entity not in folded_translation:
            missing.add(entity)
    return missing


def _has_possible_omission(
    segment: Segment,
    translation: str,
    missing_entities: set[str],
) -> bool:
    source_len = max(1, len(segment.source_text.strip()))
    ratio = len(translation.strip()) / source_len
    if ratio >= 0.55:
        return False
    important = important_source_tokens(segment.source_text)
    if len(important) < 3 and not missing_entities:
        return False
    translated_folded = translation.casefold()
    missing_tokens = {token for token in important if token not in translated_folded}
    return bool(missing_entities) or len(missing_tokens) >= max(2, len(important) // 2)


def _looks_overly_literal(segment: Segment, translation: str) -> bool:
    if segment.source_text.strip() == translation.strip():
        return False
    protected_terms = {term.casefold() for term in segment.protected_terms}
    source_tokens = [
        token
        for token in _important_tokens_in_order(segment.source_text)
        if token not in protected_terms
    ]
    if len(source_tokens) < 4:
        return False
    translated_folded = translation.casefold()
    preserved = [token for token in source_tokens if token in translated_folded]
    if len(preserved) < max(4, int(len(source_tokens) * 0.6)):
        return False
    source_positions = [segment.source_text.casefold().find(token) for token in preserved]
    translation_positions = [translated_folded.find(token) for token in preserved]
    return source_positions == sorted(source_positions) and translation_positions == sorted(
        translation_positions
    )


def _has_tone_or_style_drift(source: str, translation: str) -> bool:
    folded_source = source.casefold()
    folded_translation = translation.casefold()
    if suspicious_addition_terms(translation) - suspicious_addition_terms(source):
        return True
    if PROMOTIONAL_TONE_TERMS.intersection(folded_translation.split()) and not (
        PROMOTIONAL_TONE_TERMS.intersection(folded_source.split())
    ):
        return True
    first_word = folded_translation.split(maxsplit=1)[0] if folded_translation.split() else ""
    if first_word in IMPERATIVE_STARTERS and first_word not in folded_source.split():
        return True
    return translation.count("!") > source.count("!") + 1


def _remove_placeholders_and_urls(text: str) -> str:
    return PLACEHOLDER_OR_URL_PATTERN.sub(" ", text)


def _issue(segment_id: str, category: str, explanation: str) -> QaIssue:
    return QaIssue(
        issue_id=f"qa_{stable_id(segment_id, category, explanation)}",
        segment_id=segment_id,
        category=category,
        severity=FindingSeverity.WARNING,
        explanation=explanation,
    )
