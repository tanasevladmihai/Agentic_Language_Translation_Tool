"""Language-agnostic deterministic checks for adversarial translation QA."""

from __future__ import annotations

from dataclasses import dataclass, field

import regex

from agentic_language_translation_tool.io import stable_id
from agentic_language_translation_tool.models import FindingSeverity, QaIssue, Segment

URL_PATTERN = regex.compile(r"https?://[^\s)>\]]+", regex.IGNORECASE)
EMAIL_PATTERN = regex.compile(r"[\w.+-]+@[\w.-]+\.[\p{L}]{2,}", regex.IGNORECASE)
CODE_SPAN_PATTERN = regex.compile(r"`[^`]+`")
PLACEHOLDER_PATTERN = regex.compile(r"\{[^{}]+\}|%\([^)]+\)s|%[sd]|\$[A-Za-z_][\w-]*")
VERSION_PATTERN = regex.compile(r"\bv?\d+(?:\.\d+){1,4}(?:[-+][\p{L}\p{N}._-]+)?\b")
PERCENT_PATTERN = regex.compile(r"(?<![\p{L}\p{N}])\d+(?:[.,]\d+)?\s?%")
CURRENCY_PATTERN = regex.compile(
    r"(?:[$€£¥]\s?\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s?(?:USD|EUR|GBP|JPY))"
)
DATE_PATTERN = regex.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b"
)
NUMBER_PATTERN = regex.compile(r"(?<![\p{L}\p{N}])\d+(?:[.,]\d+)?(?![\p{L}\p{N}])")
UNIT_PATTERN = regex.compile(
    r"(?<![\p{L}\p{N}])\d+(?:[.,]\d+)?\s?"
    r"(?:kg|g|mg|km|m|cm|mm|mb|gb|tb|kb|hz|khz|mhz|ghz|ms|s|min|h|°c|°f)\b",
    regex.IGNORECASE,
)
ACRONYM_PATTERN = regex.compile(r"\b[\p{Lu}]{2,}(?:[-_/][\p{Lu}\p{N}]+)*\b")
CODE_IDENTIFIER_PATTERN = regex.compile(
    r"\b(?:[A-Za-z_][A-Za-z0-9]*_[A-Za-z0-9_]+|[a-z]+(?:[A-Z][a-z0-9]+)+|[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+)\b"
)
QUOTED_LABEL_PATTERN = regex.compile(
    r'(?:"[^"\n]{1,80}"|“[^”\n]{1,80}”|«[^»\n]{1,80}»|「[^」\n]{1,80}」|『[^』\n]{1,80}』)'
)
BRACKETED_LABEL_PATTERN = regex.compile(r"(?:\[[^\]\n]{1,80}\]|\([^\)\n]{1,80}\))")
PRODUCT_IDENTIFIER_PATTERN = regex.compile(
    r"\b[\p{L}]+(?:[-_][\p{L}\p{N}]+)+\b|\b[\p{L}]*\p{Lu}[\p{L}\p{N}]*\d[\p{L}\p{N}]*\b"
)
WORD_PATTERN = regex.compile(r"[\p{L}\p{M}\p{N}]+", regex.VERSION1)
SPACELESS_SCRIPT_PATTERN = regex.compile(
    r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Thai}\p{Lao}\p{Khmer}]",
    regex.VERSION1,
)
ALL_CAPS_EMPHASIS_PATTERN = regex.compile(r"\b[\p{Lu}\p{N}]{5,}\b")


@dataclass(frozen=True)
class SignalSet:
    """Language-neutral signals extracted from a source or translation."""

    urls: set[str] = field(default_factory=set)
    emails: set[str] = field(default_factory=set)
    code_spans: set[str] = field(default_factory=set)
    placeholders: set[str] = field(default_factory=set)
    versions: set[str] = field(default_factory=set)
    percentages: set[str] = field(default_factory=set)
    currencies: set[str] = field(default_factory=set)
    dates: set[str] = field(default_factory=set)
    units: set[str] = field(default_factory=set)
    numbers: set[str] = field(default_factory=set)
    acronyms: set[str] = field(default_factory=set)
    code_identifiers: set[str] = field(default_factory=set)
    quoted_labels: set[str] = field(default_factory=set)
    bracketed_labels: set[str] = field(default_factory=set)
    product_identifiers: set[str] = field(default_factory=set)
    protected_terms: set[str] = field(default_factory=set)
    content_tokens: set[str] = field(default_factory=set)

    def preserve_exact(self) -> set[str]:
        """Return source-side signals expected to survive translation exactly."""
        return set().union(
            self.urls,
            self.emails,
            self.code_spans,
            self.placeholders,
            self.versions,
            self.percentages,
            self.currencies,
            self.dates,
            self.units,
            self.acronyms,
            self.code_identifiers,
            self.quoted_labels,
            self.bracketed_labels,
            self.product_identifiers,
            self.protected_terms,
        )

    def addition_risks(self) -> set[str]:
        """Return target-side signals whose addition can indicate unsupported meaning."""
        return set().union(
            self.urls,
            self.emails,
            self.versions,
            self.percentages,
            self.currencies,
            self.dates,
            self.units,
            self.acronyms,
            self.quoted_labels,
            self.bracketed_labels,
            self.product_identifiers,
        )


def detect_adversarial_qa_issues(
    segment: Segment,
    translation: str,
    source_language: str = "",
    target_language: str = "",
) -> list[QaIssue]:
    """Return language-agnostic warning-level adversarial QA risks."""
    if not translation.strip():
        return []
    source_signals = extract_signal_set(segment.source_text, segment)
    target_signals = extract_signal_set(translation, segment)
    issues: list[QaIssue] = []

    missing_signals = _missing_preserved_signals(source_signals, target_signals)
    if missing_signals:
        issues.append(
            _issue(
                segment.segment_id,
                "possible_entity_damage",
                "Source signals may be missing or changed: "
                f"{', '.join(sorted(missing_signals))}.",
            )
        )

    if _has_possible_omission(segment.source_text, translation, source_signals, target_signals):
        issues.append(
            _issue(
                segment.segment_id,
                "possible_omission",
                "Translation may omit source content based on length and signal loss.",
            )
        )

    added_signals = _added_target_signals(source_signals, target_signals)
    if _has_possible_hallucination(segment.source_text, translation, added_signals):
        issues.append(
            _issue(
                segment.segment_id,
                "possible_hallucination",
                "Translation may add unsupported signals or substantially expand content: "
                f"{', '.join(sorted(added_signals)) or 'length expansion'}.",
            )
        )

    if _looks_overly_literal(
        segment,
        translation,
        source_signals,
        target_signals,
        source_language,
        target_language,
    ):
        issues.append(
            _issue(
                segment.segment_id,
                "possible_overly_literal",
                "Translation preserves unusually much source wording across languages.",
            )
        )

    if _has_tone_or_style_drift(segment.source_text, translation):
        issues.append(
            _issue(
                segment.segment_id,
                "possible_tone_or_style_drift",
                "Translation appears to shift punctuation or emphasis style.",
            )
        )
    return issues


def extract_signal_set(text: str, segment: Segment | None = None) -> SignalSet:
    """Extract language-neutral QA signals from text."""
    protected = set(segment.protected_terms) if segment is not None else set()
    placeholders = set(segment.placeholders) if segment is not None else set()
    placeholders.update(_matches(PLACEHOLDER_PATTERN, text))
    return SignalSet(
        urls=_matches(URL_PATTERN, text),
        emails=_matches(EMAIL_PATTERN, text),
        code_spans=_matches(CODE_SPAN_PATTERN, text),
        placeholders=placeholders,
        versions=_matches(VERSION_PATTERN, text),
        percentages=_matches(PERCENT_PATTERN, text),
        currencies=_matches(CURRENCY_PATTERN, text),
        dates=_matches(DATE_PATTERN, text),
        units=_matches(UNIT_PATTERN, text),
        numbers=_standalone_numbers(text),
        acronyms=_matches(ACRONYM_PATTERN, text),
        code_identifiers=_matches(CODE_IDENTIFIER_PATTERN, text),
        quoted_labels=_matches(QUOTED_LABEL_PATTERN, text),
        bracketed_labels=_matches(BRACKETED_LABEL_PATTERN, text),
        product_identifiers=_matches(PRODUCT_IDENTIFIER_PATTERN, text),
        protected_terms=protected,
        content_tokens=content_signal_tokens(text),
    )


def protected_signal_tokens(text: str, segment: Segment | None = None) -> set[str]:
    """Return language-neutral tokens that should usually be preserved."""
    return extract_signal_set(text, segment).preserve_exact()


def content_signal_tokens(text: str) -> set[str]:
    """Return Unicode-aware content tokens without English stopword assumptions."""
    stripped = _remove_noise(text)
    if SPACELESS_SCRIPT_PATTERN.search(stripped):
        script_chars = {char for char in stripped if SPACELESS_SCRIPT_PATTERN.fullmatch(char)}
        word_tokens = {
            token.casefold()
            for token in WORD_PATTERN.findall(stripped)
            if len(token) >= 3 and not SPACELESS_SCRIPT_PATTERN.fullmatch(token)
        }
        return script_chars | word_tokens
    tokens = {token.casefold() for token in WORD_PATTERN.findall(stripped) if len(token) >= 3}
    return tokens


def _missing_preserved_signals(source: SignalSet, target: SignalSet) -> set[str]:
    return _casefold_difference(source.preserve_exact(), target.preserve_exact())


def _added_target_signals(source: SignalSet, target: SignalSet) -> set[str]:
    return _casefold_difference(target.addition_risks(), source.addition_risks())


def _has_possible_hallucination(source: str, translation: str, added_signals: set[str]) -> bool:
    source_len = max(1, len(source.strip()))
    length_ratio = len(translation.strip()) / source_len
    return length_ratio >= 1.8 and bool(added_signals)


def _has_possible_omission(
    source: str,
    translation: str,
    source_signals: SignalSet,
    target_signals: SignalSet,
) -> bool:
    source_len = max(1, len(source.strip()))
    length_ratio = len(translation.strip()) / source_len
    missing_exact = _missing_preserved_signals(source_signals, target_signals)
    if length_ratio >= 0.55 and len(missing_exact) < 2:
        return False
    source_tokens = source_signals.content_tokens
    target_tokens = target_signals.content_tokens
    if len(source_tokens) < 4 and not missing_exact:
        return False
    missing_tokens = _casefold_difference(source_tokens, target_tokens)
    return bool(missing_exact) or len(missing_tokens) >= max(3, len(source_tokens) // 2)


def _looks_overly_literal(
    segment: Segment,
    translation: str,
    source_signals: SignalSet,
    target_signals: SignalSet,
    source_language: str,
    target_language: str,
) -> bool:
    if _same_language(source_language, target_language):
        return False
    if segment.source_text.strip() == translation.strip():
        return False
    protected = {token.casefold() for token in source_signals.preserve_exact()}
    source_tokens = [
        token
        for token in _content_tokens_in_order(segment.source_text)
        if token not in protected
    ]
    if len(source_tokens) < 4:
        return False
    target_tokens = target_signals.content_tokens
    preserved = [token for token in source_tokens if token in target_tokens]
    return len(preserved) >= max(4, int(len(source_tokens) * 0.65))


def _has_tone_or_style_drift(source: str, translation: str) -> bool:
    if translation.count("!") > source.count("!") + 2:
        return True
    if translation.count("?") > source.count("?") + 2:
        return True
    if regex.search(r"([!?])\1{2,}", translation) and not regex.search(r"([!?])\1{2,}", source):
        return True
    source_emphasis = _matches(ALL_CAPS_EMPHASIS_PATTERN, source)
    target_emphasis = _matches(ALL_CAPS_EMPHASIS_PATTERN, translation)
    return bool(_casefold_difference(target_emphasis, source_emphasis))


def _content_tokens_in_order(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in WORD_PATTERN.findall(_remove_noise(text)):
        normalized = token.casefold()
        if len(normalized) < 3 or normalized in seen:
            continue
        if SPACELESS_SCRIPT_PATTERN.search(normalized):
            continue
        ordered.append(normalized)
        seen.add(normalized)
    if ordered or not SPACELESS_SCRIPT_PATTERN.search(text):
        return ordered
    return [char for char in text if SPACELESS_SCRIPT_PATTERN.fullmatch(char)]


def _matches(pattern: regex.Pattern[str], text: str) -> set[str]:
    return {match.group(0).strip() for match in pattern.finditer(text) if match.group(0).strip()}


def _standalone_numbers(text: str) -> set[str]:
    numbers = _matches(NUMBER_PATTERN, text)
    covered = set().union(
        _matches(VERSION_PATTERN, text),
        _matches(PERCENT_PATTERN, text),
        _matches(CURRENCY_PATTERN, text),
        _matches(DATE_PATTERN, text),
        _matches(UNIT_PATTERN, text),
    )
    covered_parts = {part for value in covered for part in NUMBER_PATTERN.findall(value)}
    return {number for number in numbers if number not in covered_parts}


def _casefold_difference(left: set[str], right: set[str]) -> set[str]:
    right_folded = {value.casefold() for value in right}
    return {value for value in left if value.casefold() not in right_folded}


def _same_language(source_language: str, target_language: str) -> bool:
    if not source_language.strip() or not target_language.strip():
        return False
    return source_language.strip().casefold() == target_language.strip().casefold()


def _remove_noise(text: str) -> str:
    stripped = text
    for pattern in [
        URL_PATTERN,
        EMAIL_PATTERN,
        CODE_SPAN_PATTERN,
        PLACEHOLDER_PATTERN,
        QUOTED_LABEL_PATTERN,
        BRACKETED_LABEL_PATTERN,
    ]:
        stripped = pattern.sub(" ", stripped)
    return stripped


def _issue(segment_id: str, category: str, explanation: str) -> QaIssue:
    return QaIssue(
        issue_id=f"qa_{stable_id(segment_id, category, explanation)}",
        segment_id=segment_id,
        category=category,
        severity=FindingSeverity.WARNING,
        explanation=explanation,
    )
