"""Agent task-file generation."""

from __future__ import annotations

from pathlib import Path

from agentic_language_translation_tool.io import atomic_write_text
from agentic_language_translation_tool.models import (
    Batch,
    BatchPurpose,
    Segment,
    VerificationFinding,
)


def estimate_tokens(text: str) -> int:
    """Estimate tokens without requiring provider-specific tokenization."""
    return max(1, len(text) // 4)


def create_batches(
    segments: list[Segment],
    *,
    purpose: BatchPurpose,
    output_dir: Path,
    max_segments: int = 10,
) -> list[Batch]:
    """Create batch metadata and task files for agent workflows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    batches: list[Batch] = []
    for index in range(0, len(segments), max_segments):
        batch_segments = segments[index : index + max_segments]
        batch_number = (index // max_segments) + 1
        batch_id = f"{purpose.value}_{batch_number:04d}"
        file_name = f"{batch_id}.md"
        batch = Batch(
            batch_id=batch_id,
            purpose=purpose,
            segment_ids=[segment.segment_id for segment in batch_segments],
            token_estimate=sum(estimate_tokens(segment.source_text) for segment in batch_segments),
            file=str(output_dir.name + "/" + file_name),
        )
        content = (
            render_translation_batch(batch, batch_segments)
            if purpose == BatchPurpose.TRANSLATION
            else render_verification_batch(batch, batch_segments)
        )
        atomic_write_text(output_dir / file_name, content)
        batches.append(batch)
    return batches


def render_translation_batch(batch: Batch, segments: list[Segment]) -> str:
    """Render a translation task for an agent."""
    lines = [
        f"# Translation Batch {batch.batch_id}",
        "",
        "Translate each segment faithfully while preserving placeholders, links, code,",
        "brand names, product names, and formatting markers. Return JSONL with",
        "`segment_id` and `translated_text` fields.",
        "",
    ]
    for segment in segments:
        lines.extend(_segment_section(segment, include_translation_placeholder=True))
    return "\n".join(lines).rstrip() + "\n"


def render_verification_batch(batch: Batch, segments: list[Segment]) -> str:
    """Render a concept-aware verification task for an agent."""
    lines = [
        f"# Verification Batch {batch.batch_id}",
        "",
        "Compare each translation against the original source segment and context.",
        "Use an adversarial checklist for hallucinations: added meaning, omitted meaning,",
        "damaged entities, damaged brands/products/versions, bad localization, false",
        "friends, overly literal or over-literal phrasing, tone/style drift, formatting damage,",
        "and terminology drift.",
        "",
        "Important example: if Apple refers to the technology company, do not treat",
        "a literal translation such as German `Apfel` as acceptable.",
        "Product names, version strings, acronyms, quoted UI labels, code, links,",
        "variables, and placeholders must remain stable unless the glossary says",
        "otherwise.",
        "",
        "Return JSONL findings with: finding_id, segment_id, category, severity,",
        "explanation, evidence, and correction_guidance. Each finding must cite",
        "concrete source and translation evidence; do not report vague concerns.",
        "",
    ]
    for segment in segments:
        lines.extend(_segment_section(segment, include_translation_placeholder=True))
    return "\n".join(lines).rstrip() + "\n"


def render_correction_batch(
    batch: Batch,
    segment: Segment,
    findings: list[VerificationFinding],
) -> str:
    """Render a focused correction task for failed verification findings."""
    lines = [
        f"# Correction Batch {batch.batch_id}",
        "",
        "Revise the current translation so it preserves the original meaning,",
        "formatting constraints, placeholders, protected terms, and document style.",
        "Check specifically for added meaning, omitted meaning, entity damage,",
        "bad localization, false friends, over-literal phrasing, tone/style drift,",
        "formatting damage, and terminology drift.",
        "Return JSONL with `segment_id` and corrected `translated_text` fields.",
        "",
    ]
    lines.extend(_segment_section(segment, include_translation_placeholder=True))
    lines.extend(["## Verification Findings", ""])
    for finding in findings:
        lines.extend(
            [
                f"- Severity: `{finding.severity.value}`",
                f"- Category: `{finding.category.value}`",
                f"- Explanation: {finding.explanation}",
                f"- Evidence: {finding.evidence or '(none)'}",
                f"- Correction guidance: {finding.correction_guidance or '(none)'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _segment_section(segment: Segment, *, include_translation_placeholder: bool) -> list[str]:
    protected_terms = ", ".join(segment.protected_terms) if segment.protected_terms else "(none)"
    placeholders = ", ".join(segment.placeholders) if segment.placeholders else "(none)"
    glossary_rules = [note for note in segment.notes if note.startswith("Glossary: ")]
    lines = [
        f"## {segment.segment_id}",
        "",
        f"- Context: {segment.context}",
        f"- Style tags: {', '.join(segment.style_tags) or '(none)'}",
        f"- Protected terms: {protected_terms}",
        f"- Placeholders: {placeholders}",
        "",
        "Source:",
        "```text",
        segment.source_text,
        "```",
        "",
    ]
    if glossary_rules:
        lines.extend(["Glossary Rules:", ""])
        lines.extend(f"- {rule}" for rule in glossary_rules)
        lines.append("")
    if include_translation_placeholder:
        lines.extend(
            [
                "Translation:",
                "```text",
                segment.translated_text or "<TRANSLATION_HERE>",
                "```",
                "",
            ]
        )
    return lines
