from pathlib import Path

from agentic_language_translation_tool.models import (
    Batch,
    BatchPurpose,
    FindingCategory,
    FindingSeverity,
    Segment,
    VerificationFinding,
)
from agentic_language_translation_tool.tasks import (
    create_batches,
    render_correction_batch,
    render_verification_batch,
)


def test_verification_prompt_contains_concept_aware_criteria(tmp_path: Path) -> None:
    segment = Segment(
        segment_id="seg_1",
        source_text="Apple released a new API for {product}.",
        format="markdown",
        placeholders=["{product}"],
        protected_terms=["Apple"],
        checksum="abc",
    )
    batch = create_batches(
        [segment],
        purpose=BatchPurpose.VERIFICATION,
        output_dir=tmp_path / "verification_batches",
    )[0]

    prompt = render_verification_batch(batch, [segment])

    assert "hallucinations" in prompt
    assert "overly literal" in prompt
    assert "added meaning" in prompt
    assert "omitted meaning" in prompt
    assert "false" in prompt
    assert "concrete source and translation evidence" in prompt
    assert "do not assume" in prompt
    assert "version strings" in prompt
    assert "v2.1" in prompt
    assert "15%" in prompt
    assert "API_KEY" in prompt
    assert '"Export"' in prompt
    assert "Apple" in prompt
    assert "{product}" in prompt
    assert "<TRANSLATION_HERE>" in prompt


def test_correction_prompt_contains_adversarial_checklist() -> None:
    segment = Segment(
        segment_id="seg_1",
        source_text="Apple released Vision Pro 2.",
        translated_text="Apfel hat Vision veroffentlicht.",
        format="txt",
        protected_terms=["Apple"],
        checksum="abc",
    )
    batch = Batch(
        batch_id="correction_0001",
        purpose=BatchPurpose.CORRECTION,
        segment_ids=[segment.segment_id],
        token_estimate=10,
        file="correction_batches/correction_0001.md",
    )
    finding = VerificationFinding(
        finding_id="finding_1",
        segment_id=segment.segment_id,
        category=FindingCategory.ENTITY_DAMAGE,
        severity=FindingSeverity.ERROR,
        explanation="Brand and product name were damaged.",
        evidence="Apple became Apfel.",
        correction_guidance="Restore brand and product names.",
    )

    prompt = render_correction_batch(batch, segment, [finding])

    assert "added meaning" in prompt
    assert "omitted meaning" in prompt
    assert "entity damage" in prompt
    assert "false friends" in prompt
    assert "tone/style drift" in prompt
    assert "source and target languages" in prompt
    assert "version strings" in prompt
    assert "quoted labels" in prompt
    assert "Brand and product name were damaged." in prompt
