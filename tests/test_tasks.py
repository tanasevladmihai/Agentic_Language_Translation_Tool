from pathlib import Path

from agentic_language_translation_tool.models import BatchPurpose, Segment
from agentic_language_translation_tool.tasks import create_batches, render_verification_batch


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
    assert "Apple" in prompt
    assert "{product}" in prompt
    assert "<TRANSLATION_HERE>" in prompt
