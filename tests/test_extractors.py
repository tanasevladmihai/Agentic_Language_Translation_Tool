from pathlib import Path

from agentic_language_translation_tool.extractors import extract_document, inspect_input


def test_txt_paragraph_segmentation(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text(
        "First paragraph.\n\nSecond paragraph with URL https://example.com",
        encoding="utf-8",
    )

    result = extract_document(source)

    assert result.document_format == "txt"
    assert len(result.segments) == 2
    assert "https://example.com" in result.segments[1].protected_terms


def test_markdown_block_segmentation_skips_code_fence(tmp_path: Path) -> None:
    source = tmp_path / "sample.md"
    source.write_text(
        "# Title\n\n"
        "Paragraph with [Apple](https://apple.com).\n\n"
        "```python\nprint('x')\n```\n\n"
        "- Item",
        encoding="utf-8",
    )

    result = extract_document(source)

    assert result.document_format == "markdown"
    assert [segment.style_tags[0] for segment in result.segments] == [
        "heading",
        "linked_paragraph",
        "list",
    ]
    assert all("print" not in segment.source_text for segment in result.segments)


def test_inspect_reports_planned_plugin_stubs(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    source.write_text("fake", encoding="utf-8")

    metadata = inspect_input(source)

    assert metadata["supported"] is False
    assert metadata["planned_plugin_stubs"] == ["docx", "pdf"]
