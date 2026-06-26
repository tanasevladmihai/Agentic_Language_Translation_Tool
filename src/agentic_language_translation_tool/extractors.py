"""Initial TXT and Markdown extraction helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from agentic_language_translation_tool.io import stable_id
from agentic_language_translation_tool.models import Segment

PLACEHOLDER_PATTERN = re.compile(
    r"(`[^`]+`|\{[^{}]+\}|%\([^)]+\)s|%[sd]|https?://\S+|[A-Z][A-Z0-9_]{2,})"
)
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


@dataclass(frozen=True)
class ExtractionResult:
    """Segments and rebuild hints extracted from a document."""

    document_format: str
    segments: list[Segment]
    structure: dict[str, object]


class UnsupportedFormatError(ValueError):
    """Raised when no extractor exists for a file type."""


def inspect_input(path: Path) -> dict[str, object]:
    """Return lightweight metadata for an input document."""
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower().lstrip(".") or "txt"
    supported = suffix in {"txt", "md", "markdown"}
    return {
        "path": str(path),
        "format": "markdown" if suffix in {"md", "markdown"} else suffix,
        "supported": supported,
        "size_bytes": path.stat().st_size,
        "planned_extractors": ["txt", "markdown"],
        "planned_plugin_stubs": ["docx", "pdf"],
    }


def extract_document(path: Path) -> ExtractionResult:
    """Extract supported documents into segments."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".md", ".markdown"}:
        return extract_markdown(text, path)
    if suffix == ".txt" or suffix == "":
        return extract_txt(text, path)
    if suffix in {".docx", ".pdf"}:
        raise UnsupportedFormatError(f"{suffix} parsing is planned as a plugin after this slice")
    raise UnsupportedFormatError(f"unsupported file format: {suffix or '<none>'}")


def extract_txt(text: str, source_path: Path) -> ExtractionResult:
    """Extract TXT paragraphs while preserving deterministic ordering."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    segments = [
        _make_segment(
            source_text=paragraph,
            document_format="txt",
            source_path=source_path,
            index=index,
            path=["paragraph", str(index + 1)],
            style_tags=["paragraph"],
        )
        for index, paragraph in enumerate(paragraphs)
    ]
    return ExtractionResult(
        document_format="txt",
        segments=segments,
        structure={
            "format": "txt",
            "block_count": len(segments),
            "rebuild_strategy": "join_translated_paragraphs_with_blank_lines",
        },
    )


def extract_markdown(text: str, source_path: Path) -> ExtractionResult:
    """Extract Markdown block-level segments with simple formatting hints."""
    blocks = _split_markdown_blocks(text)
    segments: list[Segment] = []
    structure_blocks: list[dict[str, object]] = []
    for index, block in enumerate(blocks):
        block_type = _markdown_block_type(block)
        translatable = block_type != "code_fence"
        structure_blocks.append({"index": index, "type": block_type, "translatable": translatable})
        if not translatable:
            continue
        style_tags = [block_type]
        segments.append(
            _make_segment(
                source_text=block,
                document_format="markdown",
                source_path=source_path,
                index=index,
                path=[block_type, str(index + 1)],
                style_tags=style_tags,
            )
        )
    return ExtractionResult(
        document_format="markdown",
        segments=segments,
        structure={
            "format": "markdown",
            "blocks": structure_blocks,
            "rebuild_strategy": "replace_translatable_blocks_by_segment_id",
        },
    )


def _split_markdown_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            current.append(line)
            if in_fence:
                blocks.append("\n".join(current).strip())
                current = []
            in_fence = not in_fence
            continue
        if in_fence:
            current.append(line)
            continue
        if not line.strip():
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return blocks


def _markdown_block_type(block: str) -> str:
    stripped = block.lstrip()
    if stripped.startswith("```"):
        return "code_fence"
    if stripped.startswith("#"):
        return "heading"
    if stripped.startswith(("- ", "* ", "+ ")) or re.match(r"^\d+\.\s", stripped):
        return "list"
    if "|" in stripped and "\n" in stripped:
        return "table"
    if MARKDOWN_LINK_PATTERN.search(stripped):
        return "linked_paragraph"
    return "paragraph"


def _make_segment(
    *,
    source_text: str,
    document_format: str,
    source_path: Path,
    index: int,
    path: list[str],
    style_tags: list[str],
) -> Segment:
    placeholders = sorted(set(PLACEHOLDER_PATTERN.findall(source_text)))
    protected_terms = sorted(term for term in placeholders if _looks_protected(term))
    checksum = stable_id(source_text, length=32)
    segment_id = f"seg_{stable_id(str(source_path), str(index), source_text)}"
    return Segment(
        segment_id=segment_id,
        source_text=source_text,
        format=document_format,
        context=f"{source_path.name} block {index + 1}",
        path=path,
        style_tags=style_tags,
        placeholders=placeholders,
        protected_terms=protected_terms,
        checksum=checksum,
    )


def _looks_protected(value: str) -> bool:
    return value.startswith(("http://", "https://", "`")) or value.isupper()
