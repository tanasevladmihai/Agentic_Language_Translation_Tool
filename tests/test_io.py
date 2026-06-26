from pathlib import Path

from agentic_language_translation_tool.io import (
    atomic_write_json,
    atomic_write_text,
    read_json_model,
    read_jsonl_model,
    stable_id,
    write_jsonl,
)
from agentic_language_translation_tool.models import JobManifest, Segment


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    segment = Segment(
        segment_id="seg_1",
        source_text="Hello {name}",
        format="txt",
        placeholders=["{name}"],
        checksum="abc",
    )

    write_jsonl(path, [segment])

    assert read_jsonl_model(path, Segment) == [segment]


def test_atomic_writes_replace_content(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"

    atomic_write_text(path, "first")
    atomic_write_text(path, "second")

    assert path.read_text(encoding="utf-8") == "second"
    assert not list(tmp_path.glob("*.tmp"))


def test_json_model_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "job.json"
    manifest = JobManifest(
        job_id="job_1",
        source_path="source.txt",
        workspace_path="workspace",
        source_language="en",
        target_language="fr",
        tool_version="0.1.0",
        source_checksum="abc",
    )

    atomic_write_json(path, manifest)

    assert read_json_model(path, JobManifest).job_id == "job_1"


def test_stable_id_is_deterministic() -> None:
    assert stable_id("a", "b") == stable_id("a", "b")
    assert stable_id("a", "b") != stable_id("b", "a")
