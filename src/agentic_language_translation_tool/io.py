"""File IO helpers for the workspace protocol."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def file_sha256(path: Path) -> str:
    """Calculate a SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: str, length: int = 16) -> str:
    """Create a deterministic short ID from string parts."""
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically write UTF-8 text to a path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(content)
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: BaseModel | dict[str, Any]) -> None:
    """Atomically write JSON data."""
    if isinstance(data, BaseModel):
        payload = data.model_dump(mode="json")
    else:
        payload = data
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def read_json_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    """Read JSON data into a pydantic model."""
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict[str, Any]]) -> None:
    """Write JSON Lines data atomically."""
    lines: list[str] = []
    for row in rows:
        payload = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
        lines.append(json.dumps(payload, ensure_ascii=False))
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_jsonl_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> list[ModelT]:
    """Read JSON Lines data into pydantic models."""
    if not path.exists():
        return []
    rows: list[ModelT] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(model.model_validate_json(line))
    return rows
