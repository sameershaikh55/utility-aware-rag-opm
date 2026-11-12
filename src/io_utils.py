"""Atomic disk I/O helpers for JSON, JSONL, and Markdown artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

PathLike = Union[str, Path]


def _ensure_parent_dir(path: Path) -> None:
    """Create parent directories if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text_atomic(path: PathLike, content: str, encoding: str = "utf-8") -> Path:
    """Atomically write string content to disk using a temporary file and os.replace."""
    target = Path(path).resolve()
    _ensure_parent_dir(target)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, mode="w", encoding=encoding, newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return target


def read_json(path: PathLike) -> Dict[str, Any]:
    """Read and parse a UTF-8 JSON file."""
    target = Path(path)
    with target.open(mode="r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(
    path: PathLike, data: Any, indent: int = 2, sort_keys: bool = False
) -> Path:
    """Serialize data as formatted JSON and write atomically to disk."""
    serialized = json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=sort_keys)
    return write_text_atomic(path, serialized + "\n")


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    """Read a JSON Lines (.jsonl) file into a list of dictionaries."""
    target = Path(path)
    records: List[Dict[str, Any]] = []
    with target.open(mode="r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {target}: {exc}"
                ) from exc
    return records


def write_jsonl_atomic(path: PathLike, records: Iterable[Dict[str, Any]]) -> Path:
    """Atomically write an iterable of dictionaries to a JSON Lines (.jsonl) file."""
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    content = "\n".join(lines) + ("\n" if lines else "")
    return write_text_atomic(path, content)


def read_markdown(path: PathLike) -> str:
    """Read a UTF-8 Markdown document from disk."""
    target = Path(path)
    return target.read_text(encoding="utf-8")


def write_markdown_atomic(path: PathLike, markdown_text: str) -> Path:
    """Atomically write a UTF-8 Markdown document to disk."""
    return write_text_atomic(path, markdown_text)
