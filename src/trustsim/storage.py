"""UTF-8 JSON and JSONL files, including the append-only logs a run resumes from."""

import json
import logging
import threading
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every JSON object in a JSONL file."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write records as JSONL, replacing the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    """Write one JSON document, replacing the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class RecordLog:
    """Append-only JSONL log shared by worker threads. A later record replaces an earlier one
    with the same key, so a retried conversation's new result wins."""

    def __init__(self, path: Path, key: Callable[[dict[str, Any]], Hashable]) -> None:
        self._path = path
        self._key = key
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        """Add one record to the end of the log."""
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def latest(self) -> dict[Hashable, dict[str, Any]]:
        """The most recent record for each key."""
        if not self._path.exists():
            return {}
        lines = [
            line for line in self._path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        records: dict[Hashable, dict[str, Any]] = {}
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A run killed mid-write can leave a partial last line; that unit is simply redone.
                if index == len(lines) - 1:
                    LOGGER.warning("ignoring a partial last line in %s", self._path)
                    continue
                raise
            records[self._key(record)] = record
        return records
