from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


def parse_chat_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or ":" not in line:
        return None

    user, content = line.split(":", 1)
    user = user.strip()
    content = content.strip()

    if not user or not content:
        return None

    return user, content


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")
            count += 1
    return count


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def format_message(user: str, content: str) -> str:
    return f"{user}:{content}"


def profile_filename(user: str) -> str:
    digest = hashlib.sha1(user.encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[\\/\0\r\n\t:]+", "_", user).strip()
    stem = re.sub(r"\s+", "_", stem)
    stem = stem[:60].strip("._ ")
    if not stem:
        stem = "user"
    return f"{stem}-{digest}.md"
