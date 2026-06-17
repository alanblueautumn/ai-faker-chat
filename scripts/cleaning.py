from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable


MEDIA_MARKERS = ("[图片]", "[视频]", "[语音]", "[文件]", "[动画表情]", "[表情]")
MARKDOWN_MEDIA_RE = re.compile(r"^!\[(图片|视频|语音|文件|动画表情|表情)[^\]]*\]\([^)]+\)$")
EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u27bf"
    "]+"
)


def normalize_content(content: str) -> str:
    return re.sub(r"\s+", " ", content.strip())


def is_low_information(content: str, min_chars: int) -> bool:
    text = normalize_content(content)
    if len(text) < min_chars:
        return True
    if any(marker in text for marker in MEDIA_MARKERS):
        return True
    if MARKDOWN_MEDIA_RE.match(text):
        return True

    without_emoji = EMOJI_RE.sub("", text)
    without_symbols = re.sub(r"[\W_]+", "", without_emoji, flags=re.UNICODE)
    if not without_symbols:
        return True
    if without_symbols.isdigit():
        return True

    return False


def global_repeated_contents(rows: Iterable[dict[str, object]], threshold: int) -> set[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        content = normalize_content(str(row.get("content", "")))
        if content:
            counter[content] += 1
    return {content for content, count in counter.items() if count >= threshold}


def filter_message_rows(
    rows: list[dict[str, object]],
    min_chars: int,
    global_repeat_threshold: int,
    per_user_repeat_limit: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    global_repeats = global_repeated_contents(rows, global_repeat_threshold)
    per_user_counts: dict[str, Counter[str]] = defaultdict(Counter)
    kept: list[dict[str, object]] = []
    stats = {
        "input": len(rows),
        "kept": 0,
        "low_information": 0,
        "global_repeat": 0,
        "per_user_repeat": 0,
    }

    for row in rows:
        user = str(row.get("user", ""))
        content = normalize_content(str(row.get("content", "")))

        if is_low_information(content, min_chars):
            stats["low_information"] += 1
            continue
        if content in global_repeats:
            stats["global_repeat"] += 1
            continue

        per_user_counts[user][content] += 1
        if per_user_counts[user][content] > per_user_repeat_limit:
            stats["per_user_repeat"] += 1
            continue

        kept.append(row)

    stats["kept"] = len(kept)
    return kept, stats


def filter_window_rows(
    rows: list[dict[str, object]],
    min_chars: int,
    global_repeat_threshold: int,
    per_user_repeat_limit: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    message_like_rows = [
        {"user": row.get("user", ""), "content": row.get("target", "")}
        for row in rows
    ]
    global_repeats = global_repeated_contents(message_like_rows, global_repeat_threshold)
    per_user_counts: dict[str, Counter[str]] = defaultdict(Counter)
    kept: list[dict[str, object]] = []
    stats = {
        "input": len(rows),
        "kept": 0,
        "low_information": 0,
        "global_repeat": 0,
        "per_user_repeat": 0,
    }

    for row in rows:
        user = str(row.get("user", ""))
        content = normalize_content(str(row.get("target", "")))

        if is_low_information(content, min_chars):
            stats["low_information"] += 1
            continue
        if content in global_repeats:
            stats["global_repeat"] += 1
            continue

        per_user_counts[user][content] += 1
        if per_user_counts[user][content] > per_user_repeat_limit:
            stats["per_user_repeat"] += 1
            continue

        kept.append(row)

    stats["kept"] = len(kept)
    return kept, stats
