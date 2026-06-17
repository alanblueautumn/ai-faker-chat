from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from chatlib import read_jsonl, write_json


DEFAULT_OUT = Path("data/reports/user_stats.json")
PUNCTUATION_RE = re.compile(r"[!?！？。.,，、~～…]+")
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


def imitation_grade(message_count: int) -> str:
    if message_count >= 1000:
        return "A"
    if message_count >= 500:
        return "B"
    if message_count >= 100:
        return "C"
    return "D"


def build_stats(messages_path: Path, sample_limit: int) -> dict[str, object]:
    stats: dict[str, dict[str, object]] = {}
    punctuation: dict[str, Counter[str]] = defaultdict(Counter)
    emojis: dict[str, Counter[str]] = defaultdict(Counter)
    active_hours: dict[str, Counter[str]] = defaultdict(Counter)

    for message in read_jsonl(messages_path):
        user = str(message["user"])
        content = str(message["content"])
        message_time = str(message.get("time", ""))
        content_len = len(content)

        if user not in stats:
            stats[user] = {
                "messages": 0,
                "chars": 0,
                "min_length": None,
                "max_length": 0,
                "avg_length": 0,
                "short_messages": 0,
                "examples": [],
                "grade": "D",
            }

        user_stats = stats[user]
        user_stats["messages"] = int(user_stats["messages"]) + 1
        user_stats["chars"] = int(user_stats["chars"]) + content_len
        user_stats["max_length"] = max(int(user_stats["max_length"]), content_len)
        min_length = user_stats["min_length"]
        user_stats["min_length"] = (
            content_len if min_length is None else min(int(min_length), content_len)
        )
        if content_len <= 4:
            user_stats["short_messages"] = int(user_stats["short_messages"]) + 1
        examples = user_stats["examples"]
        if isinstance(examples, list) and len(examples) < sample_limit:
            examples.append(content)

        for match in PUNCTUATION_RE.findall(content):
            punctuation[user][match] += 1
        for match in EMOJI_RE.findall(content):
            emojis[user][match] += 1
        if len(message_time) >= 2:
            active_hours[user][message_time[:2]] += 1

    for user, user_stats in stats.items():
        messages = int(user_stats["messages"])
        chars = int(user_stats["chars"])
        user_stats["avg_length"] = round(chars / messages, 2) if messages else 0
        user_stats["short_message_ratio"] = round(
            int(user_stats["short_messages"]) / messages, 4
        ) if messages else 0
        user_stats["grade"] = imitation_grade(messages)
        user_stats["top_punctuation"] = punctuation[user].most_common(20)
        user_stats["top_emoji"] = emojis[user].most_common(20)
        user_stats["active_hours"] = {
            hour: active_hours[user].get(f"{hour:02d}", 0) for hour in range(24)
        }

    ranked_users = sorted(
        stats,
        key=lambda name: (int(stats[name]["messages"]), int(stats[name]["chars"])),
        reverse=True,
    )

    return {
        "source": str(messages_path),
        "total_users": len(stats),
        "users": {user: stats[user] for user in ranked_users},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build user statistics from messages JSONL.")
    parser.add_argument("messages", type=Path, help="Path to data/messages.jsonl.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output JSON path. Default: {DEFAULT_OUT}",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="Number of example messages to keep per user. Default: 20.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.messages.exists():
        parser.error(f"File not found: {args.messages}")

    report = build_stats(args.messages, sample_limit=args.sample_limit)
    write_json(args.out, report)
    print(f"Wrote stats for {report['total_users']} users to {args.out}")


if __name__ == "__main__":
    main()
