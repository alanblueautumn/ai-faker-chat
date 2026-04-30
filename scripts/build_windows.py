from __future__ import annotations

import argparse
from pathlib import Path

from chatlib import format_message, read_jsonl, write_jsonl


DEFAULT_OUT = Path("data/windows.jsonl")


def build_windows(messages_path: Path, before: int, after: int):
    messages = list(read_jsonl(messages_path))

    for index, message in enumerate(messages):
        before_messages = messages[max(0, index - before) : index]
        after_messages = messages[index + 1 : index + 1 + after]

        before_text = [
            format_message(str(item["user"]), str(item["content"]))
            for item in before_messages
        ]
        target_line = format_message(str(message["user"]), str(message["content"]))
        after_text = [
            format_message(str(item["user"]), str(item["content"]))
            for item in after_messages
        ]
        full_text = "\n".join([*before_text, target_line, *after_text])

        yield {
            "id": int(message["id"]),
            "target_message_id": int(message["id"]),
            "line_no": int(message["line_no"]),
            "user": message["user"],
            "before": before_text,
            "target": message["content"],
            "after": after_text,
            "full_text": full_text,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build context windows from messages JSONL.")
    parser.add_argument("messages", type=Path, help="Path to data/messages.jsonl.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output JSONL path. Default: {DEFAULT_OUT}",
    )
    parser.add_argument("--before", type=int, default=5, help="Messages before target.")
    parser.add_argument("--after", type=int, default=2, help="Messages after target.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.messages.exists():
        parser.error(f"File not found: {args.messages}")

    count = write_jsonl(args.out, build_windows(args.messages, args.before, args.after))
    print(f"Wrote {count} context windows to {args.out}")


if __name__ == "__main__":
    main()
