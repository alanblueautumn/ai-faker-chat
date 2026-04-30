from __future__ import annotations

import argparse
from pathlib import Path

from chatlib import parse_chat_line, write_jsonl


DEFAULT_OUT = Path("data/messages.jsonl")


def iter_messages(chat_file: Path):
    message_id = 0
    with chat_file.open("r", encoding="utf-8-sig", errors="replace") as file:
        for line_no, raw_line in enumerate(file, start=1):
            parsed = parse_chat_line(raw_line)
            if parsed is None:
                continue

            user, content = parsed
            message_id += 1
            yield {
                "id": message_id,
                "line_no": line_no,
                "user": user,
                "content": content,
            }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse a plain-text chat file into JSONL messages."
    )
    parser.add_argument("chat_file", type=Path, help="Path to the source txt file.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output JSONL path. Default: {DEFAULT_OUT}",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.chat_file.exists():
        parser.error(f"File not found: {args.chat_file}")
    if not args.chat_file.is_file():
        parser.error(f"Not a file: {args.chat_file}")

    count = write_jsonl(args.out, iter_messages(args.chat_file))
    print(f"Wrote {count} messages to {args.out}")


if __name__ == "__main__":
    main()
