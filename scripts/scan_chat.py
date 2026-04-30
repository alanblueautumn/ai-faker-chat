from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


DEFAULT_TOP_USERS = 30
DEFAULT_SAMPLE_LIMIT = 20


def parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or ":" not in line:
        return None

    user, content = line.split(":", 1)
    user = user.strip()
    content = content.strip()

    if not user or not content:
        return None

    return user, content


def scan_chat_file(path: Path, sample_limit: int) -> dict[str, object]:
    user_counts: Counter[str] = Counter()
    user_chars: Counter[str] = Counter()
    malformed_samples: list[tuple[int, str]] = []

    total_lines = 0
    parsed_lines = 0
    empty_lines = 0

    with path.open("r", encoding="utf-8-sig", errors="replace") as file:
        for line_number, raw_line in enumerate(file, start=1):
            total_lines += 1
            line = raw_line.rstrip("\n")

            if not line.strip():
                empty_lines += 1
                continue

            parsed = parse_line(line)
            if parsed is None:
                if len(malformed_samples) < sample_limit:
                    malformed_samples.append((line_number, line[:200]))
                continue

            user, content = parsed
            parsed_lines += 1
            user_counts[user] += 1
            user_chars[user] += len(content)

    return {
        "total_lines": total_lines,
        "parsed_lines": parsed_lines,
        "empty_lines": empty_lines,
        "malformed_lines": total_lines - parsed_lines - empty_lines,
        "user_counts": user_counts,
        "user_chars": user_chars,
        "malformed_samples": malformed_samples,
    }


def print_report(result: dict[str, object], top_users: int) -> None:
    total_lines = int(result["total_lines"])
    parsed_lines = int(result["parsed_lines"])
    empty_lines = int(result["empty_lines"])
    malformed_lines = int(result["malformed_lines"])
    user_counts = result["user_counts"]
    user_chars = result["user_chars"]
    malformed_samples = result["malformed_samples"]

    parse_rate = parsed_lines / max(total_lines - empty_lines, 1) * 100

    print("Chat Scan Report")
    print("================")
    print(f"Total lines:      {total_lines}")
    print(f"Empty lines:      {empty_lines}")
    print(f"Parsed messages:  {parsed_lines}")
    print(f"Malformed lines:  {malformed_lines}")
    print(f"Parse rate:       {parse_rate:.2f}%")
    print(f"Users found:      {len(user_counts)}")
    print()

    print(f"Top users by message count (top {top_users})")
    print("----------------------------------------")
    print(f"{'rank':>4}  {'messages':>8}  {'chars':>8}  user")
    for rank, (user, count) in enumerate(user_counts.most_common(top_users), start=1):
        print(f"{rank:>4}  {count:>8}  {user_chars[user]:>8}  {user}")

    if malformed_samples:
        print()
        print("Malformed line samples")
        print("----------------------")
        for line_number, line in malformed_samples:
            print(f"{line_number}: {line}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan a plain-text group chat file formatted as 'user:message'."
    )
    parser.add_argument("chat_file", type=Path, help="Path to the chat txt file.")
    parser.add_argument(
        "--top-users",
        type=int,
        default=DEFAULT_TOP_USERS,
        help=f"How many users to show in the ranking. Default: {DEFAULT_TOP_USERS}.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=DEFAULT_SAMPLE_LIMIT,
        help=f"How many malformed line samples to print. Default: {DEFAULT_SAMPLE_LIMIT}.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.chat_file.exists():
        parser.error(f"File not found: {args.chat_file}")
    if not args.chat_file.is_file():
        parser.error(f"Not a file: {args.chat_file}")

    result = scan_chat_file(args.chat_file, sample_limit=args.sample_limit)
    print_report(result, top_users=args.top_users)


if __name__ == "__main__":
    main()
