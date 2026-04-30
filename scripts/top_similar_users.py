from __future__ import annotations

import argparse
from pathlib import Path

from chatlib import read_json


DEFAULT_REPORT = Path("data/reports/user_similarity.json")


def print_top_users(report_path: Path, user: str, top_k: int) -> None:
    report = read_json(report_path)
    users = report.get("users", {})

    if user not in users:
        available = sorted(users)
        raise ValueError(
            f"User not found: {user}. Available users: {', '.join(available[:30])}"
        )

    row = users[user]
    similar_users = row.get("similar_users", [])[:top_k]

    print(f"目标用户：{user}")
    print(f"消息数量：{row.get('messages')}")
    print(f"向量分块：{row.get('chunks')}")
    print()
    print(f"最相似的 {top_k} 个用户")
    print("---------------------")
    print("说明：score 越接近 1，表示整体发言内容和表达习惯越相似。")
    print("注意：这是“话题 + 表达习惯”的综合相似度，不等于纯性格或纯文风。")
    print()
    print(f"{'排名':>4}  {'相似度':>7}  {'消息数':>8}  用户")

    for rank, item in enumerate(similar_users, start=1):
        print(
            f"{rank:>4}  {float(item['score']):>7.4f}  "
            f"{int(item['messages']):>8}  {item['user']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print the most similar users for one target user."
    )
    parser.add_argument("--user", required=True, help="Exact target user name.")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Similarity report path. Default: {DEFAULT_REPORT}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many similar users to print. Default: 3.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.report.exists():
        parser.error(
            f"File not found: {args.report}. Run scripts/user_similarity.py first."
        )

    try:
        print_top_users(args.report, args.user, args.top_k)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
