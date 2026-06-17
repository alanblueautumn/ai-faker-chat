from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from openai import OpenAI

from chatlib import load_dotenv, read_jsonl
from cleaning import filter_message_rows
from generate_profile import DEFAULT_DEEPSEEK_BASE_URL


DEFAULT_MESSAGES = Path("data/messages.jsonl")
DEFAULT_OUT = Path("data/profiles/group.md")
DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-v4-pro"


def build_client(provider: str) -> OpenAI:
    load_dotenv()
    if provider == "deepseek":
        import os

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")
        base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI()


def collect_group_data(
    messages_path: Path,
    min_chars: int,
    global_repeat_threshold: int,
    per_user_repeat_limit: int,
    max_users: int,
    samples_per_user: int,
    max_windows: int,
) -> dict[str, object]:
    rows = list(read_jsonl(messages_path))
    filtered_rows, clean_stats = filter_message_rows(
        rows,
        min_chars,
        global_repeat_threshold,
        per_user_repeat_limit,
    )

    user_counts: Counter[str] = Counter(str(row["user"]) for row in filtered_rows)
    content_lengths = [len(str(row["content"])) for row in filtered_rows]
    short_count = sum(1 for length in content_lengths if length <= 8)
    long_count = sum(1 for length in content_lengths if length >= 40)
    repeated_phrases = Counter(str(row["content"]).strip() for row in rows)

    by_user: dict[str, list[str]] = defaultdict(list)
    for row in filtered_rows:
        by_user[str(row["user"])].append(str(row["content"]))

    top_users = [user for user, _ in user_counts.most_common(max_users)]
    user_samples: dict[str, list[str]] = {}
    for user in top_users:
        messages = by_user[user]
        if len(messages) <= samples_per_user:
            user_samples[user] = messages
            continue

        step = len(messages) / samples_per_user
        user_samples[user] = [
            messages[int(index * step)] for index in range(samples_per_user)
        ]

    window_samples = []
    if filtered_rows:
        step = max(len(filtered_rows) // max(max_windows, 1), 1)
        for start in range(0, len(filtered_rows), step):
            window = filtered_rows[start : start + 6]
            if len(window) < 3:
                continue
            window_samples.append(
                "\n".join(
                    f"{row.get('time', '')} {row['user']}:{row['content']}".strip()
                    for row in window
                )
            )
            if len(window_samples) >= max_windows:
                break

    return {
        "total_messages": len(rows),
        "clean_stats": clean_stats,
        "filtered_messages": len(filtered_rows),
        "total_users": len(user_counts),
        "top_users": user_counts.most_common(max_users),
        "avg_length": round(sum(content_lengths) / len(content_lengths), 2)
        if content_lengths
        else 0,
        "short_message_ratio": round(short_count / len(content_lengths), 4)
        if content_lengths
        else 0,
        "long_message_ratio": round(long_count / len(content_lengths), 4)
        if content_lengths
        else 0,
        "top_repeated_phrases": [
            item for item in repeated_phrases.most_common(30) if item[1] >= 5
        ],
        "user_samples": user_samples,
        "window_samples": window_samples,
    }


def build_prompt(data: dict[str, object]) -> str:
    top_users = "\n".join(
        f"- {user}: {count} 条" for user, count in data["top_users"]
    )
    repeated = "\n".join(
        f"- {phrase}: {count} 次"
        for phrase, count in data["top_repeated_phrases"]
    )
    user_samples = "\n\n".join(
        f"## {user}\n" + "\n".join(f"- {message}" for message in messages)
        for user, messages in data["user_samples"].items()
    )
    window_samples = "\n\n".join(str(item) for item in data["window_samples"])

    return f"""请基于下面的群聊统计和代表样本，生成一份“群聊整体风格档案”。

要求：
- 使用中文 Markdown
- 不要编造群成员的真实身份、住址、联系方式或现实关系
- 重点总结这个群整体怎么聊天，而不是某一个人
- 包含整体语气、常见话题、常见表达、接话节奏、互怼/玩笑方式、复读习惯、禁忌
- 给后续 AI 生成回复使用，所以请写成可执行的风格约束
- 控制在 800 字以内

群聊统计：
- 原始消息数：{data["total_messages"]}
- 清洗后消息数：{data["filtered_messages"]}
- 用户数：{data["total_users"]}
- 平均消息长度：{data["avg_length"]}
- 短消息比例：{data["short_message_ratio"]}
- 长消息比例：{data["long_message_ratio"]}
- 清洗统计：{data["clean_stats"]}

高频用户：
{top_users}

高频复读/短句：
{repeated}

高频用户代表发言：
{user_samples}

代表对话窗口：
{window_samples}
"""


def generate_group_profile(
    messages_path: Path,
    out: Path,
    provider: str,
    model: str,
    min_chars: int,
    global_repeat_threshold: int,
    per_user_repeat_limit: int,
    max_users: int,
    samples_per_user: int,
    max_windows: int,
) -> Path:
    data = collect_group_data(
        messages_path,
        min_chars,
        global_repeat_threshold,
        per_user_repeat_limit,
        max_users,
        samples_per_user,
        max_windows,
    )
    client = build_client(provider)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(data)}],
    )
    content = response.choices[0].message.content or ""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content.strip() + "\n", encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a group style profile.")
    parser.add_argument(
        "--messages",
        type=Path,
        default=DEFAULT_MESSAGES,
        help=f"Messages JSONL path. Default: {DEFAULT_MESSAGES}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output Markdown path. Default: {DEFAULT_OUT}",
    )
    parser.add_argument(
        "--provider",
        choices=["deepseek", "openai"],
        default=DEFAULT_PROVIDER,
        help=f"Chat provider. Default: {DEFAULT_PROVIDER}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Generation model. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument("--min-chars", type=int, default=4)
    parser.add_argument("--global-repeat-threshold", type=int, default=20)
    parser.add_argument("--per-user-repeat-limit", type=int, default=3)
    parser.add_argument("--max-users", type=int, default=12)
    parser.add_argument("--samples-per-user", type=int, default=12)
    parser.add_argument("--max-windows", type=int, default=16)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.messages.exists():
        parser.error(f"File not found: {args.messages}")

    path = generate_group_profile(
        args.messages,
        args.out,
        args.provider,
        args.model,
        args.min_chars,
        args.global_repeat_threshold,
        args.per_user_repeat_limit,
        args.max_users,
        args.samples_per_user,
        args.max_windows,
    )
    print(f"Wrote group profile to {path}")


if __name__ == "__main__":
    main()
