from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from pathlib import Path

from openai import OpenAI

from chatlib import load_dotenv, profile_filename, read_jsonl
from cleaning import EMOJI_RE, filter_message_rows, normalize_content
from generate_profile import DEFAULT_DEEPSEEK_BASE_URL


DEFAULT_MESSAGES = Path("data/messages.jsonl")
DEFAULT_WINDOWS = Path("data/windows.jsonl")
DEFAULT_OUT_DIR = Path("data/profiles")
DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-v4-pro"
PUNCTUATION_RE = re.compile(r"[!?！？。.,，、~～…]+")


def build_client(provider: str) -> OpenAI:
    load_dotenv()
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")
        base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI()


def evenly_sample(items: list[str], limit: int) -> list[str]:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(index * step)] for index in range(limit)]


def top_phrases(messages: list[str], limit: int) -> list[tuple[str, int]]:
    counter = Counter(messages)
    return [
        (phrase, count)
        for phrase, count in counter.most_common(limit * 3)
        if count > 1
    ][:limit]


def build_stats(rows: list[dict[str, object]]) -> dict[str, object]:
    contents = [str(row["content"]) for row in rows]
    lengths = [len(content) for content in contents]
    punctuation = Counter()
    emoji = Counter()
    active_hours = Counter()

    for row, content in zip(rows, contents):
        punctuation.update(PUNCTUATION_RE.findall(content))
        emoji.update(EMOJI_RE.findall(content))
        message_time = str(row.get("time", ""))
        if len(message_time) >= 2:
            active_hours[message_time[:2]] += 1

    total = len(contents)
    return {
        "messages": total,
        "chars": sum(lengths),
        "avg_length": round(sum(lengths) / total, 2) if total else 0,
        "short_ratio": round(sum(1 for length in lengths if length <= 8) / total, 4)
        if total
        else 0,
        "long_ratio": round(sum(1 for length in lengths if length >= 40) / total, 4)
        if total
        else 0,
        "top_punctuation": punctuation.most_common(20),
        "top_emoji": emoji.most_common(20),
        "active_hours": active_hours.most_common(8),
        "top_repeated_phrases": top_phrases(contents, 30),
    }


def collect_windows(windows_path: Path, user: str, limit: int) -> list[str]:
    if not windows_path.exists():
        return []

    windows = [
        str(window["full_text"])
        for window in read_jsonl(windows_path)
        if str(window["user"]) == user
    ]
    return evenly_sample(windows, limit)


def collect_profile_data(
    messages_path: Path,
    windows_path: Path,
    user: str,
    sample_limit: int,
    window_limit: int,
    min_chars: int,
    global_repeat_threshold: int,
    per_user_repeat_limit: int,
) -> dict[str, object]:
    raw_rows = [row for row in read_jsonl(messages_path) if str(row["user"]) == user]
    if not raw_rows:
        raise ValueError(f"No messages found for user: {user}")

    cleaned_rows, clean_stats = filter_message_rows(
        raw_rows,
        min_chars,
        global_repeat_threshold,
        per_user_repeat_limit,
    )
    samples = evenly_sample(
        [normalize_content(str(row["content"])) for row in cleaned_rows],
        sample_limit,
    )

    return {
        "user": user,
        "raw_count": len(raw_rows),
        "clean_stats": clean_stats,
        "stats": build_stats(cleaned_rows),
        "samples": samples,
        "windows": collect_windows(windows_path, user, window_limit),
    }


def build_prompt(data: dict[str, object]) -> str:
    stats = data["stats"]
    repeated = "\n".join(
        f"- {phrase}: {count} 次" for phrase, count in stats["top_repeated_phrases"]
    )
    samples = "\n".join(f"- {sample}" for sample in data["samples"])
    windows = "\n\n".join(str(window) for window in data["windows"])

    return f"""请基于下面的历史数据，为用户“{data["user"]}”生成一份尽可能贴切的聊天风格画像。

要求：
- 使用中文 Markdown
- 只基于提供的数据，不要编造真实身份、职业、住址、现实关系
- 重点描述“这个人怎么说话”，而不是泛泛总结群聊话题
- 要区分：常态表达、怼人方式、接梗方式、解释方式、敷衍方式
- 引用真实原句作为证据，但不要大段复制
- 明确写出“不像他的表达”，方便后续 AI 避免生成跑偏
- 最后给一段“模仿指令”，可直接放进 prompt
- 输出结构要简洁，不要写太多小标题

基础统计：
- 原始消息数：{data["raw_count"]}
- 清洗统计：{data["clean_stats"]}
- 清洗后消息数：{stats["messages"]}
- 总字符数：{stats["chars"]}
- 平均长度：{stats["avg_length"]}
- 短句比例：{stats["short_ratio"]}
- 长句比例：{stats["long_ratio"]}
- 高频标点：{stats["top_punctuation"]}
- 高频 emoji：{stats["top_emoji"]}
- 活跃小时 Top：{stats["active_hours"]}

高频重复表达：
{repeated}

清洗后代表发言：
{samples}

代表上下文窗口：
{windows}

请按以下结构输出：

# 用户聊天风格画像

## 简要概括
## 说话习惯
## 典型表达
## 不要这样生成
## 模仿指令
"""


def output_path(out_dir: Path, user: str) -> Path:
    base = profile_filename(user)
    return out_dir / base.replace(".md", ".deep.md")


def generate_deep_profile(
    messages_path: Path,
    windows_path: Path,
    out_dir: Path,
    user: str,
    provider: str,
    model: str,
    sample_limit: int,
    window_limit: int,
    min_chars: int,
    global_repeat_threshold: int,
    per_user_repeat_limit: int,
) -> Path:
    data = collect_profile_data(
        messages_path,
        windows_path,
        user,
        sample_limit,
        window_limit,
        min_chars,
        global_repeat_threshold,
        per_user_repeat_limit,
    )
    print(f"Cleaning stats for {user}: {data['clean_stats']}")
    print(f"Sending {len(data['samples'])} samples and {len(data['windows'])} windows to AI")

    client = build_client(provider)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(data)}],
    )
    content = response.choices[0].message.content or ""

    out_dir.mkdir(parents=True, exist_ok=True)
    path = output_path(out_dir, user)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a high-precision user style profile.")
    parser.add_argument("--user", required=True, help="Exact user name.")
    parser.add_argument("--messages", type=Path, default=DEFAULT_MESSAGES)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--provider", choices=["deepseek", "openai"], default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample-limit", type=int, default=500)
    parser.add_argument("--window-limit", type=int, default=80)
    parser.add_argument("--min-chars", type=int, default=4)
    parser.add_argument("--global-repeat-threshold", type=int, default=20)
    parser.add_argument("--per-user-repeat-limit", type=int, default=3)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.messages.exists():
        parser.error(f"File not found: {args.messages}")

    try:
        path = generate_deep_profile(
            args.messages,
            args.windows,
            args.out_dir,
            args.user,
            args.provider,
            args.model,
            args.sample_limit,
            args.window_limit,
            args.min_chars,
            args.global_repeat_threshold,
            args.per_user_repeat_limit,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Wrote deep profile to {path}")


if __name__ == "__main__":
    main()
