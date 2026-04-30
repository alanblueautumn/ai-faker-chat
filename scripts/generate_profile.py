from __future__ import annotations

import argparse
import os
from pathlib import Path

from openai import OpenAI

from chatlib import load_dotenv, profile_filename, read_jsonl
from cleaning import filter_message_rows


DEFAULT_MESSAGES = Path("data/messages.jsonl")
DEFAULT_OUT_DIR = Path("data/profiles")
DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def collect_user_messages(
    messages_path: Path,
    user: str,
    limit: int,
    clean: bool,
    min_chars: int,
    global_repeat_threshold: int,
    per_user_repeat_limit: int,
) -> list[str]:
    rows = [message for message in read_jsonl(messages_path) if str(message["user"]) == user]
    if clean:
        # 用户画像样本要尽量信息密集；复读、空洞短句和图片占位会浪费
        # 模型上下文，也会让生成出来的风格档案变得很泛。
        rows, stats = filter_message_rows(
            rows,
            min_chars,
            global_repeat_threshold,
            per_user_repeat_limit,
        )
        print(f"Cleaning stats for {user}: {stats}")

    messages = [str(message["content"]) for message in rows]
    if len(messages) <= limit:
        return messages

    # 按用户完整发言历史均匀抽样，而不是只取最早的消息；
    # 这样昵称阶段、话题阶段变化不容易单独主导画像。
    step = len(messages) / limit
    return [messages[int(index * step)] for index in range(limit)]


def build_prompt(user: str, samples: list[str]) -> str:
    sample_text = "\n".join(f"- {sample}" for sample in samples)
    return f"""请基于下面的真实群聊发言，为用户“{user}”生成一份聊天风格档案。

要求：
- 使用中文
- 只总结说话风格，不泄露或放大隐私
- 不要编造这个人的真实身份、职业、住址、关系
- 重点描述语气、长短、常用词、标点、emoji、句式、接话习惯、禁忌
- 输出 Markdown，控制在 500 字以内

真实发言样本：
{sample_text}
"""


def build_client(provider: str) -> OpenAI:
    load_dotenv()
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")
        base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI()


def generate_profile(
    messages_path: Path,
    out_dir: Path,
    user: str,
    provider: str,
    model: str,
    limit: int,
    clean: bool,
    min_chars: int,
    global_repeat_threshold: int,
    per_user_repeat_limit: int,
) -> Path:
    samples = collect_user_messages(
        messages_path,
        user,
        limit,
        clean,
        min_chars,
        global_repeat_threshold,
        per_user_repeat_limit,
    )
    if not samples:
        raise ValueError(f"No messages found for user: {user}")

    client = build_client(provider)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(user, samples)}],
    )
    content = response.choices[0].message.content or ""

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / profile_filename(user)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a style profile for one user.")
    parser.add_argument("--user", required=True, help="Exact user name to profile.")
    parser.add_argument(
        "--messages",
        type=Path,
        default=DEFAULT_MESSAGES,
        help=f"Messages JSONL path. Default: {DEFAULT_MESSAGES}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Profile output directory. Default: {DEFAULT_OUT_DIR}",
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
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum message samples to send to the model. Default: 200.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Disable repeated/low-information content filtering before sampling.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=4,
        help="Drop messages shorter than this when cleaning. Default: 4.",
    )
    parser.add_argument(
        "--global-repeat-threshold",
        type=int,
        default=20,
        help="Drop exact messages repeated at least this many times. Default: 20.",
    )
    parser.add_argument(
        "--per-user-repeat-limit",
        type=int,
        default=3,
        help="Keep at most this many exact repeats. Default: 3.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.messages.exists():
        parser.error(f"File not found: {args.messages}")

    path = generate_profile(
        args.messages,
        args.out_dir,
        args.user,
        args.provider,
        args.model,
        args.limit,
        not args.no_clean,
        args.min_chars,
        args.global_repeat_threshold,
        args.per_user_repeat_limit,
    )
    print(f"Wrote profile to {path}")


if __name__ == "__main__":
    main()
