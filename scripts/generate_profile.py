from __future__ import annotations

import argparse
import os
from pathlib import Path

from openai import OpenAI

from chatlib import load_dotenv, profile_filename, read_jsonl


DEFAULT_MESSAGES = Path("data/messages.jsonl")
DEFAULT_OUT_DIR = Path("data/profiles")
DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def collect_user_messages(messages_path: Path, user: str, limit: int) -> list[str]:
    messages = [
        str(message["content"])
        for message in read_jsonl(messages_path)
        if str(message["user"]) == user
    ]
    if len(messages) <= limit:
        return messages

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
) -> Path:
    samples = collect_user_messages(messages_path, user, limit)
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
    )
    print(f"Wrote profile to {path}")


if __name__ == "__main__":
    main()
