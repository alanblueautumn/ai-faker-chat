from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

from openai import OpenAI

from chatlib import load_dotenv, read_jsonl
from cleaning import is_low_information, normalize_content
from generate_profile import DEFAULT_DEEPSEEK_BASE_URL


DEFAULT_MESSAGES = Path("data/messages.jsonl")
DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-v4-pro"
STRONG_WORDS = (
    "草",
    "卧槽",
    "牛逼",
    "傻逼",
    "jb",
    "离谱",
    "笑死",
    "绷",
    "逆天",
    "你懂",
    "你会",
    "随便",
)


def build_client(provider: str) -> OpenAI:
    load_dotenv()
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")
        base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI()


def quote_score(content: str, count: int) -> float:
    score = 0.0
    length = len(content)
    score += min(length, 80) / 8
    score += min(count, 10) * 2
    score += 4 if "?" in content or "？" in content else 0
    score += 2 if "!" in content or "！" in content else 0
    score += sum(3 for word in STRONG_WORDS if word.lower() in content.lower())
    if 6 <= length <= 60:
        score += 6
    if length > 120:
        score -= 10
    return score


def format_context(rows: list[dict[str, object]], index: int, before: int, after: int) -> str:
    start = max(0, index - before)
    end = min(len(rows), index + after + 1)
    return "\n".join(
        f"{row.get('time', '')} {row['user']}:{row['content']}".strip()
        for row in rows[start:end]
    )


def collect_candidates(
    messages_path: Path,
    user: str,
    min_chars: int,
    max_candidates: int,
    before: int,
    after: int,
) -> list[dict[str, object]]:
    rows = list(read_jsonl(messages_path))
    user_indexes = [index for index, row in enumerate(rows) if str(row["user"]) == user]
    if not user_indexes:
        raise ValueError(f"No messages found for user: {user}")

    counter = Counter()
    first_context: dict[str, str] = {}
    first_line_no: dict[str, int] = {}

    for index in user_indexes:
        row = rows[index]
        content = normalize_content(str(row["content"]))
        if is_low_information(content, min_chars):
            continue

        counter[content] += 1
        if content not in first_context:
            first_context[content] = format_context(rows, index, before, after)
            first_line_no[content] = int(row["line_no"])

    scored = []
    for content, count in counter.items():
        scored.append(
            {
                "content": content,
                "count": count,
                "score": round(quote_score(content, count), 2),
                "line_no": first_line_no[content],
                "context": first_context[content],
            }
        )
    scored.sort(key=lambda item: (float(item["score"]), int(item["count"])), reverse=True)
    return scored[:max_candidates]


def build_prompt(user: str, candidates: list[dict[str, object]], top_k: int) -> str:
    candidate_text = "\n".join(
        f"{index}. 出现{item['count']}次｜规则分{item['score']}｜首次行号{item['line_no']}\n"
        f"原句：{item['content']}\n"
        f"上下文：\n{item['context']}"
        for index, item in enumerate(candidates, start=1)
    )
    return f"""你要从候选发言中选出用户“{user}”最有代表性的经典语录。

要求：
- 使用中文
- 只从候选发言里选择，不要改写原句，不要编造
- 选择 {top_k} 条
- 优先考虑：辨识度、口头禅、语气强、反问/吐槽、上下文里有群聊记忆点
- 不要选择纯低俗、纯无意义、纯复读、纯图片/表情
- 输出 Markdown
- 只输出最终最经典的 {top_k} 句，不要输出候选分析过程
- 每条必须包含：排名、原句、出自聊天行号、为什么经典、出现次数
- “出自聊天行号”必须使用候选里的“首次行号”
- 如果某句靠上下文才显得经典，请说明上下文触发点

候选发言：
{candidate_text}
"""


def generate_quotes(
    messages_path: Path,
    user: str,
    provider: str,
    model: str,
    top_k: int,
    min_chars: int,
    max_candidates: int,
    before: int,
    after: int,
) -> str:
    candidates = collect_candidates(
        messages_path,
        user,
        min_chars,
        max_candidates,
        before,
        after,
    )
    client = build_client(provider)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(user, candidates, top_k)}],
    )
    return (response.choices[0].message.content or "").strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use AI to pick classic quotes for a user.")
    parser.add_argument("--user", required=True, help="Exact user name.")
    parser.add_argument(
        "--messages",
        type=Path,
        default=DEFAULT_MESSAGES,
        help=f"Messages JSONL path. Default: {DEFAULT_MESSAGES}",
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
    parser.add_argument("--top-k", type=int, default=5, help="Quotes to select.")
    parser.add_argument("--min-chars", type=int, default=4, help="Min quote length.")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=300,
        help="Candidate quotes sent to AI. Default: 300.",
    )
    parser.add_argument(
        "--before",
        type=int,
        default=3,
        help="Context messages before the quote. Default: 3.",
    )
    parser.add_argument(
        "--after",
        type=int,
        default=2,
        help="Context messages after the quote. Default: 2.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.messages.exists():
        parser.error(f"File not found: {args.messages}")

    try:
        output = generate_quotes(
            args.messages,
            args.user,
            args.provider,
            args.model,
            args.top_k,
            args.min_chars,
            args.max_candidates,
            args.before,
            args.after,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(output)


if __name__ == "__main__":
    main()
