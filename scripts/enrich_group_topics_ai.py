from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from chatlib import load_dotenv, read_json, write_json
from generate_profile import DEFAULT_DEEPSEEK_BASE_URL


DEFAULT_TOPICS = Path("data/reports/group_topics_base.json")
DEFAULT_OUT = Path("data/reports/group_topics_ai.json")
DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-v4-pro"


def build_client(provider: str) -> OpenAI:
    load_dotenv()
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")
        base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI()


def compact_topics(topics: dict[str, object], examples_per_topic: int) -> dict[str, object]:
    compact = []
    for topic in topics["topics"]:
        compact.append(
            {
                "id": topic["id"],
                "size": topic["size"],
                "keywords": topic["keywords"],
                "top_users": topic["top_users"][:8],
                "examples": topic["examples"][:examples_per_topic],
            }
        )
    return {"meta": topics["meta"], "topics": compact}


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def build_prompt(payload: dict[str, object]) -> str:
    return f"""你要基于群聊话题聚类数据，生成 HTML 展示用的话题解释层。

要求：
- 只基于给定数据，不要编造现实身份和现实关系
- 输出严格 JSON，不要 Markdown，不要代码块
- 每个 topic 都要给 name、summary、style、keywords、representative_users
- name 要短，适合做卡片标题
- summary 控制在 100 字以内

输出结构：
{{
  "summary": "全群常见话题总体概括",
  "topics": {{
    "topic:0": {{
      "name": "...",
      "summary": "...",
      "style": "这个话题下的聊天方式",
      "keywords": ["..."],
      "representative_users": ["..."]
    }}
  }}
}}

话题数据：
{json.dumps(payload, ensure_ascii=False)}
"""


def enrich_topics(
    topics_path: Path,
    out: Path,
    provider: str,
    model: str,
    examples_per_topic: int,
) -> Path:
    topics = read_json(topics_path)
    prompt = build_prompt(compact_topics(topics, examples_per_topic))
    client = build_client(provider)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    data = json.loads(strip_json_fence(response.choices[0].message.content or ""))
    write_json(out, data)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate AI explanations for group topics.")
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--provider", choices=["deepseek", "openai"], default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--examples-per-topic", type=int, default=5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.topics.exists():
        parser.error(f"File not found: {args.topics}")

    path = enrich_topics(args.topics, args.out, args.provider, args.model, args.examples_per_topic)
    print(f"Wrote AI topic enrichment to {path}")


if __name__ == "__main__":
    main()
