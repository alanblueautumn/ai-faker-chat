from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

from chatlib import load_dotenv, read_json, read_jsonl, write_json
from generate_profile import DEFAULT_DEEPSEEK_BASE_URL


DEFAULT_GRAPH = Path("data/reports/group_graph.json")
DEFAULT_MESSAGES = Path("data/messages.jsonl")
DEFAULT_OUT = Path("data/reports/group_graph_ai.json")
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


def user_name_from_id(node_id: str) -> str:
    return node_id.removeprefix("user:")


def collect_user_samples(messages_path: Path, users: set[str], limit: int) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(messages_path):
        user = str(row["user"])
        if user not in users:
            continue
        bucket = buckets[user]
        if len(bucket) >= limit:
            continue
        bucket.append(f"{row.get('time', '')} {user}:{row['content']}".strip())
    return dict(buckets)


def build_prompt(
    graph: dict[str, object],
    messages_path: Path,
    top_users: int,
    top_edges: int,
    samples_per_user: int,
    evidence_per_edge: int,
) -> str:
    nodes = graph["nodes"][:top_users]
    edges = graph["edges"][:top_edges]
    users = {str(node["name"]) for node in nodes}
    for edge in edges:
        users.add(user_name_from_id(str(edge["source"])))
        users.add(user_name_from_id(str(edge["target"])))

    user_samples = collect_user_samples(messages_path, users, samples_per_user)
    compact_nodes = [
        {
            "id": node["id"],
            "name": node["name"],
            "message_count": node["message_count"],
            "samples": user_samples.get(str(node["name"]), []),
        }
        for node in nodes
    ]
    compact_edges = []
    for edge in edges:
        compact_edges.append(
            {
                "source": edge["source"],
                "target": edge["target"],
                "weight": edge["weight"],
                "metrics": edge["metrics"],
                "evidence": edge.get("evidence", [])[:evidence_per_edge],
            }
        )

    payload = {
        "meta": graph["meta"],
        "top_users": compact_nodes,
        "top_edges": compact_edges,
    }

    return f"""你要基于群聊关系图谱数据，生成 HTML 可视化使用的 AI 解释层。

要求：
- 只基于给定 JSON 数据，不要编造现实身份、现实关系、住址、职业
- 关系都是基于消息顺序推断，只能描述为“疑似互动、共现、接话模式”
- 输出必须是严格 JSON，不要 Markdown，不要代码块
- key 必须包含：group_summary、users、relations
- users 的 key 使用节点 id，例如 user:zzZ
- relations 的 key 使用 source|target，例如 user:zzZ|user:安达桜
- 每个用户解释控制在 80 字以内
- 每条关系解释控制在 120 字以内

输出 JSON 结构：
{{
  "group_summary": {{
    "summary": "...",
    "core_users": [{{"user": "...", "role": "...", "reason": "..."}}],
    "clusters": [{{"name": "...", "users": ["..."], "description": "..."}}]
  }},
  "users": {{
    "user:xxx": {{
      "role": "...",
      "style": "...",
      "active_pattern": "...",
      "classic_quotes": ["..."]
    }}
  }},
  "relations": {{
    "user:a|user:b": {{
      "type": "...",
      "summary": "...",
      "confidence": 0.0
    }}
  }}
}}

图谱数据：
{json.dumps(payload, ensure_ascii=False)}
"""


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


def enrich_graph(
    graph_path: Path,
    messages_path: Path,
    out: Path,
    provider: str,
    model: str,
    top_users: int,
    top_edges: int,
    samples_per_user: int,
    evidence_per_edge: int,
) -> Path:
    graph = read_json(graph_path)
    prompt = build_prompt(
        graph,
        messages_path,
        top_users,
        top_edges,
        samples_per_user,
        evidence_per_edge,
    )
    client = build_client(provider)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = strip_json_fence(response.choices[0].message.content or "")
    data = json.loads(content)
    write_json(out, data)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate AI explanations for group graph HTML.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--messages", type=Path, default=DEFAULT_MESSAGES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--provider", choices=["deepseek", "openai"], default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-users", type=int, default=20)
    parser.add_argument("--top-edges", type=int, default=50)
    parser.add_argument("--samples-per-user", type=int, default=8)
    parser.add_argument("--evidence-per-edge", type=int, default=5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.graph.exists():
        parser.error(f"File not found: {args.graph}")
    if not args.messages.exists():
        parser.error(f"File not found: {args.messages}")

    path = enrich_graph(
        args.graph,
        args.messages,
        args.out,
        args.provider,
        args.model,
        args.top_users,
        args.top_edges,
        args.samples_per_user,
        args.evidence_per_edge,
    )
    print(f"Wrote AI graph enrichment to {path}")


if __name__ == "__main__":
    main()
