from __future__ import annotations

import argparse
import json
from pathlib import Path

from chatlib import ensure_parent, read_json


DEFAULT_TOPICS = Path("data/reports/group_topics_base.json")
DEFAULT_AI = Path("data/reports/group_topics_ai.json")
DEFAULT_OUT = Path("data/reports/group_topics.html")


def render_html(topics: dict[str, object], ai: dict[str, object] | None) -> str:
    payload = json.dumps({"base": topics, "ai": ai or {}}, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>群聊常见话题</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f7f4; color: #222; }}
    header {{ padding: 18px 22px; background: #fff; border-bottom: 1px solid #ddd; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .summary {{ color: #555; max-width: 980px; line-height: 1.5; }}
    main {{ padding: 18px; display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 14px; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 14px; }}
    .title {{ font-weight: 700; font-size: 17px; margin-bottom: 6px; }}
    .meta {{ color: #666; font-size: 12px; margin-bottom: 8px; }}
    .tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }}
    .tag {{ background: #eef3f4; border: 1px solid #d8e2e4; padding: 2px 7px; border-radius: 999px; font-size: 12px; }}
    .example {{ border-left: 3px solid #c7d6d9; padding-left: 8px; color: #444; font-size: 13px; white-space: pre-wrap; max-height: 180px; overflow: auto; }}
    .section {{ margin-top: 10px; }}
    .section strong {{ font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <h1>群聊常见话题</h1>
    <div id="summary" class="summary"></div>
  </header>
  <main id="topics"></main>
  <script>
    const data = {payload};
    const aiTopics = (data.ai && data.ai.topics) || {{}};
    document.getElementById("summary").textContent =
      (data.ai && data.ai.summary) || `共 ${{data.base.topics.length}} 个话题，基于 ${{data.base.meta.documents}} 个上下文窗口聚类。`;

    const root = document.getElementById("topics");
    for (const topic of data.base.topics) {{
      const ai = aiTopics[topic.id] || {{}};
      const name = ai.name || topic.id;
      const summary = ai.summary || "";
      const style = ai.style || "";
      const keywords = ai.keywords || topic.keywords || [];
      const users = ai.representative_users || (topic.top_users || []).map(x => x.user);
      const example = (topic.examples || [])[0];
      const div = document.createElement("section");
      div.className = "card";
      div.innerHTML = `
        <div class="title">${{name}}</div>
        <div class="meta">窗口数：${{topic.size}} ｜ ID：${{topic.id}}</div>
        ${{summary ? `<div>${{summary}}</div>` : ""}}
        ${{style ? `<div class="section"><strong>聊天方式</strong><br>${{style}}</div>` : ""}}
        <div class="section"><strong>关键词</strong><div class="tags">${{keywords.slice(0, 16).map(k => `<span class="tag">${{k}}</span>`).join("")}}</div></div>
        <div class="section"><strong>代表用户</strong><div class="tags">${{users.slice(0, 10).map(u => `<span class="tag">${{u}}</span>`).join("")}}</div></div>
        ${{example ? `<div class="section"><strong>代表窗口</strong><div class="example">${{example.text}}</div></div>` : ""}}
      `;
      root.appendChild(div);
    }}
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render group topic analysis to HTML.")
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS)
    parser.add_argument("--ai", type=Path, default=DEFAULT_AI)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.topics.exists():
        parser.error(f"File not found: {args.topics}")

    ai = read_json(args.ai) if args.ai.exists() else None
    html = render_html(read_json(args.topics), ai)
    ensure_parent(args.out)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote topic HTML to {args.out}")


if __name__ == "__main__":
    main()
