from __future__ import annotations

import argparse
import json
from pathlib import Path

from chatlib import ensure_parent, read_json


DEFAULT_GRAPH = Path("data/reports/group_graph.json")
DEFAULT_OUT = Path("data/reports/group_graph.html")


def render_html(graph: dict[str, object], ai: dict[str, object] | None, max_edges: int) -> str:
    nodes = graph["nodes"]
    edges = graph["edges"][:max_edges]
    payload = json.dumps(
        {"nodes": nodes, "edges": edges, "meta": graph["meta"], "ai": ai or {}},
        ensure_ascii=False,
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>群友关系图谱</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f7f4; color: #202124; }}
    header {{ padding: 16px 20px; border-bottom: 1px solid #d8d8d0; background: #fff; }}
    h1 {{ margin: 0 0 6px; font-size: 20px; }}
    .meta {{ color: #666; font-size: 13px; }}
    main {{ display: grid; grid-template-columns: 1fr 380px; height: calc(100vh - 73px); }}
    svg {{ width: 100%; height: 100%; background: #fafaf8; }}
    aside {{ border-left: 1px solid #d8d8d0; background: #fff; padding: 14px; overflow: auto; }}
    .node {{ cursor: pointer; stroke: #fff; stroke-width: 1.5; }}
    .link {{ stroke: #8a8f98; stroke-opacity: .35; }}
    .label {{ font-size: 11px; pointer-events: none; fill: #333; }}
    .card {{ border: 1px solid #ddd; border-radius: 6px; padding: 10px; margin-bottom: 10px; }}
    .muted {{ color: #666; font-size: 12px; }}
    .row {{ margin: 6px 0; }}
    @media (max-width: 820px) {{ main {{ grid-template-columns: 1fr; }} aside {{ height: 260px; border-left: 0; border-top: 1px solid #d8d8d0; }} }}
  </style>
</head>
<body>
  <header>
    <h1>群友关系图谱</h1>
    <div class="meta" id="meta"></div>
  </header>
  <main>
    <svg id="graph"></svg>
    <aside>
      <div id="summary" class="card"></div>
      <div class="card">
        <strong>说明</strong>
        <div class="muted">连线基于消息顺序推断：近邻接话、窗口共现和规则语气统计，不代表明确回复关系。</div>
      </div>
      <div id="details" class="card">点击节点或连线查看详情。</div>
    </aside>
  </main>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script>
    const data = {payload};
    const svg = d3.select("#graph");
    const details = d3.select("#details");
    const ai = data.ai || {{}};
    document.getElementById("meta").textContent =
      `用户 ${{data.meta.user_count}} 个，消息 ${{data.meta.message_count}} 条，展示边 ${{data.edges.length}} 条`;
    document.getElementById("summary").innerHTML = renderSummary(ai.group_summary);

    const width = () => svg.node().clientWidth;
    const height = () => svg.node().clientHeight;
    const edgeScale = d3.scaleSqrt()
      .domain([1, d3.max(data.edges, d => d.weight) || 1])
      .range([1, 8]);
    const nodeScale = d3.scaleSqrt()
      .domain([1, d3.max(data.nodes, d => d.message_count) || 1])
      .range([5, 22]);

    const links = data.edges.map(d => ({{...d}}));
    const nodes = data.nodes.map(d => ({{...d}}));
    const nodeById = new Map(nodes.map(d => [d.id, d]));

    const link = svg.append("g")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("class", "link")
      .attr("stroke-width", d => edgeScale(d.weight))
      .on("click", (_, d) => showEdge(d));

    const node = svg.append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("class", "node")
      .attr("r", d => nodeScale(d.message_count))
      .attr("fill", "#3f7f93")
      .on("click", (_, d) => showNode(d))
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    const label = svg.append("g")
      .selectAll("text")
      .data(nodes)
      .join("text")
      .attr("class", "label")
      .text(d => d.name);

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(d => 120 - Math.min(80, d.weight)))
      .force("charge", d3.forceManyBody().strength(-220))
      .force("center", d3.forceCenter(width() / 2, height() / 2))
      .force("collision", d3.forceCollide().radius(d => nodeScale(d.message_count) + 18));

    simulation.on("tick", () => {{
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
      node.attr("cx", d => d.x).attr("cy", d => d.y);
      label.attr("x", d => d.x + 8).attr("y", d => d.y + 4);
    }});

    function showNode(d) {{
      const userAi = (ai.users || {{}})[d.id] || {{}};
      const connected = links
        .filter(e => e.source.id === d.id || e.target.id === d.id)
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 12)
        .map(e => {{
          const other = e.source.id === d.id ? e.target : e.source;
          return `<div class="row">${{other.name}}：连接强度 ${{e.weight}}</div>`;
        }}).join("");
      details.html(`<strong>${{d.name}}</strong>
        <div class="muted">消息数：${{d.message_count}}</div>
        ${{userAi.role ? `<div class="row">角色：${{userAi.role}}</div>` : ""}}
        ${{userAi.style ? `<div class="row">风格：${{userAi.style}}</div>` : ""}}
        ${{userAi.active_pattern ? `<div class="row">互动模式：${{userAi.active_pattern}}</div>` : ""}}
        ${{userAi.classic_quotes ? `<div class="row">典型表达：${{userAi.classic_quotes.join(" / ")}}</div>` : ""}}
        <hr>
        <strong>主要连接</strong>${{connected || "<div class='muted'>无</div>"}}`);
    }}

    function showEdge(d) {{
      const relationKey = `${{d.source.id}}|${{d.target.id}}`;
      const reverseKey = `${{d.target.id}}|${{d.source.id}}`;
      const relationAi = (ai.relations || {{}})[relationKey] || (ai.relations || {{}})[reverseKey] || {{}};
      const tones = d.metrics.tones || {{}};
      const toneText = Object.entries(tones).sort((a, b) => b[1] - a[1])
        .map(([k, v]) => `${{k}}: ${{v}}`).join("，");
      const evidence = (d.evidence || []).slice(0, 5).map(e =>
        `<div class="row muted">${{e.source_time || ""}} ${{nodeById.get(d.source.id || d.source).name}}：${{e.source_message}}<br>接近 ${{e.target_time || ""}} ${{nodeById.get(d.target.id || d.target).name}}：${{e.target_message}}</div>`
      ).join("");
      details.html(`<strong>${{d.source.name}} ↔ ${{d.target.name}}</strong>
        ${{relationAi.type ? `<div class="row">关系类型：${{relationAi.type}}</div>` : ""}}
        ${{relationAi.summary ? `<div class="row">AI 解释：${{relationAi.summary}}</div>` : ""}}
        ${{relationAi.confidence !== undefined ? `<div class="row muted">置信度：${{relationAi.confidence}}</div>` : ""}}
        <div class="row">连接强度：${{d.weight}}</div>
        <div class="row">共现：${{d.metrics.co_occurrence}}</div>
        <div class="row">疑似接话：${{d.metrics.reply_source_to_target}} / ${{d.metrics.reply_target_to_source}}</div>
        <div class="row">语气：${{toneText || "无"}}</div>
        <hr><strong>证据</strong>${{evidence || "<div class='muted'>无</div>"}}`);
    }}

    function renderSummary(summary) {{
      if (!summary) return "<strong>全群摘要</strong><div class='muted'>未提供 AI 增强摘要。</div>";
      const core = (summary.core_users || []).map(item =>
        `<div class="row">${{item.user}}：${{item.role}} <span class="muted">${{item.reason || ""}}</span></div>`
      ).join("");
      const clusters = (summary.clusters || []).map(item =>
        `<div class="row">${{item.name}}：${{(item.users || []).join("、")}}<br><span class="muted">${{item.description || ""}}</span></div>`
      ).join("");
      return `<strong>全群摘要</strong>
        <div class="row">${{summary.summary || ""}}</div>
        ${{core ? `<hr><strong>核心成员</strong>${{core}}` : ""}}
        ${{clusters ? `<hr><strong>圈层</strong>${{clusters}}` : ""}}`;
    }}

    function dragstarted(event, d) {{
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    }}
    function dragged(event, d) {{ d.fx = event.x; d.fy = event.y; }}
    function dragended(event, d) {{
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null; d.fy = null;
    }}
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render group graph JSON to HTML.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--ai", type=Path, default=None, help="Optional AI enrichment JSON path.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-edges", type=int, default=300)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.graph.exists():
        parser.error(f"File not found: {args.graph}")

    ai = read_json(args.ai) if args.ai and args.ai.exists() else None
    html = render_html(read_json(args.graph), ai, args.max_edges)
    ensure_parent(args.out)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote HTML graph to {args.out}")


if __name__ == "__main__":
    main()
