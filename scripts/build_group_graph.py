from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from chatlib import read_jsonl, write_json


DEFAULT_OUT = Path("data/reports/group_graph.json")
QUESTION_WORDS = ("?", "？")
AGREE_WORDS = ("是的", "确实", "对", "可以", "没错", "好")
DISAGREE_WORDS = ("不是", "不对", "没有", "别", "少", "不行")
TEASE_WORDS = ("草", "笑死", "傻逼", "jb", "离谱", "逆天", "你懂个", "牛逼")
EXPLAIN_WORDS = ("因为", "所以", "就是", "原理", "意思是", "这个")


def tone_of(content: str) -> str:
    lowered = content.lower()
    if any(word in content for word in QUESTION_WORDS):
        return "question"
    if any(word in content for word in TEASE_WORDS):
        return "tease"
    if any(word in content for word in DISAGREE_WORDS):
        return "disagree"
    if any(word in content for word in AGREE_WORDS):
        return "agree"
    if any(word in content for word in EXPLAIN_WORDS) or "why" in lowered:
        return "explain"
    return "neutral"


def user_id(user: str) -> str:
    return f"user:{user}"


def add_evidence(
    evidences: dict[tuple[str, str], list[dict[str, object]]],
    source: dict[str, object],
    target: dict[str, object],
    limit: int,
) -> None:
    key = (str(source["user"]), str(target["user"]))
    bucket = evidences[key]
    if len(bucket) >= limit:
        return
    bucket.append(
        {
            "source_line": source["line_no"],
            "source_time": source.get("time"),
            "source_message": source["content"],
            "target_line": target["line_no"],
            "target_time": target.get("time"),
            "target_message": target["content"],
        }
    )


def build_graph(
    messages_path: Path,
    reply_window: int,
    co_window: int,
    min_weight: int,
    evidence_limit: int,
) -> dict[str, object]:
    messages = list(read_jsonl(messages_path))
    user_counts = Counter(str(row["user"]) for row in messages)
    reply_counts: Counter[tuple[str, str]] = Counter()
    co_counts: Counter[tuple[str, str]] = Counter()
    tones: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    evidences: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)

    for index, message in enumerate(messages):
        source_user = str(message["user"])

        for previous in messages[max(0, index - reply_window) : index]:
            target_user = str(previous["user"])
            if target_user == source_user:
                continue
            key = (source_user, target_user)
            reply_counts[key] += 1
            tones[key][tone_of(str(message["content"]))] += 1
            add_evidence(evidences, message, previous, evidence_limit)

        window = messages[max(0, index - co_window) : index + 1]
        users_in_window = sorted({str(row["user"]) for row in window})
        for left, right in combinations(users_in_window, 2):
            co_counts[(left, right)] += 1

    pair_keys = {
        tuple(sorted((source, target)))
        for source, target in reply_counts
    } | set(co_counts)

    edges = []
    for left, right in sorted(pair_keys):
        left_to_right = reply_counts[(left, right)]
        right_to_left = reply_counts[(right, left)]
        co_occurrence = co_counts[(left, right)]
        tone_counter = tones[(left, right)] + tones[(right, left)]
        weight = left_to_right + right_to_left + co_occurrence
        if weight < min_weight:
            continue

        edges.append(
            {
                "source": user_id(left),
                "target": user_id(right),
                "type": "INTERACTS_WITH",
                "weight": weight,
                "metrics": {
                    "reply_source_to_target": left_to_right,
                    "reply_target_to_source": right_to_left,
                    "co_occurrence": co_occurrence,
                    "tones": dict(tone_counter),
                },
                "evidence": evidences.get((left, right), []) + evidences.get((right, left), []),
            }
        )

    nodes = [
        {
            "id": user_id(user),
            "type": "User",
            "name": user,
            "message_count": count,
        }
        for user, count in user_counts.most_common()
    ]

    edges.sort(key=lambda item: item["weight"], reverse=True)
    return {
        "meta": {
            "source": str(messages_path),
            "message_count": len(messages),
            "user_count": len(nodes),
            "reply_window": reply_window,
            "co_window": co_window,
            "min_weight": min_weight,
            "note": "Edges are inferred from message order, not explicit reply metadata.",
        },
        "nodes": nodes,
        "edges": edges,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a group member relation graph.")
    parser.add_argument("messages", type=Path, help="Path to data/messages.jsonl.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reply-window", type=int, default=3)
    parser.add_argument("--co-window", type=int, default=8)
    parser.add_argument("--min-weight", type=int, default=20)
    parser.add_argument("--evidence-limit", type=int, default=5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.messages.exists():
        parser.error(f"File not found: {args.messages}")

    graph = build_graph(
        args.messages,
        args.reply_window,
        args.co_window,
        args.min_weight,
        args.evidence_limit,
    )
    write_json(args.out, graph)
    print(f"Wrote graph with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges to {args.out}")


if __name__ == "__main__":
    main()
