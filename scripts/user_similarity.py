from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from chatlib import read_jsonl, write_json
from cleaning import filter_message_rows
from embeddinglib import DEFAULT_LOCAL_EMBEDDING_MODEL, embed_local


DEFAULT_OUT = Path("data/reports/user_similarity.json")


def chunk_messages(messages: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0

    for message in messages:
        message = message.strip()
        if not message:
            continue

        message_chars = len(message)
        if current and current_chars + message_chars + 1 > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_chars = 0

        if message_chars > max_chars:
            chunks.append(message[:max_chars])
            continue

        current.append(message)
        current_chars += message_chars + 1

    if current:
        chunks.append("\n".join(current))

    return chunks


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def build_user_vectors(
    messages_path: Path,
    model: str,
    min_messages: int,
    max_chunk_chars: int,
    batch_size: int,
    clean: bool,
    min_chars: int,
    global_repeat_threshold: int,
    per_user_repeat_limit: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    by_user: dict[str, list[str]] = defaultdict(list)
    rows = list(read_jsonl(messages_path))
    clean_stats = None
    if clean:
        rows, clean_stats = filter_message_rows(
            rows,
            min_chars,
            global_repeat_threshold,
            per_user_repeat_limit,
        )
        print(f"Cleaning stats: {clean_stats}")

    for message in rows:
        by_user[str(message["user"])].append(str(message["content"]))

    user_vectors: dict[str, np.ndarray] = {}
    user_meta: dict[str, dict[str, int]] = {}

    eligible_users = [
        user for user, messages in by_user.items() if len(messages) >= min_messages
    ]
    eligible_users.sort(key=lambda user: len(by_user[user]), reverse=True)

    print(f"Eligible users: {len(eligible_users)}")

    for user_index, user in enumerate(eligible_users, start=1):
        messages = by_user[user]
        chunks = chunk_messages(messages, max_chunk_chars)
        if not chunks:
            continue

        weighted_sum = None
        total_weight = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embed_local(batch, model)
            weights = np.array([len(item) for item in batch], dtype="float32")
            batch_sum = (vectors * weights[:, None]).sum(axis=0)

            weighted_sum = batch_sum if weighted_sum is None else weighted_sum + batch_sum
            total_weight += int(weights.sum())

        if weighted_sum is None or total_weight == 0:
            continue

        user_vectors[user] = normalize(weighted_sum / total_weight)
        user_meta[user] = {
            "messages": len(messages),
            "chars": sum(len(message) for message in messages),
            "chunks": len(chunks),
            "cleaned": clean,
        }
        print(
            f"[{user_index}/{len(eligible_users)}] {user}: "
            f"{len(messages)} messages, {len(chunks)} chunks"
        )

    return user_vectors, user_meta


def build_similarity_report(
    user_vectors: dict[str, np.ndarray],
    user_meta: dict[str, dict[str, int]],
    top_k: int,
    clean_config: dict[str, object],
) -> dict[str, object]:
    users = list(user_vectors)
    matrix = np.stack([user_vectors[user] for user in users]).astype("float32")
    similarities = matrix @ matrix.T

    results: dict[str, list[dict[str, object]]] = {}
    for index, user in enumerate(users):
        ranked = np.argsort(-similarities[index])
        matches: list[dict[str, object]] = []
        for other_index in ranked:
            if other_index == index:
                continue
            other = users[int(other_index)]
            matches.append(
                {
                    "user": other,
                    "score": round(float(similarities[index, other_index]), 4),
                    "messages": user_meta[other]["messages"],
                    "chars": user_meta[other]["chars"],
                    "chunks": user_meta[other]["chunks"],
                }
            )
            if len(matches) >= top_k:
                break
        results[user] = matches

    return {
        "method": "full_user_messages_chunked_embedding_weighted_average",
        "note": "This measures combined topic and expression similarity, not pure style.",
        "cleaning": clean_config,
        "users": {
            user: {
                **user_meta[user],
                "similar_users": results[user],
            }
            for user in users
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute full-data user similarity from messages JSONL."
    )
    parser.add_argument("messages", type=Path, help="Path to data/messages.jsonl.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output JSON path. Default: {DEFAULT_OUT}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LOCAL_EMBEDDING_MODEL,
        help=f"Local embedding model. Default: {DEFAULT_LOCAL_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--min-messages",
        type=int,
        default=100,
        help="Only include users with at least this many messages. Default: 100.",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=1200,
        help="Max characters per full-data chunk. Default: 1200.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size. Default: 64.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Similar users to keep per user. Default: 10.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Disable repeated/low-information content filtering.",
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
        help="Drop exact messages repeated at least this many times globally. Default: 20.",
    )
    parser.add_argument(
        "--per-user-repeat-limit",
        type=int,
        default=3,
        help="Keep at most this many exact repeats per user. Default: 3.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.messages.exists():
        parser.error(f"File not found: {args.messages}")

    user_vectors, user_meta = build_user_vectors(
        args.messages,
        args.model,
        args.min_messages,
        args.max_chunk_chars,
        args.batch_size,
        not args.no_clean,
        args.min_chars,
        args.global_repeat_threshold,
        args.per_user_repeat_limit,
    )
    report = build_similarity_report(
        user_vectors,
        user_meta,
        args.top_k,
        {
            "enabled": not args.no_clean,
            "min_chars": args.min_chars,
            "global_repeat_threshold": args.global_repeat_threshold,
            "per_user_repeat_limit": args.per_user_repeat_limit,
        },
    )
    write_json(args.out, report)
    print(f"Wrote user similarity report to {args.out}")


if __name__ == "__main__":
    main()
