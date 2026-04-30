from __future__ import annotations

import argparse
from pathlib import Path

import faiss
from openai import OpenAI

from chatlib import read_jsonl, write_json
from cleaning import filter_window_rows
from embeddinglib import DEFAULT_LOCAL_EMBEDDING_MODEL, embed_local, embed_openai


DEFAULT_OUT_DIR = Path("data/vector")
DEFAULT_PROVIDER = "local"
DEFAULT_MODEL = DEFAULT_LOCAL_EMBEDDING_MODEL
DEFAULT_MAX_CHARS = 2000


def batched(items: list[dict[str, object]], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def build_index(
    windows_path: Path,
    out_dir: Path,
    provider: str,
    model: str,
    batch_size: int,
    users: list[str] | None,
    max_chars: int,
    clean: bool,
    min_chars: int,
    global_repeat_threshold: int,
    per_user_repeat_limit: int,
) -> None:
    windows = [
        window
        for window in read_jsonl(windows_path)
        if users is None or str(window["user"]) in users
    ]
    if not windows:
        raise ValueError(f"No windows found in {windows_path}")

    clean_stats = None
    if clean:
        windows, clean_stats = filter_window_rows(
            windows,
            min_chars,
            global_repeat_threshold,
            per_user_repeat_limit,
        )
        print(f"Cleaning stats: {clean_stats}")
        if not windows:
            raise ValueError("No windows left after cleaning.")

    client = OpenAI() if provider == "openai" else None
    index = None
    window_ids: list[int] = []
    dimensions = None

    for batch_no, batch in enumerate(batched(windows, batch_size), start=1):
        texts = [str(item["full_text"])[:max_chars] for item in batch]
        vectors = (
            embed_openai(client, texts, model)
            if provider == "openai"
            else embed_local(texts, model)
        )

        if index is None:
            dimensions = int(vectors.shape[1])
            index = faiss.IndexFlatIP(dimensions)

        index.add(vectors)
        window_ids.extend(int(item["id"]) for item in batch)
        print(f"Indexed batch {batch_no}: {len(window_ids)}/{len(windows)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "windows.faiss"))
    write_json(
        out_dir / "window_ids.json",
        {
            "source": str(windows_path),
            "embedding_provider": provider,
            "embedding_model": model,
            "distance": "cosine",
            "dimensions": dimensions,
            "users": users,
            "max_chars": max_chars,
            "clean": clean,
            "clean_stats": clean_stats,
            "min_chars": min_chars,
            "global_repeat_threshold": global_repeat_threshold,
            "per_user_repeat_limit": per_user_repeat_limit,
            "window_ids": window_ids,
        },
    )

    print(f"Wrote FAISS index to {out_dir / 'windows.faiss'}")
    print(f"Wrote window id mapping to {out_dir / 'window_ids.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a FAISS index for chat windows.")
    parser.add_argument("windows", type=Path, help="Path to data/windows.jsonl.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--provider",
        choices=["local", "openai"],
        default=DEFAULT_PROVIDER,
        help=f"Embedding provider. Default: {DEFAULT_PROVIDER}.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Embedding batch size. Default: 100.",
    )
    parser.add_argument(
        "--user",
        action="append",
        help="Only index this exact user. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"Max characters per window sent for embedding. Default: {DEFAULT_MAX_CHARS}.",
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
        help="Drop target messages shorter than this when cleaning. Default: 4.",
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

    if not args.windows.exists():
        parser.error(f"File not found: {args.windows}")

    build_index(
        args.windows,
        args.out_dir,
        args.provider,
        args.model,
        args.batch_size,
        args.user,
        args.max_chars,
        not args.no_clean,
        args.min_chars,
        args.global_repeat_threshold,
        args.per_user_repeat_limit,
    )


if __name__ == "__main__":
    main()
