from __future__ import annotations

import argparse
from pathlib import Path

from chat_as import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_PROFILE_DIR,
    DEFAULT_VECTOR_DIR,
    DEFAULT_WINDOWS,
    generate_reply,
)


def build_context(history: list[tuple[str, str]], max_turns: int) -> str:
    recent = history[-max_turns * 2 :]
    return "\n".join(f"{speaker}:{content}" for speaker, content in recent)


def run_chat(args: argparse.Namespace) -> None:
    history: list[tuple[str, str]] = []

    print(f"Talking as: {args.user}")
    print("Commands: /exit to quit, /reset to clear context")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/reset":
            history.clear()
            print("Context cleared.")
            continue

        history.append((args.you, user_input))
        context = build_context(history, args.max_turns)

        try:
            reply = generate_reply(
                args.user,
                context,
                args.windows,
                args.vector_dir,
                args.profile_dir,
                args.group_profile,
                args.embedding_provider,
                args.embedding_model,
                args.chat_provider,
                args.chat_model,
                args.top_k,
                args.overfetch,
                args.debug,
            )
        except Exception as exc:
            history.pop()
            print(f"Error: {exc}")
            continue

        print(f"{args.user}: {reply}")
        history.append((args.user, reply))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start an interactive chat where the model imitates one user."
    )
    parser.add_argument("--user", required=True, help="Exact user name to imitate.")
    parser.add_argument(
        "--you",
        default="我",
        help="Name used for your messages in the context. Default: 我.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=12,
        help="How many recent conversation turns to keep. Default: 12.",
    )
    parser.add_argument(
        "--windows",
        type=Path,
        default=DEFAULT_WINDOWS,
        help=f"Windows JSONL path. Default: {DEFAULT_WINDOWS}",
    )
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=DEFAULT_VECTOR_DIR,
        help=f"Vector directory. Default: {DEFAULT_VECTOR_DIR}",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=DEFAULT_PROFILE_DIR,
        help=f"Profile directory. Default: {DEFAULT_PROFILE_DIR}",
    )
    parser.add_argument(
        "--group-profile",
        type=Path,
        default=None,
        help="Optional group profile path. Example: data/profiles/group.md",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["local", "openai"],
        default=DEFAULT_EMBEDDING_PROVIDER,
        help=f"Embedding provider. Default: {DEFAULT_EMBEDDING_PROVIDER}",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model. Default: {DEFAULT_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--chat-provider",
        choices=["deepseek", "openai"],
        default=DEFAULT_CHAT_PROVIDER,
        help=f"Chat provider. Default: {DEFAULT_CHAT_PROVIDER}",
    )
    parser.add_argument(
        "--chat-model",
        default=DEFAULT_CHAT_MODEL,
        help=f"Chat model. Default: {DEFAULT_CHAT_MODEL}",
    )
    parser.add_argument("--top-k", type=int, default=8, help="History examples to use.")
    parser.add_argument(
        "--overfetch",
        type=int,
        default=20,
        help="Search multiplier before filtering by user. Default: 20.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieved examples and the final prompt for every turn.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_chat(args)


if __name__ == "__main__":
    main()
