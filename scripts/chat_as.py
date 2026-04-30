from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import faiss
from openai import OpenAI

from chatlib import load_dotenv, profile_filename, read_json, read_jsonl
from embeddinglib import DEFAULT_LOCAL_EMBEDDING_MODEL, embed_local, embed_openai


DEFAULT_MESSAGES = Path("data/messages.jsonl")
DEFAULT_WINDOWS = Path("data/windows.jsonl")
DEFAULT_VECTOR_DIR = Path("data/vector")
DEFAULT_PROFILE_DIR = Path("data/profiles")
DEFAULT_EMBEDDING_PROVIDER = "local"
DEFAULT_EMBEDDING_MODEL = DEFAULT_LOCAL_EMBEDDING_MODEL
DEFAULT_CHAT_PROVIDER = "deepseek"
DEFAULT_CHAT_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def build_chat_client(provider: str) -> OpenAI:
    load_dotenv()
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")
        base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI()


def build_embedding_client(provider: str):
    return OpenAI() if provider == "openai" else None


def embed_query(client, provider: str, model: str, text: str):
    if provider == "openai":
        return embed_openai(client, [text], model)
    return embed_local([text], model)


def load_windows_by_id(windows_path: Path) -> dict[int, dict[str, object]]:
    return {int(window["id"]): window for window in read_jsonl(windows_path)}


def load_profile(profile_dir: Path, user: str) -> str:
    path = profile_dir / profile_filename(user)
    if not path.exists():
        raise FileNotFoundError(
            f"Profile not found for {user}: {path}. Run scripts/generate_profile.py first."
        )
    return path.read_text(encoding="utf-8")


def load_group_profile(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def search_examples(
    client: OpenAI,
    user: str,
    context: str,
    windows_path: Path,
    vector_dir: Path,
    embedding_provider: str,
    embedding_model: str,
    top_k: int,
    overfetch: int,
    debug: bool = False,
) -> list[dict[str, object]]:
    index = faiss.read_index(str(vector_dir / "windows.faiss"))
    mapping = read_json(vector_dir / "window_ids.json")
    window_ids = [int(item) for item in mapping["window_ids"]]
    windows_by_id = load_windows_by_id(windows_path)

    if debug:
        print("\n[debug] embedding search", file=sys.stderr)
        print(f"[debug] target user: {user}", file=sys.stderr)
        print(f"[debug] embedding provider: {embedding_provider}", file=sys.stderr)
        print(f"[debug] embedding model: {embedding_model}", file=sys.stderr)
        print(f"[debug] index vectors: {len(window_ids)}", file=sys.stderr)
        print("[debug] query context:", file=sys.stderr)
        print(context, file=sys.stderr)

    query = embed_query(client, embedding_provider, embedding_model, context)
    limit = min(max(top_k * overfetch, top_k), len(window_ids))
    scores, positions = index.search(query, limit)

    examples: list[dict[str, object]] = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        window_id = window_ids[int(position)]
        window = windows_by_id.get(window_id)
        if not window or str(window["user"]) != user:
            continue
        examples.append(
            {
                "score": round(float(score), 4),
                "window_id": window_id,
                "target": window["target"],
                "full_text": window["full_text"],
            }
        )
        if len(examples) >= top_k:
            break

    if debug:
        print(f"[debug] searched candidates: {limit}", file=sys.stderr)
        print(f"[debug] matched examples after user filter: {len(examples)}", file=sys.stderr)
        for index_no, item in enumerate(examples, start=1):
            print(
                f"\n[debug] example #{index_no} "
                f"score={item['score']} window_id={item['window_id']}",
                file=sys.stderr,
            )
            print(str(item["full_text"]), file=sys.stderr)
            print(f"[debug] target reply: {item['target']}", file=sys.stderr)

    return examples


def build_generation_prompt(
    user: str,
    profile: str,
    group_profile: str,
    context: str,
    examples: list[dict[str, object]],
) -> str:
    examples_text = "\n\n".join(
        f"相似度 {item['score']}:\n{item['full_text']}\n该用户当时说：{item['target']}"
        for item in examples
    )
    if not examples_text:
        examples_text = "没有找到足够相似的历史样例，请只依据风格档案和当前上下文生成。"

    group_section = (
        f"""群聊整体风格：
{group_profile}

"""
        if group_profile
        else ""
    )

    return f"""你要模拟群聊用户“{user}”在这个群里的聊天风格生成一条回复。

{group_section}
风格档案：
{profile}

当前群聊上下文：
{context}

该用户过去类似场景的真实发言：
{examples_text}

输出要求：
- 只输出一条群聊消息正文
- 不输出用户名
- 不解释
- 不说自己是 AI
- 不声称自己就是该真人
- 不编造现实身份、现实承诺、地址、联系方式或隐私
- 长度、语气、标点尽量贴近该用户
"""


def generate_reply(
    user: str,
    context: str,
    windows_path: Path,
    vector_dir: Path,
    profile_dir: Path,
    group_profile_path: Path | None,
    embedding_provider: str,
    embedding_model: str,
    chat_provider: str,
    chat_model: str,
    top_k: int,
    overfetch: int,
    debug: bool = False,
) -> str:
    embedding_client = build_embedding_client(embedding_provider)
    chat_client = build_chat_client(chat_provider)
    profile = load_profile(profile_dir, user)
    group_profile = load_group_profile(group_profile_path)

    if debug:
        print("\n[debug] loaded profile", file=sys.stderr)
        print(profile, file=sys.stderr)
        if group_profile:
            print("\n[debug] loaded group profile", file=sys.stderr)
            print(group_profile, file=sys.stderr)

    examples = search_examples(
        embedding_client,
        user,
        context,
        windows_path,
        vector_dir,
        embedding_provider,
        embedding_model,
        top_k,
        overfetch,
        debug,
    )
    prompt = build_generation_prompt(user, profile, group_profile, context, examples)

    if debug:
        print("\n[debug] final prompt sent to chat model", file=sys.stderr)
        print(f"[debug] chat provider: {chat_provider}", file=sys.stderr)
        print(f"[debug] chat model: {chat_model}", file=sys.stderr)
        print(prompt, file=sys.stderr)

    response = chat_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}],
    )
    reply = (response.choices[0].message.content or "").strip()

    if debug:
        print("\n[debug] model reply", file=sys.stderr)
        print(reply, file=sys.stderr)

    return reply


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a chat reply as a target user.")
    parser.add_argument("--user", required=True, help="Exact user name to imitate.")
    parser.add_argument(
        "--context",
        help="Current chat context. If omitted, the script reads stdin.",
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
        help="Print retrieved examples and the final prompt sent to the chat model.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.windows.exists():
        parser.error(f"File not found: {args.windows}")
    if not (args.vector_dir / "windows.faiss").exists():
        parser.error(f"File not found: {args.vector_dir / 'windows.faiss'}")
    if not (args.vector_dir / "window_ids.json").exists():
        parser.error(f"File not found: {args.vector_dir / 'window_ids.json'}")

    context = args.context
    if context is None:
        print("Paste current chat context, then press Ctrl-D:")
        context = "".join(__import__("sys").stdin.readlines()).strip()
    if not context:
        parser.error("Empty context.")

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
    print(reply)


if __name__ == "__main__":
    main()
