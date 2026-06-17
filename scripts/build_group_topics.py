from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from chatlib import read_jsonl, write_json


DEFAULT_WINDOWS = Path("data/windows.jsonl")
DEFAULT_OUT = Path("data/reports/group_topics_base.json")
CHINESE_TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{2,}")


def normalize_text(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_keywords(texts: list[str], limit: int) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in CHINESE_TOKEN_RE.findall(text):
            if len(token) >= 2:
                counter[token] += 1
    return [word for word, _ in counter.most_common(limit)]


def load_topic_documents(windows_path: Path, max_docs: int, min_chars: int) -> list[dict[str, object]]:
    docs = []
    for window in read_jsonl(windows_path):
        text = normalize_text(str(window.get("full_text", "")))
        if len(text) < min_chars:
            continue
        docs.append(
            {
                "id": int(window["id"]),
                "user": str(window["user"]),
                "line_no": int(window["line_no"]),
                "time": window.get("time"),
                "text": text,
                "target": str(window.get("target", "")),
            }
        )
        if len(docs) >= max_docs:
            break
    return docs


def build_topics(
    windows_path: Path,
    out: Path,
    topic_count: int,
    max_docs: int,
    min_chars: int,
    examples_per_topic: int,
    keywords_per_topic: int,
) -> dict[str, object]:
    docs = load_topic_documents(windows_path, max_docs, min_chars)
    if len(docs) < topic_count:
        raise ValueError(f"Not enough documents for {topic_count} topics: {len(docs)}")

    texts = [str(doc["text"]) for doc in docs]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=3,
        max_df=0.6,
        max_features=50000,
    )
    matrix = vectorizer.fit_transform(texts)
    model = MiniBatchKMeans(n_clusters=topic_count, random_state=42, batch_size=512, n_init="auto")
    labels = model.fit_predict(matrix)
    distances = model.transform(matrix)

    grouped: dict[int, list[int]] = {index: [] for index in range(topic_count)}
    for doc_index, label in enumerate(labels):
        grouped[int(label)].append(doc_index)

    topics = []
    for label, indexes in grouped.items():
        if not indexes:
            continue
        indexes.sort(key=lambda index: distances[index, label])
        topic_docs = [docs[index] for index in indexes]
        topic_texts = [str(doc["text"]) for doc in topic_docs]
        user_counts = Counter(str(doc["user"]) for doc in topic_docs)

        topics.append(
            {
                "id": f"topic:{label}",
                "size": len(topic_docs),
                "keywords": extract_keywords(topic_texts, keywords_per_topic),
                "top_users": [
                    {"user": user, "count": count}
                    for user, count in user_counts.most_common(10)
                ],
                "examples": [
                    {
                        "window_id": doc["id"],
                        "line_no": doc["line_no"],
                        "time": doc["time"],
                        "user": doc["user"],
                        "target": doc["target"],
                        "text": doc["text"],
                    }
                    for doc in topic_docs[:examples_per_topic]
                ],
            }
        )

    topics.sort(key=lambda item: item["size"], reverse=True)
    result = {
        "meta": {
            "source": str(windows_path),
            "documents": len(docs),
            "topic_count": len(topics),
            "method": "tfidf_char_ngrams_minibatch_kmeans",
        },
        "topics": topics,
    }
    write_json(out, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build base group topic clusters from chat windows.")
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--topic-count", type=int, default=12)
    parser.add_argument("--max-docs", type=int, default=30000)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--examples-per-topic", type=int, default=8)
    parser.add_argument("--keywords-per-topic", type=int, default=20)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.windows.exists():
        parser.error(f"File not found: {args.windows}")

    result = build_topics(
        args.windows,
        args.out,
        args.topic_count,
        args.max_docs,
        args.min_chars,
        args.examples_per_topic,
        args.keywords_per_topic,
    )
    print(f"Wrote {len(result['topics'])} topics to {args.out}")


if __name__ == "__main__":
    main()
