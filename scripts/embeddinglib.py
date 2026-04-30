from __future__ import annotations

from functools import lru_cache

import faiss
import numpy as np


DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


@lru_cache(maxsize=2)
def load_local_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: sentence-transformers. Install it with "
            "`uv add sentence-transformers`."
        ) from exc

    return SentenceTransformer(model_name)


def embed_local(texts: list[str], model_name: str) -> np.ndarray:
    model = load_local_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 100,
    )
    return np.asarray(vectors, dtype="float32")


def embed_openai(client, texts: list[str], model_name: str) -> np.ndarray:
    response = client.embeddings.create(model=model_name, input=texts)
    vectors = [item.embedding for item in response.data]
    array = np.array(vectors, dtype="float32")
    faiss.normalize_L2(array)
    return array
