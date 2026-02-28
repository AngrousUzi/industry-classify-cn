"""
Utility functions for calling the BAAI/bge-m3 embedding API via SiliconFlow,
with batching and a persistent disk cache.
"""

import time
import pickle
import numpy as np
from pathlib import Path
from openai import OpenAI

MODEL    = "BAAI/bge-m3"
API_KEY  = "sk-fsplqxgvclighoynarcqyddmtmojfsnftupoudqnvpzthwrp"
BASE_URL = "https://api.siliconflow.cn/v1"

BATCH_SIZE  = 32   # texts per API call (SiliconFlow limit: 32 for bge-m3)
MAX_RETRIES = 5
RETRY_DELAY = 2    # seconds between retries (doubles on each failure)


def make_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def _embed_batch_raw(texts: list[str], client: OpenAI) -> list[np.ndarray]:
    """Call the API for a single batch; return one vector per text."""
    delay = RETRY_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(model=MODEL, input=texts)
            # response.data is sorted by index
            return [np.array(item.embedding, dtype=np.float32)
                    for item in sorted(response.data, key=lambda x: x.index)]
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  [retry {attempt + 1}/{MAX_RETRIES}] {e} — waiting {delay}s")
            time.sleep(delay)
            delay *= 2


def batch_embed(
    texts: list[str],
    client: OpenAI,
    cache_path: Path,
) -> dict[str, np.ndarray]:
    """
    Embed a list of texts using the API with a persistent disk cache.

    - Loads existing cache from `cache_path` (if it exists).
    - Only calls the API for texts not already cached.
    - Saves the updated cache back to disk after each batch.
    - Returns a dict mapping text → np.ndarray(embedding).
    """
    cache_path = Path(cache_path)

    # Load existing cache
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache: dict[str, np.ndarray] = pickle.load(f)
        print(f"Loaded {len(cache)} cached embeddings from {cache_path.name}")
    else:
        cache = {}

    # Identify texts that need embedding
    missing = [t for t in texts if t not in cache]
    print(f"{len(missing)} texts need embedding (out of {len(texts)} unique)")

    if not missing:
        return {t: cache[t] for t in texts if t in cache}

    # Process in batches
    for batch_start in range(0, len(missing), BATCH_SIZE):
        batch = missing[batch_start: batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(missing))
        print(f"  Embedding texts {batch_start + 1}–{batch_end} / {len(missing)} …", end=" ", flush=True)

        vectors = _embed_batch_raw(batch, client)
        for text, vec in zip(batch, vectors):
            cache[text] = vec

        # Save after every batch so progress is not lost if interrupted
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print("done")

    return {t: cache[t] for t in texts if t in cache}
