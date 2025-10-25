import os

import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer(
    "ai-forever/sbert_large_nlu_ru",
    cache_folder = os.path.expanduser("~/.cache/sentence_transformers")
)

def embed_texts(texts: list[str]) -> np.ndarray:
    embs = _model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")

def embed_one(text: str) -> np.ndarray:
    return embed_texts([text])[0]
