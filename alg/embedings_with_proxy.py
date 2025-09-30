import socket
import time
from typing import List, Optional
import os
import httpx

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

cache_dir = os.path.expanduser("~/.cache/sentence_transformers")

model_russian = SentenceTransformer("ai-forever/sbert_large_nlu_ru", cache_folder=cache_dir)

model_eng = SentenceTransformer("BAAI/bge-large-en-v1.5", cache_folder=cache_dir)

model_multi = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                                  cache_folder=cache_dir)

# model_RoSBERTa = SentenceTransformer("ai-forever/ru-en-RoSBERTa", cache_folder=cache_dir)

def get_tfidf_embeddings(chunks: List[str]) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)

    vectorizer = TfidfVectorizer(norm='l2')
    X = vectorizer.fit_transform(chunks)
    return X.astype(np.float32).toarray()

_grok_cache: dict[str, np.ndarray] = {}

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec if n == 0 else (vec / n)

def _is_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False

def _detect_proxy_url() -> Optional[str]:
    if _is_port_open("127.0.0.1", 53425):
        return "socks5h://127.0.0.1:53425"  # h => DNS через прокси
    if _is_port_open("127.0.0.1", 53424):
        return "http://127.0.0.1:53424"
    return None

def _get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        proxy = _detect_proxy_url()
        _http_client = httpx.Client(proxy=proxy, timeout=30.0, trust_env=False)
    return _http_client

def _get_xai_client() -> OpenAI:
    global _xai_client
    XAI_API_KEY = os.getenv("XAI_API_KEY")
    if _xai_client is None:
        _xai_client = OpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1",
            http_client=_get_http_client(),
        )
    return _xai_client

def get_grok_embeddings(chunks: List[str]) -> np.ndarray:
    client = _get_xai_client()
    out: list[np.ndarray] = []

    for chunk in chunks:
        if chunk in _grok_cache:
            emb = _grok_cache[chunk]
        else:
            print(f"Запрос к API Grok для чанка: {chunk[:50]}...")
            last_exc: Exception | None = None
            for attempt in range(1, 4):
                try:
                    resp = client.embeddings.create(
                        input=chunk,
                        model="grok-4-fast-non-reasoning",
                    )
                    emb = np.asarray(resp.data[0].embedding, dtype=np.float32)
                    _grok_cache[chunk] = emb
                    break
                except Exception as e:
                    last_exc = e
                    time.sleep(1.5)
            else:
                raise last_exc or RuntimeError("Unknown error calling xAI embeddings")
        out.append(_l2_normalize(emb))
    return np.vstack(out)

def get_embeddings(chunks: List[str], lang: str = "ru") -> np.ndarray:

    key = (lang or "").strip().lower().replace(" ", "")
    if lang == "ru":
        return model_russian.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    elif lang == "en":
        return model_eng.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    elif lang == "multi":
        return model_multi.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    # elif lang == "model_RoSBERTa":
    #     return model_RoSBERTa.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    elif key == "tfidf" or key == "tfidF" or key == "tfidf":
        return get_tfidf_embeddings(chunks)
    elif key in {"grok"}:
        return get_grok_embeddings(chunks)
    raise ValueError(f"Unsupported lang: {lang!r}. Use 'grok' or 'tfidf'.")
