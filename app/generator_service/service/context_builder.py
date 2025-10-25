from typing import Any
from .embeddings import embed_one
from .retriever import (
    select_candidate_clusters,
    fetch_style_and_topic,
    fetch_exemplars_and_similar,
)

def build_context_for_query(user_query: str) -> dict[str, Any]:
    q_emb = embed_one(user_query)

    candidates = select_candidate_clusters(q_emb, topk=8)

    if not candidates:
        return {
            "query": user_query,
            "candidates": [],
            "chosen": None,
            "style": None,
            "exemplars": [],
            "similar": [],
        }

    chosen = candidates[0]
    sec = chosen["section"]
    cid = chosen["cluster_id"]

    style = fetch_style_and_topic(cid, sec)
    samples = fetch_exemplars_and_similar(cid, sec, q_emb, exemplars_k=4, similar_k=8)

    return {
        "query": user_query,
        "candidates": candidates,
        "chosen": chosen,
        "style": style,
        "exemplars": samples["exemplars"],
        "similar": samples["similar"],
    }
