from typing import Any
from .embeddings import embed_one
from .retriever import (
    select_candidate_clusters,
    fetch_style_and_topic,
    fetch_exemplars_and_similar,
)

def build_context_for_query(user_query: str) -> dict[str, Any]:
    print(f"\n=== Build context for query: {user_query!r} ===")
    q_emb = embed_one(user_query)
    print("[context_builder] Query embedding created.")

    candidates = select_candidate_clusters(q_emb, topk=8)

    if not candidates:
        print("[context_builder] No candidates found.")
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
    print(f"[context_builder] Chosen: section={sec}, cluster_id={cid}, final_score={chosen.get('final_score'):.4f}")

    style = fetch_style_and_topic(cid, sec)
    samples = fetch_exemplars_and_similar(cid, sec, q_emb, exemplars_k=4, similar_k=8)

    ctx = {
        "query": user_query,
        "candidates": candidates,
        "chosen": chosen,
        "style": style,
        "exemplars": samples["exemplars"],
        "similar": samples["similar"],
    }

    print("[context_builder] Context built.\n")
    return ctx
