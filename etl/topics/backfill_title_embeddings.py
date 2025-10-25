import os

from etl.common.db import get_conn
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "ai-forever/sbert_large_nlu_ru",
    cache_folder = os.path.expanduser("~/.cache/sentence_transformers")
)

def vec_sql(v):
    return "[" + ",".join(f"{float(x):.8f}" for x in v.tolist()) + "]"

def main():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT section, cluster_id, topic
            FROM public.cluster_topics_by_section
            WHERE (title_emb IS NULL) AND topic IS NOT NULL AND topic <> ''
        """)
        rows = cur.fetchall()
        if not rows:
            print("Нет тем без эмбеддингов.")
            return
        topics = [r[2] for r in rows]
        embs = model.encode(topics, convert_to_numpy=True, normalize_embeddings=True)
        for (section, cluster_id, _), emb in zip(rows, embs):
            cur.execute("""
                UPDATE public.cluster_topics_by_section
                SET title_emb=%s::vector
                WHERE section=%s AND cluster_id=%s
            """, (vec_sql(emb), section, cluster_id))
        conn.commit()
    print("Эмбеддинги тем заполнены.")

if __name__ == "__main__":
    main()
