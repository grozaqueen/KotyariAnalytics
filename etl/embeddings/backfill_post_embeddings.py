import os

import numpy as np
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_values
from etl.common.db import get_conn

model = SentenceTransformer(
    "ai-forever/sbert_large_nlu_ru",
    cache_folder = os.path.expanduser("~/.cache/sentence_transformers")
)

def embed(texts: list[str]) -> np.ndarray:
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")

def main(batch=1000):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT t.id, t.text_clean
            FROM public.topics t
            LEFT JOIN public.post_embeddings pe ON pe.post_id = t.id
            WHERE pe.post_id IS NULL
            ORDER BY t.id
        """)
        rows = cur.fetchall()
        if not rows:
            print("Нет новых постов для эмбеддинга.")
            return
        print(f"К эмбеддингу: {len(rows)}")

        for i in range(0, len(rows), batch):
            chunk = rows[i:i+batch]
            ids = [r[0] for r in chunk]
            texts = [r[1] or "" for r in chunk]
            embs = embed(texts)
            vals = [(pid, "[" + ",".join(f"{float(x):.8f}" for x in v.tolist()) + "]") for pid, v in zip(ids, embs)]
            execute_values(cur,
                "INSERT INTO public.post_embeddings(post_id, emb) VALUES %s "
                "ON CONFLICT (post_id) DO UPDATE SET emb = EXCLUDED.emb",
                vals
            )
            conn.commit()
            print(f"Upsert {i+len(chunk)}/{len(rows)}")
    print("Готово.")

if __name__ == "__main__":
    main()
