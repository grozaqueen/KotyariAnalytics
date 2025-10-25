import os
import json
import numpy as np
from sklearn.preprocessing import normalize
import hdbscan, umap
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from etl.common.db import get_conn

load_dotenv()

def _to_vec(x) -> np.ndarray:
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype="float32")
    if isinstance(x, memoryview):
        x = x.tobytes().decode("utf-8", errors="ignore")
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        return np.asarray(json.loads(x), dtype="float32")
    raise TypeError(f"Unexpected emb type: {type(x)}")

def load_embeddings():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT t.id, t.section, pe.emb::text
            FROM public.topics t
            JOIN public.post_embeddings pe ON pe.post_id = t.id
            ORDER BY t.id
        """)
        rows = cur.fetchall()

    if not rows:
        return [], [], np.zeros((0, 1024), dtype="float32")

    ids = [r[0] for r in rows]
    sections = [r[1] for r in rows]
    embs = np.vstack([_to_vec(r[2]) for r in rows])

    if embs.shape[1] != 1024:
        raise RuntimeError(f"Ожидалось 1024 измерения, получено {embs.shape[1]}")
    return ids, sections, embs

def wipe_previous_results():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE public.message_clusters;")
        cur.execute("TRUNCATE TABLE public.cluster_centroids;")
        conn.commit()

def write_clusters(ids, sections, labels):
    if len(ids) != len(labels) or len(sections) != len(labels):
        raise RuntimeError("Размерности ids/sections/labels не совпадают")

    payload = [(int(pid), int(lbl), sec) for pid, sec, lbl in zip(ids, sections, labels)]
    with get_conn() as conn, conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO public.message_clusters(topic_id, cluster_id, section)
            VALUES %s
            ON CONFLICT (topic_id)
            DO UPDATE SET cluster_id = EXCLUDED.cluster_id,
                          section    = EXCLUDED.section
        """, payload, page_size=10_000)
        conn.commit()

def recompute_centroids():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            WITH agg AS (
              SELECT
                mc.cluster_id,
                COUNT(*) AS sz,
                AVG(pe.emb) AS centroid
              FROM public.message_clusters mc
              JOIN public.post_embeddings pe ON pe.post_id = mc.topic_id
              WHERE mc.cluster_id <> -1
              GROUP BY mc.cluster_id
            )
            INSERT INTO public.cluster_centroids(cluster_id, centroid, size)
            SELECT cluster_id, centroid, sz FROM agg
            ON CONFLICT (cluster_id)
            DO UPDATE SET centroid  = EXCLUDED.centroid,
                          size      = EXCLUDED.size,
                          updated_at= now();
        """)
        conn.commit()

def main():
    ids, sections, X = load_embeddings()
    if len(ids) == 0:
        print("Нет эмбеддингов — кластеризовать нечего.")
        return

    wipe_previous_results()

    Xn = normalize(X)

    reducer = umap.UMAP(
        n_neighbors=5, min_dist=0.0, n_components=55,
        metric="cosine", random_state=42
    )
    Xr = reducer.fit_transform(Xn)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=4, min_samples=1, metric="euclidean",
        cluster_selection_method="leaf", cluster_selection_epsilon=0.01,
        prediction_data=True
    )
    labels = clusterer.fit_predict(Xr)

    write_clusters(ids, sections, labels)
    recompute_centroids()
    print("Кластеризация и центроиды обновлены (полная пересборка).")

if __name__ == "__main__":
    main()
