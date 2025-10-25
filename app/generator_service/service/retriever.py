from typing import Any
import numpy as np
from .db import get_conn

def _vec_sql(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vec.tolist()) + "]"

def _is_matview_populated(conn, schema: str, name: str) -> bool:
    sql = """
    SELECT c.relispopulated
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = %s AND c.relname = %s AND c.relkind = 'm';
    """
    with conn.cursor() as cur:
        cur.execute(sql, (schema, name))
        row = cur.fetchone()
        return bool(row and row[0])

def select_candidate_clusters(q_emb: np.ndarray, topk: int = 8, section: str | None = None) -> list[dict[str, Any]]:
    q = _vec_sql(q_emb)
    with get_conn() as conn, conn.cursor() as cur:
        mv_ready = _is_matview_populated(conn, "public", "mv_cluster_trend_score")

        if mv_ready:
            sql = """
            WITH nn_centroid AS (
              SELECT cc.cluster_id,
                     1.0 - (cc.centroid <#> %s::vector) AS sim_centroid
              FROM public.cluster_centroids cc
              ORDER BY cc.centroid <#> %s::vector
              LIMIT %s
            ),
            joined AS (
              SELECT
                ct.section,
                n.cluster_id,
                n.sim_centroid,
                1.0 - (ct.title_emb <#> %s::vector) AS sim_title,
                COALESCE(ts.trend_score, 0)        AS trend_score,
                ct.topic
              FROM nn_centroid n
              JOIN public.cluster_topics_by_section ct
                ON ct.cluster_id = n.cluster_id
              LEFT JOIN public.mv_cluster_trend_score ts
                ON ts.section = ct.section AND ts.cluster_id = ct.cluster_id
              WHERE (%s IS NULL OR ct.section = %s)
            )
            SELECT *,
                   (0.55*sim_centroid + 0.25*COALESCE(sim_title,0) + 0.20*trend_score) AS final_score
            FROM joined
            ORDER BY final_score DESC
            LIMIT 3;
            """
            params = (q, q, topk, q, section, section)
        else:
            sql = """
            WITH nn_centroid AS (
              SELECT cc.cluster_id,
                     1.0 - (cc.centroid <#> %s::vector) AS sim_centroid
              FROM public.cluster_centroids cc
              ORDER BY cc.centroid <#> %s::vector
              LIMIT %s
            ),
            joined AS (
              SELECT
                ct.section,
                n.cluster_id,
                n.sim_centroid,
                1.0 - (ct.title_emb <#> %s::vector) AS sim_title,
                0::double precision             AS trend_score,
                ct.topic
              FROM nn_centroid n
              JOIN public.cluster_topics_by_section ct
                ON ct.cluster_id = n.cluster_id
              WHERE (%s IS NULL OR ct.section = %s)
            )
            SELECT *,
                   (0.55*sim_centroid + 0.25*COALESCE(sim_title,0) + 0.20*trend_score) AS final_score
            FROM joined
            ORDER BY final_score DESC
            LIMIT 3;
            """
            params = (q, q, topk, q, section, section)

        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

def fetch_style_and_topic(cluster_id: int, section: str) -> dict:
    with get_conn() as conn, conn.cursor() as cur:
        mv_ready = _is_matview_populated(conn, "public", "mv_cluster_trend_score")
        if mv_ready:
            sql = """
            SELECT
              cs.section,
              cs.cluster_id,
              cs.avg_len_tokens, cs.p95_len_tokens, cs.emoji_rate, cs.exclam_rate,
              cs.question_rate, cs.markdown_rate, cs.top_ngrams, cs.slang_ngrams,
              ct.topic,
              COALESCE(ts.trend_score, 0) AS trend_score
            FROM public.cluster_style cs
            JOIN public.cluster_topics_by_section ct
              ON ct.cluster_id = cs.cluster_id AND ct.section = cs.section
            LEFT JOIN public.mv_cluster_trend_score ts
              ON ts.section = ct.section AND ts.cluster_id = ct.cluster_id
            WHERE cs.cluster_id = %s AND cs.section = %s;
            """
            cur.execute(sql, (cluster_id, section))
        else:
            sql = """
            SELECT
              cs.section,
              cs.cluster_id,
              cs.avg_len_tokens, cs.p95_len_tokens, cs.emoji_rate, cs.exclam_rate,
              cs.question_rate, cs.markdown_rate, cs.top_ngrams, cs.slang_ngrams,
              ct.topic,
              0::double precision AS trend_score
            FROM public.cluster_style cs
            JOIN public.cluster_topics_by_section ct
              ON ct.cluster_id = cs.cluster_id AND ct.section = cs.section
            WHERE cs.cluster_id = %s AND cs.section = %s;
            """
            cur.execute(sql, (cluster_id, section))

        row = cur.fetchone()
        if not row:
            return {}
        cols = [d.name for d in cur.description]
        return dict(zip(cols, row))

def fetch_exemplars_and_similar(cluster_id: int, section: str, q_emb: np.ndarray, exemplars_k=4, similar_k=8) -> dict:
    q = _vec_sql(q_emb)
    sql_exemplars = """
    SELECT p.id, p.text_clean AS text
    FROM public.cluster_style cs
    JOIN LATERAL unnest(cs.exemplar_post_ids) AS e(pid) ON TRUE
    JOIN public.topics p ON p.id = e.pid
    WHERE cs.cluster_id = %s AND cs.section = %s
    LIMIT %s;
    """
    sql_similar = """
    SELECT p.id, p.text_clean AS text
    FROM public.message_clusters mc
    JOIN public.topics p ON p.id = mc.topic_id
    JOIN public.post_embeddings pe ON pe.post_id = p.id
    WHERE mc.cluster_id = %s AND mc.section = %s
    ORDER BY pe.emb <#> %s::vector
    LIMIT %s;
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql_exemplars, (cluster_id, section, exemplars_k))
        cols_e = [d.name for d in cur.description]
        exemplars = [dict(zip(cols_e, r)) for r in cur.fetchall()]

        cur.execute(sql_similar, (cluster_id, section, q, similar_k))
        cols_s = [d.name for d in cur.description]
        similar = [dict(zip(cols_s, r)) for r in cur.fetchall()]

    return {"exemplars": exemplars, "similar": similar}
