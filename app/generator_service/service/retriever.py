from typing import Any
import numpy as np
from .db import get_conn

__all__ = [
    "select_candidate_clusters",
    "fetch_style_and_topic",
    "fetch_exemplars_and_similar",
]

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

def _print_table(rows: list[dict[str, Any]], cols: list[str], title: str):
    print(f"\n===== {title} =====")
    if not rows:
        print("(empty)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    print("=" * len(header))

def select_candidate_clusters(q_emb: np.ndarray, topk: int = 8, section: str | None = None) -> list[dict[str, Any]]:
    q = _vec_sql(q_emb)
    with get_conn() as conn, conn.cursor() as cur:
        mv_ready = _is_matview_populated(conn, "public", "mv_cluster_trend_score")
        print(f"[retriever] mv_cluster_trend_score populated: {mv_ready}")

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
                ct.topic,
                tp.total_count_30d,
                tp.top_max_phrase,
                tp.top_max_count,
                tp.top_requests,
                tp.associations
              FROM nn_centroid n
              JOIN public.cluster_topics_by_section ct
                ON ct.cluster_id = n.cluster_id
              LEFT JOIN public.mv_cluster_trend_score ts
                ON ts.section = ct.section AND ts.cluster_id = ct.cluster_id
              LEFT JOIN public.topic_popularity_wordstat tp
                ON tp.section = ct.section AND tp.cluster_id = ct.cluster_id
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
                0::double precision                 AS trend_score,
                ct.topic,
                tp.total_count_30d,
                tp.top_max_phrase,
                tp.top_max_count,
                tp.top_requests,
                tp.associations
              FROM nn_centroid n
              JOIN public.cluster_topics_by_section ct
                ON ct.cluster_id = n.cluster_id
              LEFT JOIN public.topic_popularity_wordstat tp
                ON tp.section = ct.section AND tp.cluster_id = ct.cluster_id
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
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

        _print_table(
            rows,
            ["section", "cluster_id", "sim_centroid", "sim_title", "trend_score", "final_score", "topic"],
            title=f"Candidate clusters (top {min(3, len(rows))})"
        )
        return rows

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
              COALESCE(ts.trend_score, 0) AS trend_score,
              tp.total_count_30d, tp.top_max_phrase, tp.top_max_count,
              tp.top_requests, tp.associations
            FROM public.cluster_style cs
            JOIN public.cluster_topics_by_section ct
              ON ct.cluster_id = cs.cluster_id AND ct.section = cs.section
            LEFT JOIN public.mv_cluster_trend_score ts
              ON ts.section = ct.section AND ts.cluster_id = ct.cluster_id
            LEFT JOIN public.topic_popularity_wordstat tp
              ON tp.section = ct.section AND tp.cluster_id = ct.cluster_id
            WHERE cs.cluster_id = %s AND cs.section = %s;
            """
        else:
            sql = """
            SELECT
              cs.section,
              cs.cluster_id,
              cs.avg_len_tokens, cs.p95_len_tokens, cs.emoji_rate, cs.exclam_rate,
              cs.question_rate, cs.markdown_rate, cs.top_ngrams, cs.slang_ngrams,
              ct.topic,
              0::double precision AS trend_score,
              tp.total_count_30d, tp.top_max_phrase, tp.top_max_count,
              tp.top_requests, tp.associations
            FROM public.cluster_style cs
            JOIN public.cluster_topics_by_section ct
              ON ct.cluster_id = cs.cluster_id AND ct.section = cs.section
            LEFT JOIN public.topic_popularity_wordstat tp
              ON tp.section = ct.section AND tp.cluster_id = ct.cluster_id
            WHERE cs.cluster_id = %s AND cs.section = %s;
            """
        cur.execute(sql, (cluster_id, section))
        row = cur.fetchone()
        if not row:
            print(f"[retriever] No style/topic for section={section}, cluster_id={cluster_id}")
            return {}
        cols = [d.name for d in cur.description]
        out = dict(zip(cols, row))

        print("\n===== Chosen cluster style & popularity =====")
        print(f"section:        {out.get('section')}")
        print(f"cluster_id:     {out.get('cluster_id')}")
        print(f"topic:          {out.get('topic')}")
        print(f"trend_score:    {float(out.get('trend_score') or 0):.3f}")
        print(f"avg_len_tokens: {out.get('avg_len_tokens')} | p95_len_tokens: {out.get('p95_len_tokens')}")
        print(f"emoji_rate:     {float(out.get('emoji_rate') or 0):.3f} | "
              f"exclam_rate: {float(out.get('exclam_rate') or 0):.3f} | "
              f"question_rate: {float(out.get('question_rate') or 0):.3f}")
        print(f"markdown_rate:  {float(out.get('markdown_rate') or 0):.3f}")
        print(f"top_ngrams:     {', '.join((out.get('top_ngrams') or [])[:10])}")
        print(f"slang_ngrams:   {', '.join((out.get('slang_ngrams') or [])[:10])}")
        print(f"wordstat total_count_30d: {out.get('total_count_30d')}")
        print(f"wordstat top_max_phrase:  {out.get('top_max_phrase')}")
        print(f"wordstat top_max_count:   {out.get('top_max_count')}")
        print("============================================\n")

        return out

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

    print(f"[retriever] Exemplars: {len(exemplars)} | Similar to query: {len(similar)}")
    return {"exemplars": exemplars, "similar": similar}
