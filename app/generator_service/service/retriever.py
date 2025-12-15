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


def _has_pg_trgm(conn) -> bool:
    sql = "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm');"
    with conn.cursor() as cur:
        cur.execute(sql)
        return bool(cur.fetchone()[0])


# def _print_table(rows: list[dict[str, Any]], cols: list[str], title: str):
#     print(f"\n===== {title} =====")
#     if not rows:
#         print("(empty)")
#         return
#     widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
#     header = " | ".join(c.ljust(widths[c]) for c in cols)
#     print(header)
#     print("-" * len(header))
#     for r in rows:
#         print(" | ".join(str(r.get(c, ""))[:widths[c]].ljust(widths[c]) for c in cols))
#     print("=" * len(header))


def _dedup_by_cluster(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        cid = int(r["cluster_id"])
        if cid not in seen:
            seen.add(cid)
            out.append(r)
    return out


def _nz(v, neg_big: float = -1e12) -> float:
    return float(v) if v is not None else neg_big


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return "NA"
    try:
        fv = float(v)
        if abs(fv - int(fv)) < 1e-9:
            return str(int(fv))
        return f"{fv:.{nd}f}"
    except Exception:
        return str(v)


def _rank_sort_key_tuple(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float]:
    section_lex_boost = 1.0 if row.get("section_lex_hit") else 0.0
    return (
        _nz(row.get("lex_hits_count")),
        section_lex_boost,
        _nz(row.get("section_match_score")),
        _nz(row.get("post_hits_at_tau")),
        _nz(row.get("lex_dens")),
        _nz(row.get("post_dens_at_tau")),
        _nz(row.get("post_topk_mean_sim")),
        _nz(row.get("final_score")),
    )


def _print_ranking_metrics(rows: list[dict[str, Any]], title: str):
    materialized = []
    for r in rows:
        sk = _rank_sort_key_tuple(r)
        materialized.append({
            "section": r.get("section"),
            "cluster_id": r.get("cluster_id"),
            "lex_hits_count": _fmt(r.get("lex_hits_count"), 0),
            "section_lex_hit": _fmt(1 if r.get("section_lex_hit") else 0, 0),
            "section_match_score": _fmt(r.get("section_match_score")),
            "post_hits_at_tau": _fmt(r.get("post_hits_at_tau"), 0),
            "lex_dens": _fmt(r.get("lex_dens")),
            "post_dens_at_tau": _fmt(r.get("post_dens_at_tau")),
            "post_topk_mean_sim": _fmt(r.get("post_topk_mean_sim")),
            "final_score": _fmt(r.get("final_score")),
            "sort_key": f"({', '.join(_fmt(x) for x in sk)})",
            "topic": r.get("topic"),
        })
    # _print_table(
    #     materialized,
    #     ["section", "cluster_id",
    #      "lex_hits_count", "section_lex_hit", "section_match_score",
    #      "post_hits_at_tau", "lex_dens", "post_dens_at_tau",
    #      "post_topk_mean_sim", "final_score",
    #      "sort_key", "topic"],
    #     title=title
    # )


def _check_params(sql: str, params: tuple):
    ph = sql.count("%s")
    if ph != len(params):
        print(f"[retriever][WARN] placeholders={ph}, params={len(params)}")


# =======================
#   POST-LEVEL METRICS
#   (используем mv_post_search)
# =======================

def _rerank_by_posts(
    cur,
    q_vec_sql: str,
    section: str | None,
    cluster_ids: list[int],
    *,
    query_text: str | None = None,
    ts_config: str = "simple",
    topk_posts: int = 50,
    tau: float | None = None,
) -> dict[int, dict[str, Any]]:

    if not cluster_ids:
        return {}

    # глобальный лимит по постам: topk_posts на каждый кластер
    global_limit = topk_posts * max(len(cluster_ids), 1)

    sql = """
    WITH sims AS (
      SELECT
        ps.cluster_id,
        ps.post_id,
        1.0 - (ps.emb <#> %s::vector) AS sim,
        CASE
          WHEN %s IS NULL THEN NULL
          ELSE ps.text_tsv @@ plainto_tsquery(%s, %s)
        END AS lex_hit,
        ROW_NUMBER() OVER (
          PARTITION BY ps.cluster_id
          ORDER BY ps.emb <#> %s::vector
        ) AS rn,
        COUNT(*) OVER (PARTITION BY ps.cluster_id) AS cluster_size
      FROM public.mv_post_search ps
      WHERE (%s IS NULL OR ps.section = %s)
      ORDER BY ps.emb <#> %s::vector
      LIMIT %s
    )
    SELECT
      cluster_id,
      MAX(cluster_size) AS cluster_size,
      AVG(sim) FILTER (WHERE rn <= %s) AS post_topk_mean_sim,
      SUM(CASE WHEN %s IS NULL THEN NULL ELSE (sim >= %s)::int END) AS post_hits_at_tau,
      (SUM(CASE WHEN %s IS NULL THEN NULL ELSE (sim >= %s)::int END))::float
        / NULLIF(MAX(cluster_size), 0) AS post_dens_at_tau,
      SUM(CASE WHEN lex_hit IS NULL THEN 0 ELSE lex_hit::int END) AS lex_hits_count,
      SUM(CASE WHEN lex_hit IS NULL THEN 0 ELSE lex_hit::int END)::float
        / NULLIF(MAX(cluster_size), 0) AS lex_dens
    FROM sims
    WHERE cluster_id = ANY(%s)
    GROUP BY cluster_id;
    """
    params = (
        q_vec_sql,
        query_text, ts_config, query_text,
        q_vec_sql,
        section, section,
        q_vec_sql, global_limit,
        topk_posts,
        query_text, tau,
        query_text, tau,
        cluster_ids,
    )
    _check_params(sql, params)
    cur.execute(sql, params)
    cols = [d.name for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    out: dict[int, dict[str, Any]] = {int(r["cluster_id"]): r for r in rows}

    # _print_table(
    #     rows,
    #     [
    #         "cluster_id", "cluster_size",
    #         "lex_hits_count", "lex_dens",
    #         "post_hits_at_tau", "post_dens_at_tau",
    #         "post_topk_mean_sim",
    #     ],
    #     title=f"Post-level metrics (approx, K={topk_posts}, global_limit={global_limit}, tau={'NA' if tau is None else tau}, ts_config={ts_config})"
    # )
    return out


def _discover_clusters_from_posts(
    cur,
    q_vec_sql: str,
    section: str | None,
    initial_ids: list[int],
    *,
    query_text: str | None = None,
    ts_config: str = "simple",
    global_posts_limit: int = 120,
    tau: float | None = None,
    max_new_clusters: int = 5,
) -> list[int]:

    sql = """
    WITH top_posts AS (
      SELECT
        ps.section,
        ps.cluster_id,
        1.0 - (ps.emb <#> %s::vector) AS sim,
        CASE
          WHEN %s IS NULL THEN NULL
          ELSE ps.text_tsv @@ plainto_tsquery(%s, %s)
        END AS lex_hit
      FROM public.mv_post_search ps
      WHERE (%s IS NULL OR ps.section = %s)
      ORDER BY ps.emb <#> %s::vector
      LIMIT %s
    )
    SELECT
      section,
      cluster_id,
      COUNT(*) AS hits_in_top,
      AVG(sim)  AS mean_sim,
      SUM(CASE WHEN %s IS NULL THEN 0 ELSE lex_hit::int END) AS lex_hits_in_top,
      SUM(CASE WHEN %s IS NULL THEN NULL ELSE (sim >= %s)::int END) AS sem_hits_in_top
    FROM top_posts
    GROUP BY section, cluster_id
    HAVING NOT (cluster_id = ANY(%s))
    ORDER BY
      lex_hits_in_top DESC,
      sem_hits_in_top DESC NULLS LAST,
      hits_in_top DESC,
      mean_sim DESC
    LIMIT %s;
    """
    params = (
        q_vec_sql,
        query_text, ts_config, query_text,
        section, section,
        q_vec_sql, global_posts_limit,
        query_text,
        query_text, tau,
        initial_ids,
        max_new_clusters,
    )
    _check_params(sql, params)
    cur.execute(sql, params)
    rows = cur.fetchall()
    new_ids = [int(r[1]) for r in rows]

    # _print_table(
    #     [
    #         {
    #             "section": r[0],
    #             "cluster_id": int(r[1]),
    #             "hits_in_top": r[2],
    #             "mean_sim": r[3],
    #             "lex_hits_in_top": r[4],
    #             "sem_hits_in_top": r[5],
    #         }
    #         for r in rows
    #     ],
    #     ["section", "cluster_id", "lex_hits_in_top", "sem_hits_in_top", "hits_in_top", "mean_sim"],
    #     title=f"Global top-post discovery by (section, cluster_id) (limit={global_posts_limit}, tau={'NA' if tau is None else tau})"
    # )
    return new_ids

def _fetch_candidates_for_ids(
    cur,
    q_vec_sql: str,
    section: str | None,
    cluster_ids: list[int],
    mv_ready: bool,
    *,
    query_text: str | None = None,
    ts_config: str = "simple",
    trgm_enabled: bool = True,
) -> list[dict[str, Any]]:

    if not cluster_ids:
        return []

    if trgm_enabled:
        sql = """
        WITH base AS (
          SELECT
            m.section,
            m.cluster_id,
            1.0 - (m.centroid <#> %s::vector) AS sim_centroid,
            1.0 - (m.title_emb <#> %s::vector) AS sim_title,
            m.trend_score AS trend_score,
            m.topic,
            m.total_count_30d,
            m.top_max_phrase,
            m.top_max_count,
            m.top_requests,
            m.associations,
            CASE
              WHEN %s IS NULL THEN NULL
              ELSE m.section_tsv @@ plainto_tsquery(%s, %s)
            END AS section_lex_hit,
            similarity(lower(m.section), lower(COALESCE(%s::text,''))) AS section_trgm_sim,
            (
              COALESCE(
                (CASE
                   WHEN %s IS NULL THEN NULL
                   ELSE (m.section_tsv @@ plainto_tsquery(%s, %s))::int
                 END), 0
              )::float * 0.7
              +
              similarity(lower(m.section), lower(COALESCE(%s::text,''))) * 0.3
            ) AS section_match_score
          FROM public.mv_cluster_retriever m
          WHERE m.cluster_id = ANY(%s)
            AND (%s IS NULL OR m.section = %s)
        )
        SELECT *,
               (0.55*sim_centroid + 0.25*COALESCE(sim_title,0) + 0.20*trend_score) AS final_score
        FROM base;
        """
        params = (
            q_vec_sql, q_vec_sql,
            query_text, ts_config, query_text,
            query_text,
            query_text, ts_config, query_text,
            query_text,
            cluster_ids, section, section,
        )
    else:
        sql = """
        WITH base AS (
          SELECT
            m.section,
            m.cluster_id,
            1.0 - (m.centroid <#> %s::vector) AS sim_centroid,
            1.0 - (m.title_emb <#> %s::vector) AS sim_title,
            m.trend_score AS trend_score,
            m.topic,
            m.total_count_30d,
            m.top_max_phrase,
            m.top_max_count,
            m.top_requests,
            m.associations,
            CASE
              WHEN %s IS NULL THEN NULL
              ELSE m.section_tsv @@ plainto_tsquery(%s, %s)
            END AS section_lex_hit,
            0.0::double precision AS section_trgm_sim,
            COALESCE(
              (CASE
                 WHEN %s IS NULL THEN NULL
                 ELSE (m.section_tsv @@ plainto_tsquery(%s, %s))::int
               END), 0
            )::float AS section_match_score
          FROM public.mv_cluster_retriever m
          WHERE m.cluster_id = ANY(%s)
            AND (%s IS NULL OR m.section = %s)
        )
        SELECT *,
               (0.55*sim_centroid + 0.25*COALESCE(sim_title,0) + 0.20*trend_score) AS final_score
        FROM base;
        """
        params = (
            q_vec_sql, q_vec_sql,
            query_text, ts_config, query_text,
            query_text, ts_config, query_text,
            cluster_ids, section, section,
        )

    _check_params(sql, params)
    cur.execute(sql, params)
    cols = [d.name for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    # _print_table(
    #     rows,
    #     ["section", "cluster_id", "sim_centroid", "sim_title", "trend_score",
    #      "section_match_score", "section_trgm_sim", "section_lex_hit",
    #      "final_score", "topic"],
    #     title=f"Augmented candidates fetched ({len(rows)}) (mv_cluster_retriever)"
    # )
    return rows


def select_candidate_clusters(
    q_emb: np.ndarray,
    topk: int = 8,
    section: str | None = None,
    query_text: str | None = None,
    ts_config: str = "simple",
    posts_topk: int = 50,
    posts_tau: float | None = None,
    reorder_by_posts: bool = True,
    enable_global_post_discovery: bool = True,
    global_posts_limit: int = 120,
    max_new_clusters: int = 5,
    final_limit: int = 3,
) -> list[dict[str, Any]]:

    q = _vec_sql(q_emb)
    with get_conn() as conn, conn.cursor() as cur:
        mv_ready = _is_matview_populated(conn, "public", "mv_cluster_trend_score")
        trgm_enabled = _has_pg_trgm(conn)
        print(f"[retriever] mv_cluster_trend_score populated: {mv_ready}")
        print(f"[retriever] pg_trgm enabled: {trgm_enabled}")

        if trgm_enabled:
            sql = """
            WITH nn_centroid AS (
              SELECT
                m.cluster_id,
                1.0 - (m.centroid <#> %s::vector) AS sim_centroid
              FROM public.mv_cluster_retriever m
              ORDER BY m.centroid <#> %s::vector
              LIMIT %s
            ),
            joined AS (
              SELECT
                m.section,
                n.cluster_id,
                n.sim_centroid,
                1.0 - (m.title_emb <#> %s::vector) AS sim_title,
                m.trend_score AS trend_score,
                m.topic,
                m.total_count_30d,
                m.top_max_phrase,
                m.top_max_count,
                m.top_requests,
                m.associations,
                CASE
                  WHEN %s IS NULL THEN NULL
                  ELSE m.section_tsv @@ plainto_tsquery(%s, %s)
                END AS section_lex_hit,
                similarity(lower(m.section), lower(COALESCE(%s::text,''))) AS section_trgm_sim,
                (
                  COALESCE(
                    (CASE
                       WHEN %s IS NULL THEN NULL
                       ELSE (m.section_tsv @@ plainto_tsquery(%s, %s))::int
                     END), 0
                  )::float * 0.7
                  +
                  similarity(lower(m.section), lower(COALESCE(%s::text,''))) * 0.3
                ) AS section_match_score
              FROM nn_centroid n
              JOIN public.mv_cluster_retriever m
                ON m.cluster_id = n.cluster_id
              WHERE (%s IS NULL OR m.section = %s)
            )
            SELECT *,
                   (0.55*sim_centroid + 0.25*COALESCE(sim_title,0) + 0.20*trend_score) AS final_score
            FROM joined
            ORDER BY final_score DESC
            LIMIT %s;
            """
            params = (
                q, q, topk, q,
                query_text, ts_config, query_text,
                query_text,
                query_text, ts_config, query_text,
                query_text,
                section, section,
                final_limit,
            )
        else:
            sql = """
            WITH nn_centroid AS (
              SELECT
                m.cluster_id,
                1.0 - (m.centroid <#> %s::vector) AS sim_centroid
              FROM public.mv_cluster_retriever m
              ORDER BY m.centroid <#> %s::vector
              LIMIT %s
            ),
            joined AS (
              SELECT
                m.section,
                n.cluster_id,
                n.sim_centroid,
                1.0 - (m.title_emb <#> %s::vector) AS sim_title,
                m.trend_score AS trend_score,
                m.topic,
                m.total_count_30d,
                m.top_max_phrase,
                m.top_max_count,
                m.top_requests,
                m.associations,
                CASE
                  WHEN %s IS NULL THEN NULL
                  ELSE m.section_tsv @@ plainto_tsquery(%s, %s)
                END AS section_lex_hit,
                0.0::double precision AS section_trgm_sim,
                COALESCE(
                  (CASE
                     WHEN %s IS NULL THEN NULL
                     ELSE (m.section_tsv @@ plainto_tsquery(%s, %s))::int
                   END), 0
                )::float AS section_match_score
              FROM nn_centroid n
              JOIN public.mv_cluster_retriever m
                ON m.cluster_id = n.cluster_id
              WHERE (%s IS NULL OR m.section = %s)
            )
            SELECT *,
                   (0.55*sim_centroid + 0.25*COALESCE(sim_title,0) + 0.20*trend_score) AS final_score
            FROM joined
            ORDER BY final_score DESC
            LIMIT %s;
            """
            params = (
                q, q, topk, q,
                query_text, ts_config, query_text,
                query_text, ts_config, query_text,
                section, section,
                final_limit,
            )

        _check_params(sql, params)
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

        # _print_table(
        #     rows,
        #     ["section", "cluster_id", "sim_centroid", "sim_title", "trend_score",
        #      "section_match_score", "section_trgm_sim", "section_lex_hit",
        #      "final_score", "topic"],
        #     title=f"Candidate clusters (stage 1, HNSW via mv_cluster_retriever, up to {final_limit})"
        # )

        if section is None and rows:
            locked_section = rows[0]["section"]
            print(f"[retriever] No 'section' provided. Locking to section='{locked_section}' for post-level checks.")
            before, rows = len(rows), [r for r in rows if r["section"] == locked_section]
            print(f"[retriever] Filtered candidates by section='{locked_section}': {before} → {len(rows)}")
            section = locked_section

        if enable_global_post_discovery and rows:
            initial_ids = [int(r["cluster_id"]) for r in rows]
            extra_ids = _discover_clusters_from_posts(
                cur, q, section, initial_ids,
                query_text=query_text,
                ts_config=ts_config,
                global_posts_limit=global_posts_limit,
                tau=posts_tau,
                max_new_clusters=max_new_clusters
            )
            if extra_ids:
                extra_rows = _fetch_candidates_for_ids(
                    cur, q, section, extra_ids, mv_ready,
                    query_text=query_text, ts_config=ts_config,
                    trgm_enabled=trgm_enabled
                )
                rows = _dedup_by_cluster(rows + extra_rows)
                # _print_table(
                #     rows,
                #     ["section", "cluster_id", "sim_centroid", "sim_title", "trend_score",
                #      "section_match_score", "section_trgm_sim", "section_lex_hit",
                #      "final_score", "topic"],
                #     title="Candidates after global post discovery (stage 2, section-locked)"
                # )

        if rows:
            cluster_ids_union = [int(r["cluster_id"]) for r in rows]
            post_metrics = _rerank_by_posts(
                cur, q, section, cluster_ids_union,
                query_text=query_text,
                ts_config=ts_config,
                topk_posts=posts_topk,
                tau=posts_tau
            )
            for r in rows:
                m = post_metrics.get(int(r["cluster_id"])) or {}
                r["lex_hits_count"]     = m.get("lex_hits_count")
                r["lex_dens"]           = m.get("lex_dens")
                r["post_topk_mean_sim"] = m.get("post_topk_mean_sim")
                r["post_hits_at_tau"]   = m.get("post_hits_at_tau")
                r["post_dens_at_tau"]   = m.get("post_dens_at_tau")
                r["cluster_size"]       = m.get("cluster_size")

            # _print_table(
            #     rows,
            #     ["section", "cluster_id", "cluster_size",
            #      "lex_hits_count", "lex_dens",
            #      "post_hits_at_tau", "post_dens_at_tau",
            #      "post_topk_mean_sim",
            #      "section_match_score", "section_trgm_sim", "section_lex_hit"],
            #     title="Candidates enriched with post-level metrics (stage 3, section-locked)"
            # )

            _print_ranking_metrics(rows, title="Ranking metrics (pre-sort)")
            if reorder_by_posts:
                rows.sort(key=_rank_sort_key_tuple, reverse=True)
                _print_ranking_metrics(rows, title="Ranking metrics (final order)")
                # _print_table(
                #     rows,
                #     ["section", "cluster_id", "lex_hits_count", "section_lex_hit",
                #      "section_match_score", "post_hits_at_tau",
                #      "lex_dens", "post_dens_at_tau", "post_topk_mean_sim",
                #      "final_score", "topic"],
                #     title="Final candidates order (lexical-count + section correlation first)"
                # )

        return rows[:final_limit]

def fetch_style_and_topic(cluster_id: int, section: str) -> dict:
    with get_conn() as conn, conn.cursor() as cur:
        sql = """
        SELECT *
        FROM public.mv_cluster_retriever
        WHERE cluster_id = %s AND section = %s;
        """
        cur.execute(sql, (cluster_id, section))
        row = cur.fetchone()
        if not row:
            print(f"[retriever] No style/topic for section={section}, cluster_id={cluster_id}")
            return {}
        cols = [d.name for d in cur.description]
        out = dict(zip(cols, row))

        print("\n===== Chosen cluster style & popularity (mv_cluster_retriever) =====")
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


def fetch_exemplars_and_similar(
    cluster_id: int,
    section: str,
    q_emb: np.ndarray,
    exemplars_k: int = 4,
    similar_k: int = 8
) -> dict:
    q = _vec_sql(q_emb)
    sql_exemplars = """
    SELECT p.id, p.text AS text
    FROM public.cluster_style cs
    JOIN LATERAL unnest(cs.exemplar_post_ids) AS e(pid) ON TRUE
    JOIN public.topics p ON p.id = e.pid
    WHERE cs.cluster_id = %s AND cs.section = %s
    LIMIT %s;
    """
    sql_similar = """
    SELECT post_id, text
    FROM public.mv_post_search
    WHERE cluster_id = %s AND section = %s
    ORDER BY emb <#> %s::vector
    LIMIT %s;
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql_exemplars, (cluster_id, section, exemplars_k))
        cols_e = [d.name for d in cur.description]
        exemplars = [dict(zip(cols_e, r)) for r in cur.fetchall()]

        cur.execute(sql_similar, (cluster_id, section, q, similar_k))
        cols_s = [d.name for d in cur.description]
        similar = [dict(zip(cols_s, r)) for r in cur.fetchall()]

    print(f"[retriever] Exemplars: {len(exemplars)} | Similar to query (HNSW via mv_post_search): {len(similar)}")
    return {"exemplars": exemplars, "similar": similar}

