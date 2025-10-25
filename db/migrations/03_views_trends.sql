-- Свежесть по кластерам за 7/30 дней
DROP MATERIALIZED VIEW IF EXISTS public.mv_cluster_recency CASCADE;
CREATE MATERIALIZED VIEW public.mv_cluster_recency AS
SELECT
  mc.section,
  mc.cluster_id,
COUNT(*) FILTER (WHERE t.created_at >= now() - interval '7 days')  AS cnt_7d,
COUNT(*) FILTER (WHERE t.created_at >= now() - interval '30 days') AS cnt_30d
FROM public.message_clusters mc
JOIN public.topics t ON t.id = mc.topic_id
GROUP BY 1, 2
WITH NO DATA;

-- уникальный индекс для CONCURRENTLY
DROP INDEX IF EXISTS public.mv_cluster_recency_pk;
CREATE UNIQUE INDEX mv_cluster_recency_pk
  ON public.mv_cluster_recency(section, cluster_id);

-- Интегральный тренд: Wordstat + свежесть
DROP MATERIALIZED VIEW IF EXISTS public.mv_cluster_trend_score CASCADE;
CREATE MATERIALIZED VIEW public.mv_cluster_trend_score AS
SELECT
  ws.section,
  ws.cluster_id,
  ws.total_count_30d,
  COALESCE(r.cnt_7d, 0)  AS cnt_7d,
  COALESCE(r.cnt_30d, 0) AS cnt_30d,
  (
    LEAST(LOG(1+ws.total_count_30d)/10.0, 1.0)*0.7
  + LEAST(LOG(1+COALESCE(r.cnt_7d, 0)), 1.0)*0.2
  + LEAST(LOG(1+COALESCE(r.cnt_30d,0)),1.0)*0.1
  )::float AS trend_score
FROM public.topic_popularity_wordstat ws
LEFT JOIN public.mv_cluster_recency r
  ON r.section = ws.section AND r.cluster_id = ws.cluster_id
WITH NO DATA;

DROP INDEX IF EXISTS public.mv_cluster_trend_score_pk;
CREATE UNIQUE INDEX mv_cluster_trend_score_pk
  ON public.mv_cluster_trend_score(section, cluster_id);
