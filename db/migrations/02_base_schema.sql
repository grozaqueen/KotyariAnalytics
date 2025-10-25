-- Источник (существующая): public.topics(id, text_clean, section, dt TIMESTAMPTZ)

-- Эмбеддинги постов
DROP TABLE IF EXISTS public.post_embeddings CASCADE;
CREATE TABLE public.post_embeddings (
  post_id    BIGINT PRIMARY KEY REFERENCES public.topics(id) ON DELETE CASCADE,
  emb        vector(1024) NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Кластеризация постов (topic_id ↔ cluster)
DROP TABLE IF EXISTS public.message_clusters CASCADE;
CREATE TABLE public.message_clusters (
  topic_id   BIGINT PRIMARY KEY REFERENCES public.topics(id) ON DELETE CASCADE,
  cluster_id INTEGER NOT NULL,
  section    TEXT    NOT NULL
);

-- Названия тем по кластерам (на секцию)
DROP TABLE IF EXISTS public.cluster_topics_by_section CASCADE;
CREATE TABLE public.cluster_topics_by_section (
  section       TEXT    NOT NULL,
  cluster_id    INTEGER NOT NULL,
  size_raw      INTEGER NOT NULL,
  size_kept     INTEGER NOT NULL,
  cohesion_raw  DOUBLE PRECISION,
  cohesion_kept DOUBLE PRECISION,
  topic         TEXT,
  title_emb     vector(1024),
  exemplars     BIGINT[],
  PRIMARY KEY (section, cluster_id)
);

-- Центроиды (или медоиды) кластеров
DROP TABLE IF EXISTS public.cluster_centroids CASCADE;
CREATE TABLE public.cluster_centroids (
  cluster_id   INTEGER PRIMARY KEY,
  centroid     vector(1024) NOT NULL,
  size         INTEGER NOT NULL,
  updated_at   TIMESTAMPTZ DEFAULT now()
);

-- Профиль стиля кластера
DROP TABLE IF EXISTS public.cluster_style CASCADE;
CREATE TABLE public.cluster_style (
  cluster_id        INTEGER PRIMARY KEY,
  avg_len_tokens    INTEGER,
  p95_len_tokens    INTEGER,
  emoji_rate        REAL,
  exclam_rate       REAL,
  question_rate     REAL,
  markdown_rate     REAL,
  top_ngrams        TEXT[],
  slang_ngrams      TEXT[],
  exemplar_post_ids BIGINT[],
  updated_at        TIMESTAMPTZ DEFAULT now()
);

-- Популярность тем (Wordstat)
DROP TABLE IF EXISTS public.topic_popularity_wordstat CASCADE;
CREATE TABLE public.topic_popularity_wordstat (
  section          TEXT        NOT NULL,
  cluster_id       INTEGER     NOT NULL,
  topic            TEXT        NOT NULL,
  request_phrase   TEXT        NOT NULL,
  total_count_30d  BIGINT      NOT NULL,
  top_max_phrase   TEXT,
  top_max_count    BIGINT,
  top_requests     JSONB,
  associations     JSONB,
  regions_used     INTEGER[],
  devices_used     TEXT[],
  requested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (section, cluster_id)
);
