-- Векторные индексы
DROP INDEX IF EXISTS public.idx_post_embeddings_hnsw;
CREATE INDEX idx_post_embeddings_hnsw
  ON public.post_embeddings USING hnsw (emb vector_l2_ops);

DROP INDEX IF EXISTS public.idx_cluster_centroids_hnsw;
CREATE INDEX idx_cluster_centroids_hnsw
  ON public.cluster_centroids USING hnsw (centroid vector_l2_ops);

-- Для джоинов и фильтров
DROP INDEX IF EXISTS public.idx_message_clusters_section_cluster;
CREATE INDEX idx_message_clusters_section_cluster
  ON public.message_clusters(section, cluster_id);

DROP INDEX IF EXISTS public.idx_topics_created_at;
CREATE INDEX idx_topics_created_at ON public.topics (created_at DESC);
