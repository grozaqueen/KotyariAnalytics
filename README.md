### 1) Миграции БД: расширения, таблицы, индексы, материализованные вьюхи.
####    Запускаем ОДИН раз (или когда меняется схема).
``` make migrate ```

### 2) Бэкфилл эмбеддингов постов (topics → post_embeddings).
####    Пройдётся по всем записям без эмбеддингов и досчитает их (SBERT ru).
``` python -m etl.embeddings.backfill_post_embeddings ```

### 3) Кластеризация постов (уменьшение размерности UMAP → HDBSCAN).
####    Перезапишет public.message_clusters и пересчитает центроиды.
``` python -m etl.clustering.run_clustering ```

### 4) Генерация человекочитаемых тем для кластеров через LLM (Grok).
####    Заполняет/обновляет public.cluster_topics_by_section.
``` python -m etl.topics.build_cluster_topics ```

### 5) Эмбеддинги тем (title_emb) для последующего ретривала по темам.
####    Обновляет emb в public.cluster_topics_by_section.title_emb.
``` python -m etl.topics.backfill_title_embeddings ```

### 6) Профиль “стиля” по каждому кластеру (средняя длина, эмодзи, n-граммы, эталоны).
####    Пишет/апдейтит public.cluster_style (теперь с ключом section+cluster_id).
``` python -m etl.style_profile.build_cluster_style ```

### 7) Популярность тем по Wordstat (трафик, ассоциации, топ-запросы).
####    Апсертит в public.topic_popularity_wordstat по (section, cluster_id).
``` python -m etl.wordstat.fetch_wordstat_popularity ```

### 8) Освежить материализованные вьюхи трендов (recency/score).
####    Первый запуск — без CONCURRENTLY, далее — CONCURRENTLY.
``` python -m etl.trends.refresh_trend_views ```

### 9) Запуск генератора: строит контекст (ретрив), подбирает стиль и примеры, генерирует текст.
####    Пример запроса ниже: замените на свой.
``` python -m app.generator_service.main "Напиши пост про популярную породу кошек" ```
