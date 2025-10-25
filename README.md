make migrate
python -m etl.embeddings.backfill_post_embeddings
python -m etl.clustering.run_clustering
python -m etl.topics.build_cluster_topics
python -m etl.topics.backfill_title_embeddings
python -m etl.style_profile.build_cluster_style
python -m etl.wordstat.fetch_wordstat_popularity
python -m etl.trends.refresh_trend_views
python -m app.generator_service.main


python etl/embeddings/backfill_post_embeddings.py

python etl/clustering/run_clustering.py

python etl/topics/build_cluster_topics.py

python etl/topics/backfill_title_embeddings.py

python etl/style_profile/build_cluster_style.py

python etl/wordstat/fetch_wordstat_popularity.py

python etl/trends/refresh_trend_views.py

python apps/generator_service/main.py "Напиши пост про ..."