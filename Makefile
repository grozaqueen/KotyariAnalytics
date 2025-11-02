SHELL := /bin/bash

migrate:
	@docker compose run --rm db-init

etl-once:
	docker compose exec analytics bash -c " \
	set -euo pipefail && \
	echo '[0/1] DB ping' && \
	curl --proxy 'xray-proxy:8100' -sS https://httpbin.org/ip && \
	python -c \"import os, psycopg2; conn=psycopg2.connect(host=os.getenv('DB_HOST','127.0.0.1'),port=int(os.getenv('DB_PORT','5432')),dbname=os.getenv('DB_NAME'),user=os.getenv('DB_USER'),password=os.getenv('DB_PASSWORD')); cur=conn.cursor(); cur.execute('SELECT 1'); print('DB OK:', cur.fetchone()[0]); conn.close()\" && \
	echo '[1] embeddings' && python -m etl.embeddings.backfill_post_embeddings && \
	echo '[2] clustering' && python -m etl.clustering.run_clustering && \
	echo '[3] topics via LLM' && python -m etl.topics.build_cluster_topics && \
	echo '[4] title embeddings' && python -m etl.topics.backfill_title_embeddings && \
	echo '[5] style profile' && python -m etl.style_profile.build_cluster_style && \
	if [[ -n \"\$${YANDEX_OAUTH_TOKEN:-}\" && -n \"\$${YANDEX_CLIENT_ID:-}\" ]]; then \
	     echo '[6] wordstat'; \
	     python -m etl.wordstat.fetch_wordstat_popularity; \
	   else echo '[6] wordstat — skipped'; fi && \
	echo '[7] refresh trend MVs' && python -m etl.trends.refresh_trend_views && \
	echo '=== ETL DONE ===' \
	"

grpc-smoke:
	@docker run --rm --network kotyari-net -v "$$PWD/api/protos:/protos:ro" \
	  fullstorydev/grpcurl -plaintext -import-path /protos -proto posts/posts.proto \
	  -d "{\"user_prompt\":\"мейнкун\",\"profile_prompt\":\"\",\"bot_prompt\":\"\"}" \
	  analytics:50051 posts.PostsService/GetPost
