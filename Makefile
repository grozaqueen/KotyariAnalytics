.PHONY: refresh-trends run-gen run-api

.PHONY: migrate
migrate:
	@set -a; source .env; set +a; \
	psql -h "$$DB_HOST" -p "$$DB_PORT" -d "$$DB_NAME" -U "$$DB_USER" -v ON_ERROR_STOP=1 -f db/migrations/01_pgvector.sql; \
	psql -h "$$DB_HOST" -p "$$DB_PORT" -d "$$DB_NAME" -U "$$DB_USER" -v ON_ERROR_STOP=1 -f db/migrations/02_base_schema.sql; \
	psql -h "$$DB_HOST" -p "$$DB_PORT" -d "$$DB_NAME" -U "$$DB_USER" -v ON_ERROR_STOP=1 -f db/migrations/03_views_trends.sql; \
	psql -h "$$DB_HOST" -p "$$DB_PORT" -d "$$DB_NAME" -U "$$DB_USER" -v ON_ERROR_STOP=1 -f db/migrations/04_indexes.sql

migrate-one:
	@set -a; source .env; set +a; \
	psql -h "$$DB_HOST" -p "$$DB_PORT" -d "$$DB_NAME" -U "$$DB_USER" -v ON_ERROR_STOP=1 -f $(FILE)

refresh-trends:
	python etl/trends/refresh_trend_views.py

run-gen:
	cd apps/generator_service && python main.py "Напиши пост про оптимизацию бюджета на SMM"

run-api:
	cd apps/generator_service && uvicorn api:app --reload --port 8080
