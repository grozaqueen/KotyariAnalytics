#!/usr/bin/env bash
set -euo pipefail

sleep 2

DB_HOST="${DB_HOST:-}"
DB_PORT="${DB_PORT:-5432}"

# Включаем форвардинг ТОЛЬКО если база локальная (postgres/localhost/127.0.0.1)
if [[ "$DB_HOST" == "postgres" || "$DB_HOST" == "127.0.0.1" || "$DB_HOST" == "localhost" || -z "$DB_HOST" ]]; then
  LISTEN_PORT="$DB_PORT"
  echo "[analytics] forwarding 127.0.0.1:${LISTEN_PORT} -> postgres:5432"
  socat "TCP-LISTEN:${LISTEN_PORT},fork,reuseaddr" "TCP:postgres:5432" &
  SOCAT_PID=$!
  trap 'kill -TERM "$SOCAT_PID" 2>/dev/null || true' EXIT

  # если кто-то подключается к localhost — гарантируем корректный хост
  export DB_HOST="127.0.0.1"
else
  echo "[analytics] using remote Postgres: ${DB_HOST}:${DB_PORT} (no socat)"
fi

echo "[analytics] starting gRPC server..."
exec python -m app.grpc_server
