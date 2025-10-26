#!/usr/bin/env bash
set -e

sleep 2

LISTEN_PORT="${DB_PORT:-5432}"
echo "[analytics] forwarding 127.0.0.1:${LISTEN_PORT} -> postgres:5432"
socat TCP-LISTEN:${LISTEN_PORT},fork,reuseaddr TCP:postgres:5432 &
SOCAT_PID=$!
trap "kill -TERM $SOCAT_PID 2>/dev/null || true" EXIT

echo "[analytics] starting gRPC server..."
exec python -m app.grpc_server
