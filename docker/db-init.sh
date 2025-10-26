#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

echo "Applying SQL from /migrations..."
for f in /migrations/*.sql; do
  echo ">> $f"
  PGPASSWORD="$POSTGRES_PASSWORD" psql \
    -h "$PGHOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 \
    -f "$f"
done

if [ -d /seeds ]; then
  echo "Seeding data from /seeds..."
  for s in /seeds/*.sql; do
    echo ">> $s"
    PGPASSWORD="$POSTGRES_PASSWORD" psql \
      -h "$PGHOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 \
      -f "$s"
  done
fi

echo "Done."
