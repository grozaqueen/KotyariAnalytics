DROP TABLE IF EXISTS staging_topics;
CREATE TEMP TABLE staging_topics(
  id BIGINT,
  source TEXT,
  "text" TEXT,
  hash TEXT,
  created_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ,
  section TEXT,
  text_clean TEXT
);

\copy staging_topics (id,source,"text",hash,created_at,updated_at,section,text_clean) FROM '/seeds/topics.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '"');

INSERT INTO public.topics (id,source,"text",hash,created_at,updated_at,section,text_clean)
SELECT id,source,"text",hash,created_at,updated_at,section,text_clean
FROM staging_topics
ON CONFLICT (hash) DO NOTHING;

SELECT setval(
  pg_get_serial_sequence('public.topics','id'),
  GREATEST(1, COALESCE((SELECT MAX(id) FROM public.topics), 0)),
  true
);
