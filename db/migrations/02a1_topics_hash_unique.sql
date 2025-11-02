DELETE FROM public.topics t
USING public.topics t2
WHERE t.hash = t2.hash AND t.ctid > t2.ctid;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM   pg_constraint
    WHERE  conname = 'ux_topics_hash'
    AND    conrelid = 'public.topics'::regclass
  ) THEN
    ALTER TABLE public.topics
      ADD CONSTRAINT ux_topics_hash UNIQUE (hash);
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_topics_section ON public.topics(section);
