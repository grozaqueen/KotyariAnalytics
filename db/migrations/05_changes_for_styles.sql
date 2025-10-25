-- 1) Добавляем колонку section, если её нет
ALTER TABLE public.cluster_style
  ADD COLUMN IF NOT EXISTS section TEXT;

-- 2) Подтягиваем section из message_clusters (на всякий случай)
UPDATE public.cluster_style cs
SET section = mc.section
FROM public.message_clusters mc
WHERE cs.section IS NULL
  AND cs.cluster_id = mc.cluster_id;

-- 3) Делаем колонки под PK not null (PK не может быть с NULL)
ALTER TABLE public.cluster_style
  ALTER COLUMN cluster_id SET NOT NULL;

UPDATE public.cluster_style SET section = '' WHERE section IS NULL;

ALTER TABLE public.cluster_style
  ALTER COLUMN section SET NOT NULL;

DO $$
DECLARE
  pk_name text;
BEGIN
  SELECT conname
  INTO pk_name
  FROM pg_constraint
  WHERE conrelid = 'public.cluster_style'::regclass
    AND contype  = 'p'
  LIMIT 1;

  IF pk_name IS NOT NULL THEN
    EXECUTE format('ALTER TABLE public.cluster_style DROP CONSTRAINT %I', pk_name);
  END IF;
END$$;

WITH d AS (
  SELECT section, cluster_id,
         ctid,
         ROW_NUMBER() OVER (PARTITION BY section, cluster_id ORDER BY updated_at DESC NULLS LAST, ctid DESC) AS rn
  FROM public.cluster_style
)
DELETE FROM public.cluster_style cs
USING d
WHERE cs.ctid = d.ctid AND d.rn > 1;

-- 6) Создаём составной первичный ключ
ALTER TABLE public.cluster_style
  ADD CONSTRAINT cluster_style_pkey PRIMARY KEY (section, cluster_id);
