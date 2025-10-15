import os
import re
import time
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
import httpx

load_dotenv()

WORDSTAT_BASE = "https://api.wordstat.yandex.net"
TOP_REQUESTS_URL = f"{WORDSTAT_BASE}/v1/topRequests"
USERINFO_URL = f"{WORDSTAT_BASE}/v1/userInfo"

OAUTH = os.getenv("YANDEX_OAUTH_TOKEN")
CLIENT_ID = os.getenv("YANDEX_CLIENT_ID")  # для логов/отладки
if not OAUTH:
    raise RuntimeError("Set YANDEX_OAUTH_TOKEN in environment")

def _parse_csv_ints(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out or None

REGIONS = _parse_csv_ints(os.getenv("WORDSTAT_REGIONS"))
DEVICES_ENV = os.getenv("WORDSTAT_DEVICES", "all")
DEVICES = [d.strip() for d in DEVICES_ENV.split(",") if d.strip()] or ["all"]
NUM_PHRASES = int(os.getenv("WORDSTAT_NUM_PHRASES", "50"))

BASE_DELAY = 0.12  # ~8-9 rps
MAX_RETRIES = 5

PROXY = os.getenv("ALL_PROXY")

def _make_http_client() -> httpx.Client:
    headers = {
        "Authorization": f"Bearer {OAUTH}",
        "Content-Type": "application/json; charset=utf-8",
        "X-Client-Id": CLIENT_ID or "cluster-topics-fetcher",
    }
    client_kwargs = dict(
        timeout=30.0,
        headers=headers,
        follow_redirects=True,
        trust_env=(not PROXY),
    )
    try:
        import h2  # noqa: F401
        client_kwargs["http2"] = True
    except Exception:
        pass

    if PROXY:
        transport = httpx.HTTPTransport(proxy=PROXY)
        return httpx.Client(transport=transport, **client_kwargs)
    return httpx.Client(**client_kwargs)

http_client = _make_http_client()

def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        sslmode="disable",
        connect_timeout=5,
    )

DDL = """
CREATE TABLE IF NOT EXISTS public.topic_popularity_wordstat (
  section          TEXT        NOT NULL,
  cluster_id       INTEGER     NOT NULL,
  topic            TEXT        NOT NULL,
  request_phrase   TEXT        NOT NULL,   -- фактически отправленная в Wordstat фраза
  total_count_30d  BIGINT      NOT NULL,   -- Wordstat.totalCount (последние 30 дней)
  top_max_phrase   TEXT,
  top_max_count    BIGINT,
  top_requests     JSONB,                  -- массив объектов {phrase,count}
  associations     JSONB,                  -- массив объектов {phrase,count}
  regions_used     INTEGER[],              -- какие регионы учитывали
  devices_used     TEXT[],                 -- какие девайсы учитывали
  requested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (section, cluster_id)
);
"""

UPSERT = """
INSERT INTO public.topic_popularity_wordstat
  (section, cluster_id, topic, request_phrase, total_count_30d,
   top_max_phrase, top_max_count, top_requests, associations,
   regions_used, devices_used, requested_at)
VALUES
  (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW())
ON CONFLICT (section, cluster_id) DO UPDATE SET
  topic            = EXCLUDED.topic,
  request_phrase   = EXCLUDED.request_phrase,
  total_count_30d  = EXCLUDED.total_count_30d,
  top_max_phrase   = EXCLUDED.top_max_phrase,
  top_max_count    = EXCLUDED.top_max_count,
  top_requests     = EXCLUDED.top_requests,
  associations     = EXCLUDED.associations,
  regions_used     = EXCLUDED.regions_used,
  devices_used     = EXCLUDED.devices_used,
  requested_at     = NOW();
"""

SELECT_TOPICS = """
SELECT section, cluster_id, topic
FROM public.cluster_topics_by_section
WHERE topic IS NOT NULL AND topic <> ''
ORDER BY section, cluster_id;
"""

# -------------------- phrase sanitation --------------------
_WS_ALLOWED = re.compile(r'[A-Za-zА-Яа-яЁё0-9\s\+\-\|\!\[\]\(\)"«»]', re.UNICODE)

def _collapse_spaces(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def sanitize_phrase_for_wordstat(raw: str) -> str:
    if not raw:
        return ""
    s = re.sub(r'https?://\S+|www\.\S+', ' ', raw, flags=re.IGNORECASE)
    s = ''.join(ch if _WS_ALLOWED.match(ch) else ' ' for ch in s)
    s = _collapse_spaces(s)
    return s[:256]

def simplify_phrase(raw: str) -> str:
    s = re.sub(r'https?://\S+|www\.\S+', ' ', raw, flags=re.IGNORECASE)
    s = re.sub(r'[^A-Za-zА-Яа-яЁё0-9\s]', ' ', s)
    return _collapse_spaces(s)[:256]

# -------------------- optional: quick capability check --------------------
def _check_user_info_or_raise():
    resp = http_client.post(USERINFO_URL)
    if resp.status_code != 200:
        body = (resp.text or "")[:500]
        req_id = resp.headers.get("x-request-id") or resp.headers.get("X-Request-Id")
        raise RuntimeError(f"userInfo error {resp.status_code}: {body} (request-id={req_id})")

# -------------------- Wordstat call (topRequests) --------------------
def wordstat_top_requests(original_phrase: str) -> Dict[str, Any]:

    clean = sanitize_phrase_for_wordstat(original_phrase)
    if not clean:
        raise RuntimeError("Empty phrase after sanitization")

    payload: Dict[str, Any] = {"phrase": clean}
    if REGIONS:
        payload["regions"] = REGIONS
    if DEVICES:
        payload["devices"] = DEVICES
    if NUM_PHRASES:
        payload["numPhrases"] = min(max(NUM_PHRASES, 1), 2000)

    delay = BASE_DELAY
    last_err: Optional[Exception] = None
    tried_simplified = False

    for _ in range(1, MAX_RETRIES + 1):
        try:
            resp = http_client.post(TOP_REQUESTS_URL, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                data["__request_phrase"] = payload["phrase"]
                return data

            req_id = resp.headers.get("x-request-id") or resp.headers.get("X-Request-Id")
            body = (resp.text or "")[:500]

            if resp.status_code == 400:
                if not tried_simplified:
                    simple = simplify_phrase(original_phrase)
                    if simple and simple != payload["phrase"]:
                        payload["phrase"] = simple
                        tried_simplified = True
                        continue
                raise RuntimeError(
                    f"Wordstat error 400 Invalid query. phrase='{payload.get('phrase')}' "
                    f"body='{body}' request-id={req_id}"
                )

            if resp.status_code in (429, 503):
                time.sleep(delay)
                delay = min(delay * 2.0, 10.0)
                continue

            raise RuntimeError(f"Wordstat error {resp.status_code}: {body} (request-id={req_id})")

        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2.0, 10.0)

    raise RuntimeError(f"topRequests failed after {MAX_RETRIES} attempts. Last error: {last_err}")

def main():
    _check_user_info_or_raise()

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
            conn.commit()

        with conn.cursor() as cur:
            cur.execute(SELECT_TOPICS)
            rows = cur.fetchall()

        total = len(rows)
        if total == 0:
            print("Нет тем в cluster_topics_by_section.")
            return

        print(f"Тем для запроса: {total}. Регионов: {len(REGIONS) if REGIONS else 'ALL'}, devices={DEVICES}")

        done = 0
        for section, cluster_id, topic in rows:
            original = (topic or "").strip()
            if not original:
                continue

            data = wordstat_top_requests(original)

            request_phrase = data.get("__request_phrase") or data.get("requestPhrase") or original
            total_count = int(data.get("totalCount", 0) or 0)
            top_requests = data.get("topRequests", []) or []
            associations = data.get("associations", []) or []

            top_max_phrase = None
            top_max_count = None
            if top_requests:
                top_sorted = sorted(top_requests, key=lambda x: int(x.get("count", 0)), reverse=True)
                top_max_phrase = top_sorted[0].get("phrase")
                top_max_count = int(top_sorted[0].get("count", 0))

            with conn.cursor() as cur:
                cur.execute(
                    UPSERT,
                    (
                        section,
                        int(cluster_id),
                        topic,
                        request_phrase,
                        total_count,
                        top_max_phrase,
                        top_max_count if top_max_count is not None else None,
                        Json(top_requests),
                        Json(associations),
                        REGIONS,
                        DEVICES,
                    ),
                )
            conn.commit()

            done += 1
            if done % 20 == 0 or done == total:
                print(f"[{done}/{total}] {section}:{cluster_id} | “{request_phrase}” => total_count_30d={total_count}")

            time.sleep(BASE_DELAY)

        print("Готово.")

    except Exception as e:
        print("Ошибка:", e)
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()
