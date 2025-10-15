import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import time
import socket
from typing import Dict, List

import psycopg2
from dotenv import load_dotenv

load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4-fast-non-reasoning")

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        sslmode="disable",
        connect_timeout=5,
    )

import httpx
from openai import OpenAI

_http_client = None
_xai_client = None

def _is_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False

def _detect_proxy_url() -> str | None:
    # Предпочтение SOCKS (с DNS через прокси), затем HTTP
    if _is_port_open("127.0.0.1", 53425):
        return "socks5h://127.0.0.1:53425"
    if _is_port_open("127.0.0.1", 53424):
        return "http://127.0.0.1:53424"
    return None

def _get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        proxy = _detect_proxy_url()
        _http_client = httpx.Client(proxy=proxy, timeout=30.0, trust_env=False)
    return _http_client

def _get_grok_client() -> OpenAI:
    global _xai_client
    if _xai_client is None:
        if not XAI_API_KEY:
            raise RuntimeError("XAI_API_KEY is not set in environment")
        _xai_client = OpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1",
            http_client=_get_http_client(),
        )
    return _xai_client

_CAP_SEQ = re.compile(
    r"(?:\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,3}\b|"
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b)"
)

def _extract_capitalized_phrases(raw_text: str) -> list[str]:
    out = []
    for m in _CAP_SEQ.finditer(raw_text or ""):
        ph = m.group(0).strip()
        if 1 <= len(ph.split()) <= 5:
            out.append(ph)
    return out

def _clean_topic(s: str) -> str:
    s = (s or "").strip().strip('“”"\' \n\t.')
    if len(s) > 120:
        s = s[:120].rsplit(" ", 1)[0]
    return s[:1].upper() + s[1:] if s else s

def generate_single_topic_with_grok(examples: list[str]) -> str:

    client = _get_grok_client()

    ex = [e.strip() for e in examples if (e and e.strip())][:12]
    joined_examples = "\n".join(f"- {t[:700]}" for t in ex) or "-"

    capitalized_hints: list[str] = []
    for t in ex:
        capitalized_hints.extend(_extract_capitalized_phrases(t))
    hints = ", ".join(capitalized_hints[:12]) or "-"

    prompt = (
        "Определи одну простую и понятную тему для набора текстов.\n"
        "Формулировка: 3–10 слов, по-русски, разговорно-поисковый стиль.\n"
        "Избегай слов: инструкция, статья, текст, кластер. Без кавычек.\n\n"
        "Примеры текстов:\n{EX}\n\n"
        "Подсказки по именам собственным/терминам: {H}\n\n"
        "Верни только саму фразу темы, без пояснений."
    ).format(EX=joined_examples, H=hints)

    last_err = None
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=GROK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=64,
            )
            topic = (resp.choices[0].message.content or "").strip()
            topic = _clean_topic(topic)
            if topic:
                return topic
        except Exception as e:
            last_err = e
            time.sleep(1.2)
    print("[WARN] Grok single-topic failed:", last_err)
    return "Тема не определена"

def compute_section_cluster_topics_simple(
    sec_records: List[tuple[int, int, str]],  # (topic_id, cluster_id, text)
) -> List[dict]:

    if not sec_records:
        return []

    # группируем по cluster_id
    by_cluster: Dict[int, List[int]] = {}
    texts: List[str] = []
    ids: List[int] = []

    for idx, (topic_id, cluster_id, text) in enumerate(sec_records):
        if cluster_id == -1:
            continue
        by_cluster.setdefault(cluster_id, []).append(idx)
        texts.append(text or "")
        ids.append(topic_id)

    out_rows: List[dict] = []

    for cid in sorted(by_cluster.keys()):
        idxs = by_cluster[cid]
        size_raw = len(idxs)
        size_kept = size_raw
        exemplars = [sec_records[i][0] for i in idxs[:3]]  # первые 3 id

        example_texts = [sec_records[i][2] for i in idxs[:10]]
        try:
            topic = generate_single_topic_with_grok(example_texts)
        except Exception as e:
            print(f"[WARN] Grok failed for cluster {cid}:", e)
            topic = "Тема не определена"

        out_rows.append({
            "section": None,
            "cluster_id": int(cid),
            "size_raw": int(size_raw),
            "size_kept": int(size_kept),
            "cohesion_raw": 0.0,
            "cohesion_kept": 0.0,
            "topic": topic,
            "exemplars": exemplars,
        })

    return out_rows

def save_topics_by_section(conn, section: str, rows: List[dict]):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS public.cluster_topics_by_section (
                section       TEXT NOT NULL,
                cluster_id    INTEGER NOT NULL,
                size_raw      INTEGER NOT NULL,
                size_kept     INTEGER NOT NULL,
                cohesion_raw  DOUBLE PRECISION,
                cohesion_kept DOUBLE PRECISION,
                topic         TEXT,
                exemplars     INTEGER[],
                PRIMARY KEY (section, cluster_id)
            );
        """)
        cur.execute("DELETE FROM public.cluster_topics_by_section WHERE section = %s;", (section,))
        for r in rows:
            cur.execute("""
                INSERT INTO public.cluster_topics_by_section
                (section, cluster_id, size_raw, size_kept, cohesion_raw, cohesion_kept, topic, exemplars)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                section,
                r["cluster_id"],
                r["size_raw"],
                r["size_kept"],
                r["cohesion_raw"],
                r["cohesion_kept"],
                r["topic"],
                r["exemplars"],
            ))
        conn.commit()

def build_and_save_topics_for_all_sections():
    if not XAI_API_KEY:
        raise RuntimeError("XAI_API_KEY is required for Grok")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT topic_id, section, cluster_id, text
                FROM public.message_clusters
                WHERE cluster_id <> -1
                ORDER BY section, cluster_id, topic_id
            """)
            rows = cur.fetchall()

        by_section: Dict[str, List[tuple[int, int, str]]] = {}
        for topic_id, section, cluster_id, text in rows:
            by_section.setdefault(section, []).append((topic_id, cluster_id, text))

        for section, sec_records in by_section.items():
            print(f"\n=== Section: {section} | docs={len(sec_records)} ===")
            topics = compute_section_cluster_topics_simple(sec_records=sec_records)
            for t in topics:
                t["section"] = section
            save_topics_by_section(conn, section, topics)

            # превью
            preview = sorted(topics, key=lambda x: -x["size_kept"])[:8]
            for t in preview:
                print(f"  cluster {t['cluster_id']:>4} | kept={t['size_kept']:>3}/{t['size_raw']:>3} "
                      f"| topic: {t['topic']}")
    except Exception as e:
        print("Ошибка:", e)
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    build_and_save_topics_for_all_sections()
