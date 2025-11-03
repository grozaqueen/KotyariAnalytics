import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import time
from typing import Dict, List

from dotenv import load_dotenv
from etl.common.db import get_conn
from etl.common.grok_client import get_grok_client, XAI_API_KEY, GROK_MODEL

load_dotenv()

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
    client = get_grok_client()

    ex = [e.strip() for e in examples if (e and e.strip())][:12]
    joined_examples = "\n".join("- " + t[:700] for t in ex) or "-"

    capitalized_hints: list[str] = []
    for t in ex:
        capitalized_hints.extend(_extract_capitalized_phrases(t))
    hints = ", ".join(capitalized_hints[:12]) or "-"

    prompt = (
        "Определи одну простую и понятную тему для набора текстов.\n"
        "Формулировка: 3–10 слов, по-русски, разговорно-поисковый стиль.\n"
        "Избегай слов: инструкция, статья, текст, кластер. Без кавычек.\n\n"
        f"Примеры текстов:\n{joined_examples}\n\n"
        f"Подсказки по именам собственным/терминам: {hints}\n\n"
        "Верни только саму фразу темы, без пояснений."
    )

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
    sec_records: List[tuple[int, int, str]],
) -> List[dict]:
    if not sec_records:
        return []

    by_cluster: Dict[int, List[int]] = {}
    for idx, (topic_id, cluster_id, text) in enumerate(sec_records):
        if cluster_id == -1:
            continue
        by_cluster.setdefault(cluster_id, []).append(idx)

    out_rows: List[dict] = []
    for cid in sorted(by_cluster.keys()):
        idxs = by_cluster[cid]
        size_raw = len(idxs)
        size_kept = size_raw
        exemplars = [sec_records[i][0] for i in idxs[:3]]

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

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT mc.topic_id, mc.section, mc.cluster_id, t.text
                FROM public.message_clusters mc
                JOIN public.topics t ON t.id = mc.topic_id
                WHERE mc.cluster_id <> -1
                ORDER BY mc.section, mc.cluster_id, mc.topic_id
            """)
            rows = cur.fetchall()

        by_section: Dict[str, List[tuple[int, int, str]]] = {}
        for topic_id, section, cluster_id, text in rows:
            by_section.setdefault(section or "", []).append((int(topic_id), int(cluster_id), text or ""))

        for section, sec_records in by_section.items():
            print(f"\n=== Section: {section} | docs={len(sec_records)} ===")
            topics = compute_section_cluster_topics_simple(sec_records=sec_records)
            for t in topics:
                t["section"] = section
            save_topics_by_section(conn, section, topics)

            preview = sorted(topics, key=lambda x: -x["size_kept"])[:8]
            for t in preview:
                print(f"  cluster {t['cluster_id']:>4} | kept={t['size_kept']:>3}/{t['size_raw']:>3} | topic: {t['topic']}")
    except Exception as e:
        print("Ошибка:", e)
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    build_and_save_topics_for_all_sections()
