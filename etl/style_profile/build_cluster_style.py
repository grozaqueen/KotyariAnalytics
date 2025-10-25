import json
import re
import numpy as np
from collections import Counter, defaultdict
from etl.common.db import get_conn

EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]")

def rate_emoji(s: str) -> float:
    return 0.0 if not s else len(EMOJI.findall(s)) / max(len(s), 1)

def rate_char(s: str, ch: str) -> float:
    return 0.0 if not s else s.count(ch) / max(len(s), 1)

def token_len(s: str) -> int:
    return len((s or "").split())

def top_ngrams(texts: list[str], n=2, k=20):
    cnt = Counter()
    for t in texts:
        toks = [x for x in re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", (t or "").lower()) if len(x) > 1]
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            cnt[" ".join(toks[i:i+n])] += 1
    return [w for w, _ in cnt.most_common(k)]

def _to_vec(x) -> np.ndarray:
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype="float32")
    if isinstance(x, memoryview):
        x = x.tobytes().decode("utf-8", errors="ignore")
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        return np.asarray(json.loads(x), dtype="float32")
    raise TypeError(f"Unexpected emb type: {type(x)}")

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n = np.maximum(n, 1e-8)
    return mat / n

def pick_exemplars(ids: list[int], embs: np.ndarray, k=5):
    embs_n = l2_normalize(embs)
    c = embs_n.mean(axis=0)
    c = c / max(np.linalg.norm(c), 1e-8)
    sims = embs_n @ c
    order = np.argsort(-sims)
    return [int(ids[i]) for i in order[:k]]

def main():
    # 1) читаем посты с секцией
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT mc.section, mc.cluster_id, t.id, t.text_clean, pe.emb::text
            FROM public.message_clusters mc
            JOIN public.topics t ON t.id = mc.topic_id
            JOIN public.post_embeddings pe ON pe.post_id = t.id
            WHERE mc.cluster_id <> -1
            ORDER BY mc.section, mc.cluster_id, t.id
        """)
        rows = cur.fetchall()

    # 2) группируем по (section, cluster_id)
    by_sc: dict[tuple[str, int], dict[str, list]] = defaultdict(lambda: {"ids": [], "texts": [], "embs": []})
    for section, cid, pid, text, emb in rows:
        key = (section or "", int(cid))
        by_sc[key]["ids"].append(int(pid))
        by_sc[key]["texts"].append(text or "")
        by_sc[key]["embs"].append(_to_vec(emb))

    print(f"Связок (section, cluster_id): {len(by_sc)}")

    # 3) считаем метрики и пишем с upsert по (section, cluster_id)
    with get_conn() as conn, conn.cursor() as cur:
        for (section, cid), pack in by_sc.items():
            ids = pack["ids"]
            texts = pack["texts"]
            embs = np.vstack(pack["embs"])

            lens = [token_len(t) for t in texts]
            avg_len = int(np.mean(lens)) if lens else 0
            p95_len = int(np.percentile(lens, 95)) if lens else 0
            emoji_r = float(np.mean([rate_emoji(t) for t in texts])) if texts else 0.0
            excl_r  = float(np.mean([rate_char(t, "!") for t in texts])) if texts else 0.0
            qst_r   = float(np.mean([rate_char(t, "?") for t in texts])) if texts else 0.0
            md_r    = 0.0

            top2 = top_ngrams(texts, n=2, k=20)
            slang = top_ngrams(texts, n=1, k=10)
            exemplars = pick_exemplars(ids, embs, k=5)

            cur.execute("""
                INSERT INTO public.cluster_style
                (section, cluster_id, avg_len_tokens, p95_len_tokens, emoji_rate, exclam_rate, question_rate,
                 markdown_rate, top_ngrams, slang_ngrams, exemplar_post_ids, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                ON CONFLICT (section, cluster_id) DO UPDATE SET
                  avg_len_tokens      = EXCLUDED.avg_len_tokens,
                  p95_len_tokens      = EXCLUDED.p95_len_tokens,
                  emoji_rate          = EXCLUDED.emoji_rate,
                  exclam_rate         = EXCLUDED.exclam_rate,
                  question_rate       = EXCLUDED.question_rate,
                  markdown_rate       = EXCLUDED.markdown_rate,
                  top_ngrams          = EXCLUDED.top_ngrams,
                  slang_ngrams        = EXCLUDED.slang_ngrams,
                  exemplar_post_ids   = EXCLUDED.exemplar_post_ids,
                  updated_at          = now()
            """, (section, cid, avg_len, p95_len, emoji_r, excl_r, qst_r, md_r, top2, slang, exemplars))
        conn.commit()
    print("Профили стиля (section+cluster) обновлены.")

if __name__ == "__main__":
    main()
