import json
import re
import numpy as np
from collections import Counter, defaultdict
from typing import Iterable, Tuple, List
from sklearn.feature_extraction.text import CountVectorizer
from etl.common.db import get_conn

EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]")
STOP = {
    "и","в","во","на","но","а","как","о","об","от","до","за","из","по","при","к","с","у","же","ли","или",
    "что","это","то","так","там","тут","для","без","про","над","под","между","не","ни","да","ну","же",
    "the","a","an","to","of","in","on","at","for","and","or","but","is","are","be","by","with","as","from"
}
_token_re = re.compile(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9\-]+")

def rate_emoji(s: str) -> float:
    return 0.0 if not s else len(EMOJI.findall(s)) / max(len(s), 1)

def rate_char(s: str, ch: str) -> float:
    return 0.0 if not s else s.count(ch) / max(len(s), 1)

def token_len(s: str) -> int:
    return len((s or "").split())

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

def _normalize_text(t: str) -> str:
    t = (t or "").lower()
    toks = [w for w in _token_re.findall(t) if w not in STOP and len(w) > 2 and not w.isdigit()]
    return " ".join(toks)

def _c_tf_idf(docs: List[str], ngram_range=(1,3), min_df=1, max_df=0.95) -> Tuple[CountVectorizer, np.ndarray]:
    vec = CountVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)
    X = vec.fit_transform(docs)
    tf = X.astype(float)
    tf = tf / np.maximum(tf.sum(axis=1), 1)
    df = (X > 0).sum(axis=0)
    N = X.shape[0]
    idf = np.log((N + 1) / (df + 1)) + 1.0
    c_tfidf = tf.multiply(idf)
    return vec, np.asarray(c_tfidf.todense())

def _dedup_phrases(phrases: List[str], max_k=20, overlap=0.66) -> List[str]:
    keep = []
    seen_sets = []
    for ph in phrases:
        s = set(ph.split())
        if not s:
            continue
        drop = False
        for ss in seen_sets:
            inter = len(s & ss)
            if inter / max(1, min(len(s), len(ss))) >= overlap:
                drop = True
                break
        if not drop:
            keep.append(ph)
            seen_sets.append(s)
        if len(keep) >= max_k:
            break
    return keep

def smart_keyphrases_for_cluster(cluster_texts: List[str], background_texts: Iterable[str] | None = None, topk_phrases: int = 20, topk_unigrams: int = 12) -> Tuple[List[str], List[str]]:
    cl_norm = [_normalize_text(t) for t in cluster_texts if t and t.strip()]
    if not cl_norm:
        return [], []
    if background_texts:
        bg_norm = [_normalize_text(t) for t in background_texts if t and t.strip()]
        if not bg_norm:
            bg_norm = []
        docs = [" ".join(cl_norm), " ".join(bg_norm) if bg_norm else " "]
    else:
        cnt = Counter(" ".join(cl_norm).split())
        bg_fake = " ".join([w for w, c in cnt.items() for _ in range(max(1, c // 5))]) or " "
        docs = [" ".join(cl_norm), bg_fake]
    vec, ctfidf = _c_tf_idf(docs, ngram_range=(1,3), min_df=1, max_df=0.95)
    vocab = np.array(vec.get_feature_names_out())
    scores = np.asarray(ctfidf[0]).ravel()
    order = np.argsort(-scores)
    phrases = vocab[order].tolist()
    phrases = [p for p in phrases if len(p.split()) >= 2][: topk_phrases * 3]
    keyphrases = _dedup_phrases(phrases, max_k=topk_phrases, overlap=0.66)
    uni = [w for w in vocab[order] if (len(w.split()) == 1 and len(w) > 2 and w not in STOP and not w.isdigit())]
    used_tokens = set()
    for ph in keyphrases:
        used_tokens |= set(ph.split())
    salient_unigrams = [w for w in uni if w not in used_tokens][:topk_unigrams]
    return keyphrases, salient_unigrams

def main():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT mc.section, mc.cluster_id, t.id, t.text, pe.emb::text
            FROM public.message_clusters mc
            JOIN public.topics t ON t.id = mc.topic_id
            JOIN public.post_embeddings pe ON pe.post_id = t.id
            WHERE mc.cluster_id <> -1
            ORDER BY mc.section, mc.cluster_id, t.id
        """)
        rows = cur.fetchall()
    by_sc = defaultdict(lambda: {"ids": [], "texts": [], "embs": []})
    section_pairs = defaultdict(list)
    for section, cid, pid, text, emb in rows:
        key = (section or "", int(cid))
        by_sc[key]["ids"].append(int(pid))
        by_sc[key]["texts"].append(text or "")
        by_sc[key]["embs"].append(_to_vec(emb))
        section_pairs[section or ""].append((int(pid), text or ""))
    with get_conn() as conn, conn.cursor() as cur:
        for (section, cid), pack in by_sc.items():
            ids = pack["ids"]
            texts = pack["texts"]
            embs = np.vstack(pack["embs"])
            lens = [token_len(t) for t in texts]
            avg_len = int(np.mean(lens)) if lens else 0
            p95_len = int(np.percentile(lens, 95)) if lens else 0
            emoji_r = float(np.mean([rate_emoji(t) for t in texts])) if texts else 0.0
            excl_r = float(np.mean([rate_char(t, "!") for t in texts])) if texts else 0.0
            qst_r = float(np.mean([rate_char(t, "?") for t in texts])) if texts else 0.0
            md_r = 0.0
            ids_set = set(ids)
            bg = [txt for pid, txt in section_pairs[section] if pid not in ids_set]
            keyphrases, salient = smart_keyphrases_for_cluster(texts, background_texts=bg, topk_phrases=20, topk_unigrams=12)
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
            """, (section, cid, avg_len, p95_len, emoji_r, excl_r, qst_r, md_r, keyphrases, salient, exemplars))
        conn.commit()
    print("Профили стиля (section+cluster) обновлены.")

if __name__ == "__main__":
    main()
