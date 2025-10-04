from typing import Dict, List, Union, Optional
import os
from transformers import pipeline

# ENV:
#   CONTENT_MOD_DEBUG=1                 — отладочный вывод
#   CONTENT_MOD_EXT_THR=0.60            — порог для extremism_support
#   CONTENT_MOD_ILL_THR=0.60            — порог для illegal_content

DEBUG = os.getenv("CONTENT_MOD_DEBUG", "0") == "1"
EXT_THR = float(os.getenv("CONTENT_MOD_EXT_THR", "0.60"))
ILL_THR = float(os.getenv("CONTENT_MOD_ILL_THR", "0.60"))

def _dbg(*args):
    if DEBUG:
        print("[ML]", *args)

def _normalize_out(res: Union[dict, List[dict], List[List[dict]]]) -> Dict[str, float]:
    """Приводим выход pipeline к виду {label_lower: score} для разных форматов."""
    if isinstance(res, dict) and "label" in res and "score" in res:
        items = [res]
    elif isinstance(res, list):
        items = res[0] if res and isinstance(res[0], list) else res
    else:
        return {}
    out: Dict[str, float] = {}
    for it in items:
        try:
            lab = str(it["label"]).lower()
            sc = float(it["score"])
            out[lab] = sc
        except Exception:
            continue
    return out

_TOXICITY_PIPE: Optional[object] = None
_ZS_PIPE: Optional[object] = None

def _tox():
    """Русская токс-модель (ruBERT tiny toxicity)."""
    global _TOXICITY_PIPE
    if _TOXICITY_PIPE is None:
        _TOXICITY_PIPE = pipeline(
            task="text-classification",
            model="cointegrated/rubert-tiny-toxicity",
        )
        _dbg("TOX model loaded:", getattr(_TOXICITY_PIPE.model.config, "id2label", None))
    return _TOXICITY_PIPE

def _zs():
    global _ZS_PIPE
    if _ZS_PIPE is None:
        _ZS_PIPE = pipeline(
            "zero-shot-classification",
            model="cointegrated/rubert-base-cased-nli-threeway",
        )
        _dbg("ZS model loaded (ruBERT NLI)")
    return _ZS_PIPE

ALL_LABELS = ("profanity", "extremism_support", "illegal_content")

ZS_LABELS_EXT = [
    "экстремизм", "терроризм", "призыв к насилию", "радикальные призывы", "угроза расправы",
]
ZS_LABELS_ILL = [
    "незаконная торговля оружием", "наркотики", "поддельные документы", "контрабанда",
    "детская эксплуатация", "изготовление взрывчатки", "планирование теракта",
]

def ml_scores(text: str) -> Dict[str, float]:
    """
      profanity — токсичность/брань/оскорбления (ruBERT tiny toxicity)
      extremism_support — экстремизм/призыв к насилию/террор (zero-shot, ru NLI)
      illegal_content — оружие/наркотики/взрывчатка/фальш.документы/дет.экспл. (zero-shot, ru NLI)
    """
    scores = {k: 0.0 for k in ALL_LABELS}
    if not text or not text.strip():
        return scores

    try:
        tox = _tox()
        res = tox(text, truncation=True, max_length=256, return_all_scores=True)
        by = _normalize_out(res)
        if not by and hasattr(tox.model, "config"):
            # fallback для старых версий
            res2 = tox(text, truncation=True, max_length=256)
            by = _normalize_out(res2)
            id2 = getattr(tox.model.config, "id2label", None)
            if isinstance(id2, dict):
                by = {
                    (str(id2.get(int(k.split("_")[-1]), k)).lower() if k.startswith("label_") else k): v
                    for k, v in by.items()
                }
        _dbg("TOX labels:", by)
        scores["profanity"] = max(
            by.get("toxic", 0.0), by.get("obscenity", 0.0), by.get("obscene", 0.0),
            by.get("insult", 0.0), by.get("threat", 0.0), by.get("dangerous", 0.0),
            by.get("profanity", 0.0), by.get("abuse", 0.0),
        )
    except Exception as e:
        _dbg("TOX error:", e)

    try:
        zs = _zs()
        zx1 = zs(
            text,
            candidate_labels=ZS_LABELS_EXT,
            hypothesis_template="Этот текст про {}.",
            multi_label=True,
        )
        zx2 = zs(
            text,
            candidate_labels=ZS_LABELS_ILL,
            hypothesis_template="Этот текст про {}.",
            multi_label=True,
        )

        p_ext = max(zx1.get("scores", [0.0]) or [0.0])
        p_ill = max(zx2.get("scores", [0.0]) or [0.0])

        # Порогование для отсечки ложноположительных
        p_ext_post = p_ext if p_ext >= EXT_THR else 0.0
        p_ill_post = p_ill if p_ill >= ILL_THR else 0.0

        _dbg("ZS ext/ill (raw):", p_ext, p_ill)
        _dbg("ZS ext/ill (post-thr):", p_ext_post, p_ill_post)

        scores["extremism_support"] = float(p_ext_post)
        # экстремизм считаем частью illegal: берём максимум из illegal и extremism
        scores["illegal_content"]   = float(max(p_ill_post, p_ext_post))
    except Exception as e:
        _dbg("ZS error:", e)

    return scores
