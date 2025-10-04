import os
import ahocorasick
from functools import lru_cache
from typing import Dict, List, Set, Tuple
from .normalization import normalize_surface, normalize_compact, morph


def read_list(path: str) -> List[str]:
    res: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                res.append(t)
    return res


@lru_cache(maxsize=50_000)
def _parse_first(word: str):
    return morph.parse(word)[0]


def expand_lexemes(lemma: str) -> Set[str]:
    lemma = lemma.strip()
    if not lemma:
        return set()

    if " " in lemma:
        return {lemma}

    p = _parse_first(lemma)
    forms: Set[str] = set()
    for f in p.lexeme:
        w = f.word.lower()
        forms.add(w)
        forms.add(w.replace("ё", "е"))
    return forms


def _build_ac(pairs: List[Tuple[str, str]]) -> ahocorasick.Automaton:
    """
    pairs: список (key, payload), где key — нормализованный surface-ключ,
    payload — исходная лемма/термин для отчёта.
    """
    A = ahocorasick.Automaton()
    seen: Set[str] = set()
    for key, payload in pairs:
        if not key:
            continue
        if key in seen:
            continue
        A.add_word(key, payload)
        seen.add(key)
    A.make_automaton()
    return A


def build_dual_automata(lemmas: List[str], *, norm_to: str = "cyr") -> Tuple[ahocorasick.Automaton, ahocorasick.Automaton]:
    """
    Возвращает пару автоматов (AC_spaced, AC_compact):
      - spaced: normalize_surface(…)  — сохраняет пробелы
      - compact: normalize_compact(…) — удаляет разделители
    """
    spaced_pairs: List[Tuple[str, str]] = []
    compact_pairs: List[Tuple[str, str]] = []

    for lemma in lemmas:
        lemma = lemma.strip()
        if not lemma:
            continue

        forms = expand_lexemes(lemma)

        for form in forms:
            spaced_pairs.append((normalize_surface(form, to=norm_to), lemma))

        if " " not in lemma:
            for form in forms:
                compact_pairs.append((normalize_compact(form, to=norm_to), lemma))

    return _build_ac(spaced_pairs), _build_ac(compact_pairs)


def build_ac_surface(lemmas: List[str], *, norm_to: str = "cyr") -> ahocorasick.Automaton:
    """
    Backward-compat: один автомат по surface-ключам (без compact-логики).
    Полезно для старых демо. Для боевого поиска лучше используйте build_dual_automata().
    """
    pairs: List[Tuple[str, str]] = []
    for lemma in lemmas:
        lemma = lemma.strip()
        if not lemma:
            continue
        for form in expand_lexemes(lemma):
            pairs.append((normalize_surface(form, to=norm_to), lemma))
    return _build_ac(pairs)


def load_automata(dict_dir: str, norm_to: str = "cyr") -> Dict[str, Dict[str, ahocorasick.Automaton]]:
    """
    Загружает словари и строит по каждой категории два автомата:
      automata[cat]["spaced"] / automata[cat]["compact"]
    """
    def _load(filename: str) -> Tuple[ahocorasick.Automaton, ahocorasick.Automaton]:
        lemmas = read_list(os.path.join(dict_dir, filename))
        return build_dual_automata(lemmas, norm_to=norm_to)

    prof_sp, prof_cp = _load("profanity.txt")
    ext_sp,  ext_cp  = _load("banned_orgs.txt")
    ill_sp,  ill_cp  = _load("illegal_phrases.txt")

    return {
        "profanity":         {"spaced": prof_sp, "compact": prof_cp},
        "extremism_support": {"spaced": ext_sp,  "compact": ext_cp},
        "illegal_content":   {"spaced": ill_sp,  "compact": ill_cp},
    }


def dict_hits(text: str, automata: Dict[str, Dict[str, ahocorasick.Automaton]], *, norm_to: str = "cyr"):
    norm_spaced  = normalize_surface(text, to=norm_to)
    norm_compact = normalize_compact(text, to=norm_to)

    out: Dict[str, List[str]] = {}
    for cat, pair in automata.items():
        found: Set[str] = set()

        for _, payload in _iter_safe(pair.get("spaced"), norm_spaced):
            found.add(payload)
        for _, payload in _iter_safe(pair.get("compact"), norm_compact):
            found.add(payload)

        if found:
            out[cat] = sorted(found)
    return out


def _iter_safe(A: ahocorasick.Automaton, text: str):
    """Итерируем только если автомат не пустой."""
    if A is None:
        return
    try:
        if len(A) == 0:
            return
    except Exception:
        return
    for item in A.iter(text):
        yield item