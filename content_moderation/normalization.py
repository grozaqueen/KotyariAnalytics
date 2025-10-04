import re
import unicodedata
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

LAT2CYR = str.maketrans({
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
    "k": "к", "m": "м", "t": "т", "i": "і", "h": "һ",
})

LEET = str.maketrans({
    "@": "a", "4": "a", "€": "e", "3": "e", "1": "i", "!": "i", "|": "i",
    "0": "o", "5": "s", "$": "s", "7": "t", "+": "t",
})

RE_SEPARATORS = re.compile(r"[\s\.\,\-\_\*\|\+\=\~\\/]+", re.UNICODE)
RE_REPEAT = re.compile(r"(.)\1{2,}", re.UNICODE)
RE_PUNCT_DENSE = re.compile(r"[^\w\u0400-\u04FF\s]+", re.UNICODE)

def strip_diacritics(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def normalize_surface(s: str, *, to="cyr") -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = strip_diacritics(s)
    s = s.translate(LEET)
    s = RE_SEPARATORS.sub(" ", s)
    s = RE_REPEAT.sub(r"\1\1", s)
    if to == "cyr":
        s = s.translate(LAT2CYR)
    s = RE_PUNCT_DENSE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("ё", "е")
    return s

def normalize_compact(s: str, *, to="cyr") -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = strip_diacritics(s)
    s = s.translate(LEET)
    s = RE_SEPARATORS.sub("", s)    # <-- вообще без пробелов
    s = RE_REPEAT.sub(r"\1\1", s)
    if to == "cyr":
        s = s.translate(LAT2CYR)
    s = s.replace("ё", "е")
    s = RE_PUNCT_DENSE.sub("", s)
    return s

def normalize_text(s: str, *, to="cyr", allow_unidecode=True) -> str:

    if not s:
        return ""
    surface = normalize_surface(s, to=to)
    words = surface.split()
    lemmatized = " ".join(morph.parse(word)[0].normal_form for word in words)
    return lemmatized
