import re

def redact_terms(original: str, terms, placeholder="[REDACTED]") -> str:
    red = original
    for term in terms:
        pat = "".join(re.escape(ch) + r"\W*" for ch in term).rstrip(r"\W*")
        red = re.sub(pat, placeholder, red, flags=re.IGNORECASE | re.UNICODE)
    return red
