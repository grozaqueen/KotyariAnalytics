from typing import Any, Dict
from .utils import load_yaml, module_path
from .dictionary import load_automata, dict_hits
from .redaction import redact_terms

class ModerationPipeline:
    def __init__(self, cfg_path: str | None = None):
        cfg_file = cfg_path or module_path("config.yaml")
        self.cfg = load_yaml(cfg_file)
        self.norm_to = self.cfg["features"].get("normalize_to", "cyr")
        self.automata = load_automata(module_path("dicts"), norm_to=self.norm_to)

    def moderate(self, text: str) -> Dict[str, Any]:
        raw_text = text

        if self.cfg["features"].get("normalize_input", False):
            from .normalization import normalize_text
            try:
                text = normalize_text(text)
            except Exception:
                text = raw_text

        hits = dict_hits(text, self.automata, norm_to=self.norm_to)

        scores: Dict[str, float] = {}
        if self.cfg["features"].get("use_ml", True):
            try:
                from .ml import ml_scores
                scores = ml_scores(text)
            except Exception as e:
                scores = {}

        from .policy import decide
        verdict = decide(hits, scores, self.cfg)

        out: Dict[str, Any] = {
            "decision": verdict["decision"],
            "reason": verdict.get("reason", "unknown"),
            "hits": hits,
            "scores": scores,
        }

        if verdict["decision"] == "block":
            out["text"] = None
        elif verdict["decision"] == "redact":
            all_terms = set()
            for lst in hits.values():
                all_terms.update(lst)
            placeholder = self.cfg.get("redaction", {}).get("placeholder", "[REDACTED]")
            out["text"] = redact_terms(raw_text, all_terms, placeholder=placeholder)
        else:
            out["text"] = raw_text

        return out
