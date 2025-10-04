from typing import Dict, Any

def decide(hits: Dict[str, list], scores: Dict[str, float], cfg: dict) -> Dict[str, Any]:
    th = cfg["thresholds"]
    decision = "allow"
    reason = "clean"

    if "illegal_content" in hits or scores.get("illegal_content", 0) >= th["block_illegal"]:
        return {"decision": "block", "reason": "illegal_content"}
    if "extremism_support" in hits or scores.get("extremism_support", 0) >= th["block_extremism"]:
        return {"decision": "block", "reason": "extremism_support"}

    if "profanity" in hits or scores.get("profanity", 0) >= th["redact_profanity"]:
        return {"decision": "redact", "reason": "profanity"}

    if any(th["review_band_low"] <= scores.get(k, 0) < th["review_band_high"]
           for k in ("illegal_content", "extremism_support")):
        return {"decision": "review", "reason": "uncertain_high_risk"}

    return {"decision": decision, "reason": reason}
