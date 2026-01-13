import sys
from .service.context_builder import build_context_for_query
from .service.prompt_templates import make_prompt
from .service.generator import generate

def generate_post(user_query: str) -> str:
    blocks = parse_tagged_prompt(user_query)

    user_request = blocks["USER"] or ""
    profile_prompt = blocks["PROFILE"] or ""
    bot_prompt = blocks["BOT"] or ""

    ctx = build_context_for_query(user_request)  # <-- ВАЖНО: только user_request
    if not ctx.get("chosen"):
        return "Не удалось подобрать подходящий кластер — проверь оффлайн данные."

    print("=== Building messages for LLM ===")
    messages = make_prompt(ctx, user_profile=profile_prompt, bot_prompt=bot_prompt)
    for i, m in enumerate(messages, 1):
        role = m.get("role")
        content_preview = (m.get("content") or "")[:240].replace("\n", " ")
        print(f"[{i}] role={role} | content[:240]={content_preview!r}")
    print("=================================\n")

    out = generate(messages)
    print("\n=== Generation done ===")
    return out

import re
from typing import Dict

_TAG_RE = re.compile(r"^\[(BOT|PROFILE|USER)\]\s*$")

def parse_tagged_prompt(raw: str) -> Dict[str, str]:
    """
    Разбирает строку вида:
      [BOT]\n...\n\n[PROFILE]\n...\n\n[USER]\n...
    Если тегов нет — всё считаем USER.
    """
    blocks = {"BOT": "", "PROFILE": "", "USER": ""}
    current = None

    for line in (raw or "").splitlines():
        m = _TAG_RE.match(line.strip())
        if m:
            current = m.group(1)
            continue

        if current is None:
            current = "USER"

        blocks[current] += line + "\n"

    # trim
    for k in list(blocks.keys()):
        blocks[k] = blocks[k].strip()

    return blocks

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "напиши пост как правильно красить ногти"
    print(generate_post(q))
