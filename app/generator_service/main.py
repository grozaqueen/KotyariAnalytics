import sys
from .service.context_builder import build_context_for_query
from .service.prompt_templates import make_prompt
from .service.generator import generate

def generate_post(user_query: str) -> str:
    ctx = build_context_for_query(user_query)
    if not ctx.get("chosen"):
        return "Не удалось подобрать подходящий кластер — проверь оффлайн данные."

    print("=== Building messages for LLM ===")
    messages = make_prompt(ctx)
    for i, m in enumerate(messages, 1):
        role = m.get("role")
        content_preview = (m.get("content") or "")[:240].replace("\n", " ")
        print(f"[{i}] role={role} | content[:240]={content_preview!r}")
    print("=================================\n")

    out = generate(messages)
    print("\n=== Generation done ===")
    return out

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Напиши пост про мейн-кунов"
    print(generate_post(q))
