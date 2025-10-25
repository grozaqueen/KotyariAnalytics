import sys
from .service.context_builder import build_context_for_query
from .service.prompt_templates import make_prompt
from .service.generator import generate

def generate_post(user_query: str) -> str:
    ctx = build_context_for_query(user_query)
    if not ctx.get("chosen"):
        return "Не удалось подобрать подходящий кластер — проверь оффлайн данные."
    messages = make_prompt(ctx)
    return generate(messages)

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Напиши пост про популярную породу кошек"
    print(generate_post(q))
