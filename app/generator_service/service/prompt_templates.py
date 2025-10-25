def make_prompt(context: dict) -> list[dict]:
    s = context.get("style") or {}
    ch = context.get("chosen") or {}

    topic_title = ch.get("topic") or "(тема не определена)"
    trend = s.get("trend_score") or 0.0

    avg_len = int(s.get("avg_len_tokens") or 300)
    p95_len = int(s.get("p95_len_tokens") or max(400, avg_len * 2))

    NL = "\n"

    style_lines = [
        f"Длина: ~{avg_len} токенов, не превышать {p95_len}.",
        f"Формат: markdown_rate={s.get('markdown_rate', 0):.2f}.",
        f"Пунктуация: !={s.get('exclam_rate', 0):.2f}, ?={s.get('question_rate', 0):.2f}, emoji={s.get('emoji_rate', 0):.2f}.",
        "Частые фразы: " + ", ".join((s.get("top_ngrams") or [])[:10]),
        "Сленг: " + ", ".join((s.get("slang_ngrams") or [])[:10]),
    ]
    style_block = NL.join(style_lines)

    exemplars_list = []
    for i, ex in enumerate((context.get("exemplars") or [])[:4]):
        text = (ex.get("text") or "")[:600]
        exemplars_list.append("#Exemplar{}:{}{}".format(i + 1, NL, text))
    exemplars_block = NL.join(exemplars_list) if exemplars_list else "(нет эталонов)"

    similar_items = []
    for sm in (context.get("similar") or [])[:8]:
        txt = (sm.get("text") or "")[:200].replace("\n", " ")
        similar_items.append("- " + txt)
    similar_snippets = NL.join(similar_items) if similar_items else "(нет похожих фрагментов)"

    system = (
        "Ты пишешь посты под формат форума. Соблюдай стиль и длину кластера. "
        "Не копируй примеры — пиши оригинально на их манер. Избегай фактических ошибок."
    )

    user = (
        "[TopicBrief]\n"
        f"Тема: {topic_title}\n"
        f"Интегральный тренд (0..1): {trend:.2f}\n\n"
        "[StyleGuide]\n"
        f"{style_block}\n\n"
        "[Exemplars]\n"
        f"{exemplars_block}\n\n"
        "[SimilarSnippets]\n"
        f"{similar_snippets}\n\n"
        "[UserRequest]\n"
        f"{context.get('query')}\n\n"
        "[Task]\n"
        "1) Дай краткий план (3–6 пунктов).\n"
        "2) Затем напиши сам пост в стиле форума и в рамках длины.\n"
        "3) Избегай копирования из примеров; используй общий тон/лексикон."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
