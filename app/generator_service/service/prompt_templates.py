def make_prompt(context: dict) -> list[dict]:
    s = context.get("style") or {}
    ch = context.get("chosen") or {}

    topic_title = ch.get("topic") or "(тема не определена)"
    trend = float(s.get("trend_score") or ch.get("trend_score") or 0.0)

    avg_len = int(s.get("avg_len_tokens") or 300)
    p95_len = int(s.get("p95_len_tokens") or max(400, avg_len * 2))

    total_count = int(
        (s.get("total_count_30d") if s.get("total_count_30d") is not None else ch.get("total_count_30d") or 0)
    )
    top_max_phrase = (s.get("top_max_phrase") or ch.get("top_max_phrase") or "")[:200]
    top_max_count = int(
        (s.get("top_max_count") if s.get("top_max_count") is not None else ch.get("top_max_count") or 0)
    )

    NL = "\n"

    style_lines = [
        f"Длина: ~{avg_len} токенов, не превышать {p95_len}.",
        f"Формат: markdown_rate={float(s.get('markdown_rate') or 0):.2f}.",
        f"Пунктуация: !={float(s.get('exclam_rate') or 0):.2f}, ?={float(s.get('question_rate') or 0):.2f}, emoji={float(s.get('emoji_rate') or 0):.2f}.",
        "Частые фразы: " + ", ".join((s.get("top_ngrams") or [])[:10]),
        "Сленг: " + ", ".join((s.get("slang_ngrams") or [])[:10]),
    ]
    style_block = NL.join(style_lines)

    if top_max_phrase or total_count > 0:
        pop_lines = [
            "Ориентир на спрос (Wordstat):",
            f"- самый популярный запрос: «{top_max_phrase}»" if top_max_phrase else "- популярные формулировки присутствуют",
        ]
        if top_max_count:
            pop_lines.append(f"- частота ключа-максимума: ~{top_max_count:,}".replace(",", " "))
        if total_count:
            pop_lines.append(f"- общий месячный спрос по теме: ~{total_count:,}".replace(",", " "))
        pop_lines.append(
            "- используй 1–2 точных вхождения и 2–3 близких перефраза, без переспама; вплетай естественно в заголовки и первый абзац."
        )
        popularity_block = NL.join(pop_lines)
    else:
        popularity_block = "(данных по популярности нет)"

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
        "Не копируй примеры — пиши оригинально на их манер. Избегай фактических ошибок. "
        "Учитывай популярные поисковые формулировки, но не допускай keyword stuffing."
    )

    user = (
        "[TopicBrief]\n"
        f"Тема: {topic_title}\n"
        f"Интегральный тренд (0..1): {trend:.2f}\n\n"
        "[Popularity]\n"
        f"{popularity_block}\n\n"
        "[StyleGuide]\n"
        f"{style_block}\n\n"
        "[Exemplars]\n"
        f"{exemplars_block}\n\n"
        "[SimilarSnippets]\n"
        f"{similar_snippets}\n\n"
        "[UserRequest]\n"
        f"{context.get('query')}\n\n"
        "[Task]\n"
        "1) Дай краткий план (3–6 пунктов), интегрируй 1–2 самые релевантные популярные формулировки естественно.\n"
        "2) Затем напиши сам пост в стиле форума, удерживая длину и тон; вариируй ключи с синонимами.\n"
        "3) Не копируй из примеров; используй общий тон/лексикон.\n"
        "4) Если данных по популярности мало, ориентируйся на стиль кластера и запрос пользователя."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
