def make_prompt(context: dict) -> list[dict]:
    s = context.get("style") or {}
    ch = context.get("chosen") or {}

    topic_title = ch.get("topic") or s.get("topic") or "(тема не определена)"
    trend = float(s.get("trend_score") or ch.get("trend_score") or 0.0)

    avg_len = int(s.get("avg_len_tokens") or 300)
    p95_len = int(s.get("p95_len_tokens") or max(400, avg_len * 2))

    total_count = int((s.get("total_count_30d") if s.get("total_count_30d") is not None else ch.get("total_count_30d") or 0))
    top_max_phrase = (s.get("top_max_phrase") or ch.get("top_max_phrase") or "")[:200]
    top_max_count = int((s.get("top_max_count") if s.get("top_max_count") is not None else ch.get("top_max_count") or 0))

    top_requests = (s.get("top_requests") or ch.get("top_requests") or []) or []
    associations = (s.get("associations") or ch.get("associations") or []) or []

    cluster_bigrams = (s.get("top_ngrams") or [])[:20]         # из cluster_style
    cluster_unigrams = (s.get("slang_ngrams") or [])[:15]      # из cluster_style

    STOP = {
        "и","в","на","с","по","за","из","к","у","от","для","о","об","а","но","как","что","это","то",
        "или","да","же","ли","бы","не","до","над","под","при","про","без","через","так","же","уже",
        "the","a","an","to","of","in","on","for","is","are","be","with","by","at","as","from","it"
    }

    def _norm_kw(x: str) -> str:
        x = (x or "").strip().strip("«»\"'`“”")
        x = " ".join(x.split())
        return x

    def _take_phrases(arr, k):
        out = []
        for x in arr:
            ph = (x.get("phrase") if isinstance(x, dict) else None) or ""
            ph = _norm_kw(ph)
            if ph:
                out.append(ph)
            if len(out) >= k:
                break
        return out

    def _filter_kw(kw: str) -> bool:
        if not kw: return False
        toks = [t for t in kw.lower().split() if t not in STOP and len(t) > 1]
        return len(toks) > 0

    def _dedup_preserve(seq):
        seen = set()
        out = []
        for item in seq:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out

    ws_primary = [_norm_kw(top_max_phrase)] if top_max_phrase else []
    ws_top = _take_phrases(top_requests, 8)
    ws_assoc = _take_phrases(associations, 4)

    cl_big = [_norm_kw(x) for x in cluster_bigrams]
    cl_uni = [_norm_kw(x) for x in cluster_unigrams]

    ws_all = [*ws_primary, *ws_top, *ws_assoc]
    ws_all = [kw for kw in ws_all if _filter_kw(kw)]
    ws_all = _dedup_preserve(ws_all)

    cl_all = [*cl_big, *cl_uni]
    cl_all = [kw for kw in cl_all if _filter_kw(kw)]
    cl_all = _dedup_preserve(cl_all)

    KW_MAX = 12
    kw_merged = _dedup_preserve(ws_all + cl_all)[:KW_MAX]

    print("===== Prompt factors =====")
    print(f"Topic:                {topic_title}")
    print(f"Trend score:          {trend:.3f}")
    print(f"Avg/P95 length:       {avg_len} / {p95_len}")
    print(f"Wordstat total 30d:   {total_count}")
    print(f"Wordstat top phrase:  {top_max_phrase!r} ({top_max_count})")
    print(f"WS keys (raw):        {', '.join(ws_all) if ws_all else '(none)'}")
    print(f"Cluster keys (raw):   {', '.join(cl_all) if cl_all else '(none)'}")
    print(f"Final keys (<=12):    {', '.join(kw_merged) if kw_merged else '(none)'}")
    print(f"Exemplars:            {len(context.get('exemplars') or [])}")
    print(f"Similar snippets:     {len(context.get('similar') or [])}")
    print("=========================\n")

    NL = "\n"

    style_lines = [
        f"Длина: ~{avg_len} токенов, не превышать {p95_len}.",
        f"Формат: markdown_rate={float(s.get('markdown_rate') or 0):.2f}.",
        f"Пунктуация: !={float(s.get('exclam_rate') or 0):.2f}, ?={float(s.get('question_rate') or 0):.2f}, emoji={float(s.get('emoji_rate') or 0):.2f}.",
        "Частые фразы (кластер): " + ", ".join((s.get("top_ngrams") or [])[:10]),
        "Сленг (кластер): " + ", ".join((s.get("slang_ngrams") or [])[:10]),
    ]
    style_block = NL.join(style_lines)

    if kw_merged or top_max_phrase or total_count:
        pop_lines = ["Ориентир на спрос (Wordstat + кластер):"]
        if top_max_phrase:
            pop_lines.append(f"- самый популярный запрос: «{top_max_phrase}»")
        if top_max_count:
            pop_lines.append(f"- частота ключа-максимума: ~{top_max_count:,}".replace(",", " "))
        if total_count:
            pop_lines.append(f"- общий месячный спрос по теме: ~{total_count:,}".replace(",", " "))
        if kw_merged:
            pop_lines.append("- приоритетные ключи (естественно, без переспама):")
            pop_lines.extend([f"  • {k}" for k in kw_merged])
        pop_lines.append("- включай 1–2 точных вхождения из первых ключей и 2–3 вариации в заголовок и первый абзац.")
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
        "Учитывай популярные формулировки и лексический фон кластера, но не допускай keyword stuffing и не пиши явно, что тема, которую ты заводишь, популярна, просто рассказывай о том, что тебя просят"
        "Нужно понимать, что пишешь действительно по теме, если вдруг в примерах для манера не те данные вообще, не стоит их использовать"
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
        "1) Напиши пост в стиле форума, удерживая длину и тон; вариируй ключи с синонимами, без переспама.\n"
        "2) Не копируй из примеров; используй общий тон/лексикон примеров.\n"
        "3) Если данных по популярности мало, опирайся на стиль кластера и запрос пользователя."
    )

    print("===== MESSAGE TO GROK =====")
    print("--- SYSTEM ---")
    print(system)
    print("\n--- USER ---")
    print(user)
    print("===== END OF MESSAGE =====\n")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
