from content_moderation.dictionary import build_ac_surface, dict_hits

PROF_LEMMAS = ["сука", "блядь", "хуй", "пидор", "ебать"]
A = build_ac_surface(PROF_LEMMAS)

tests = [
    "ну и суки они",
    "СУКА!!!",
    "с-у_к-а",
    "су**о-чинки",
    "блядская работа",
]

for t in tests:
    hits = dict_hits(t, {"profanity": {"spaced": A, "compact": A}})
    print(t, "=>", hits)
