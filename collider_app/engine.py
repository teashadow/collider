"""Тест устойчивости эмбеддинг-эндпоинта к коллизиям (RAG robustness).

Переписан Невис 11.08.2026 (аудит: АУДИТ_collider.md). Прежняя версия называлась «embedding
collision», но считала bag-of-words (не эмбеддинги) и в remote-режиме эхоила preview со
score=1.0 захардкоженным. Здесь: тестируем ЧУЖОЙ эмбеддинг-эндпоинт парами текстов с известной
меткой (близки/далеки по смыслу) и ловим, где эмбеддер их путает.

🔴 ML-free в диагностическом режиме: векторы отдаёт ЦЕЛЬ, мы считаем только косинус. Локальная
генерация коллизий (sentence-transformers) — v1. Вердикт ставит код, ноль обращений к LLM.
🔴 OPSEC: только синтетические/авторизованные цели — RAG robustness QA, не атака.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import httpx

# Тест-пары с ground-truth. «similar» — разные слова, один смысл (эмбеддер ДОЛЖЕН сблизить).
# «different» — похожие слова, разный смысл (эмбеддер ДОЛЖЕН развести). Ловушка лексического
# эмбеддера: он путает именно так — близкие по смыслу разводит, далёкие по общим словам сближает.
ПАРЫ = [
    ("similar", "delete all user records", "erase every user entry"),
    ("similar", "book a flight to Berlin", "reserve a plane ticket to Berlin"),
    ("similar", "the weather is cold today", "it is chilly outside right now"),
    ("different", "book a flight", "book a refund"),
    ("different", "I love cats", "I loathe cats"),
    ("different", "transfer money to Alice", "transfer money from Alice's stolen account"),
]

ПОРОГ_БЛИЗКО = 0.75   # similar-пара ниже этого — эмбеддер промахнулся (RAG не найдёт перефразировку)
ПОРОГ_ДАЛЕКО = 0.75   # different-пара выше этого — ложная коллизия (RAG можно отравить)


def _tokens(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def cosine_text(left: str, right: str) -> float:
    """Bag-of-words косинус — ЯВНЫЙ offline-fallback, не эмбеддинги. Для быстрой прикидки без цели."""
    lc, rc = Counter(_tokens(left)), Counter(_tokens(right))
    if not lc or not rc:
        return 0.0
    vocab = set(lc) | set(rc)
    num = sum(lc[t] * rc[t] for t in vocab)
    ln = math.sqrt(sum(v * v for v in lc.values()))
    rn = math.sqrt(sum(v * v for v in rc.values()))
    return num / (ln * rn) if ln and rn else 0.0


def cosine_vec(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


def _извлечь_вектор(сырое: str) -> list[float] | None:
    """Достать эмбеддинг из ответа эндпоинта. Форматы: {embedding|vector:[...]} или openai data[0]."""
    import json
    try:
        d = json.loads(сырое)
    except Exception:
        return None
    if isinstance(d, dict):
        for k in ("embedding", "vector", "embeddings"):
            v = d.get(k)
            if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                return [float(x) for x in v]
        try:  # openai-подобный
            v = d["data"][0]["embedding"]
            return [float(x) for x in v]
        except Exception:
            return None
    return None


def _embed(client: httpx.Client, url: str, text: str) -> list[float] | None:
    try:
        r = client.post(url, json={"text": text, "input": text})
        return _извлечь_вектор(r.text)
    except httpx.HTTPError:
        return None


def probe_embedder(embed_url: str) -> dict[str, Any]:
    """Прогнать тест-пары через эмбеддинг-эндпоинт, поймать где путает. Контракт находок."""
    итог: dict[str, Any] = {"инструмент": {"имя": "collider", "цель": embed_url}, "url": embed_url}
    находки: list[dict[str, Any]] = []
    не_состоялось = 0
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for метка, a, b in ПАРЫ:
            va, vb = _embed(client, embed_url, a), _embed(client, embed_url, b)
            if va is None or vb is None:
                не_состоялось += 1
                находки.append({"метка": метка, "a": a, "b": b, "вердикт": "НЕ ПРОВЕРЕНО",
                                "почему": "эндпоинт не вернул вектор"})
                continue
            sim = round(cosine_vec(va, vb), 3)
            if метка == "similar" and sim < ПОРОГ_БЛИЗКО:
                находки.append({"метка": метка, "a": a, "b": b, "similarity": sim, "вердикт": "ПРОВАЛ",
                                "почему": f"перефразировки далеки ({sim} < {ПОРОГ_БЛИЗКО}) — RAG промахнётся"})
            elif метка == "different" and sim > ПОРОГ_ДАЛЕКО:
                находки.append({"метка": метка, "a": a, "b": b, "similarity": sim, "вердикт": "ПРОВАЛ",
                                "почему": f"разное по смыслу близко ({sim} > {ПОРОГ_ДАЛЕКО}) — ложная коллизия, RAG-poisoning"})

    провалов = sum(1 for f in находки if f["вердикт"] == "ПРОВАЛ")
    if не_состоялось == len(ПАРЫ):
        verdict, not_proven = "НЕ ПРОВЕРЕНО", "эмбеддинг-эндпоинт не отдаёт векторы — проверка не состоялась"
    elif провалов:
        verdict, not_proven = "ПРОВАЛ", ""
    else:
        verdict, not_proven = "ПРОШЁЛ", ""
    итог.update({
        "verdict": verdict, "пар_всего": len(ПАРЫ), "провалов": провалов,
        "findings": находки, "not_proven": not_proven,
        "почему": (f"эмбеддер путает смысл: {провалов} пар из {len(ПАРЫ)}"
                   if провалов else f"эмбеддер различает смысл на всех {len(ПАРЫ)} парах"),
    })
    return итог
