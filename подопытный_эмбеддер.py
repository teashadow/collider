#!/usr/bin/env python3
"""Синтетические эмбеддинг-эндпоинты для проверки collider. 🔴 OPSEC: localhost, не реальные цели.

  семантичный — вектор по СМЫСЛУ (ручная тема-схема): перефразировки → близкие векторы,
                разное по смыслу → далёкие. collider обязан дать ПРОШЁЛ.
  лексический — вектор по bag-of-words (общие слова): «book a flight»/«book a refund» ложно
                близки (общее «book a»), «delete»/«erase» ложно далеки. collider обязан дать
                ПРОВАЛ (эмбеддер путает смысл).

Формат: POST {text} → {embedding: [...]}.
Запуск: python3 подопытный_эмбеддер.py семантичный 8899  |  лексический 8898
"""
import hashlib
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# Ручная тема-схема для семантичного эмбеддера: ключевые смыслы → индекс оси вектора.
# Перефразировки делят тему → близкие векторы; разный смысл → разные оси.
ТЕМЫ = {
    "delete": ["delete", "erase", "remove", "wipe"],
    "book_travel": ["flight", "plane", "ticket", "reserve travel"],
    "book_money": ["refund", "money back", "reimburse"],
    "weather_cold": ["cold", "chilly", "freezing"],
    "affection": ["love", "adore", "like"],
    "aversion": ["loathe", "hate", "detest"],
    "transfer_to": ["transfer money to", "send money to"],
    "transfer_theft": ["stolen", "theft", "fraud"],
}
РАЗМЕР = 32


def _семантический_вектор(text: str) -> list[float]:
    низ = text.lower()
    v = [0.0] * РАЗМЕР
    for i, (тема, слова) in enumerate(ТЕМЫ.items()):
        if any(с in низ for с in слова):
            # тема кладёт «массу» на свои 4 оси — перефразировки одной темы совпадут
            for j in range(4):
                v[(i * 4 + j) % РАЗМЕР] += 1.0
    if not any(v):
        v[hash(низ) % РАЗМЕР] = 1.0
    return v


def _лексический_вектор(text: str) -> list[float]:
    # bag-of-words по хешу слов — общие слова дают общие оси, смысл игнорируется
    v = [0.0] * РАЗМЕР
    for слово in re.findall(r"[a-z]+", text.lower()):
        v[int(hashlib.md5(слово.encode()).hexdigest(), 16) % РАЗМЕР] += 1.0
    return v


def обработчик(режим: str):
    вектор = _семантический_вектор if режим == "семантичный" else _лексический_вектор

    class Ручка(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            сырое = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            try:
                text = json.loads(сырое).get("text") or json.loads(сырое).get("input", "")
            except Exception:
                self.send_response(400); self.end_headers(); return
            тело = json.dumps({"embedding": вектор(text)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(тело)))
            self.end_headers()
            self.wfile.write(тело)

    return Ручка


if __name__ == "__main__":
    режим = sys.argv[1] if len(sys.argv) > 1 else "семантичный"
    порт = int(sys.argv[2]) if len(sys.argv) > 2 else 8899
    HTTPServer(("127.0.0.1", порт), обработчик(режим)).serve_forever()
