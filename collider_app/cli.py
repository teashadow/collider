from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .banner import COLLIDER_BANNER
from .engine import cosine_text, probe_embedder

console = Console()


def _banner() -> None:
    console.print(f"[bold green]{COLLIDER_BANNER}[/bold green]")


class BannerGroup(click.Group):
    def get_help(self, ctx: click.Context) -> str:
        _banner()
        return super().get_help(ctx)


@click.group(cls=BannerGroup)
def main() -> None:
    """MAD embedding collision — устойчивость эмбеддера/RAG к коллизиям."""


@main.command("bow")
@click.argument("text1")
@click.argument("text2")
def bow_cmd(text1: str, text2: str) -> None:
    """Быстрая bag-of-words прикидка (offline-fallback, НЕ эмбеддинги)."""
    console.print(f"bag-of-words cosine = {cosine_text(text1, text2):.4f}  "
                  f"[dim](для смысла нужен эмбеддинг-эндпоинт: команда probe)[/dim]")


@main.command("probe")
@click.argument("embed_url")
@click.option("--json", "as_json", type=click.Path(), default=None,
              help="сохранить JSON-находки (контракт пайплайна)")
def probe_cmd(embed_url: str, as_json: str | None) -> None:
    """Прогнать тест-пары через эмбеддинг-эндпоинт, поймать где путает смысл."""
    d = probe_embedder(embed_url)
    if as_json:
        Path(as_json).write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    # 🔴 rc=2 «не состоялась» ≠ rc=0 «чисто»
    if d["verdict"] == "НЕ ПРОВЕРЕНО":
        console.print(f"[yellow]НЕ ПРОВЕРЕНО[/yellow]: {d['not_proven']}")
        raise SystemExit(2)

    if d["findings"]:
        t = Table(title=f"collider: {embed_url}  ·  пар: {d['пар_всего']}")
        t.add_column("метка"); t.add_column("similarity"); t.add_column("почему")
        for f in d["findings"]:
            цвет = "red" if f["вердикт"] == "ПРОВАЛ" else "yellow"
            t.add_row(f"[{цвет}]{f['метка']}[/{цвет}]", str(f.get("similarity", "—")), f["почему"])
        console.print(t)
    цвет = "red" if d["verdict"] == "ПРОВАЛ" else "green"
    console.print(f"Вердикт: [{цвет}]{d['verdict']}[/{цвет}] — {d['почему']}")

    if d["verdict"] == "ПРОВАЛ":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
