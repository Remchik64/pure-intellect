"""CLI entry point для Чистый Интеллект."""

import sys
import logging
from pathlib import Path
import click

from .engine import ModelManager, MODEL_REGISTRY


@click.group()
@click.version_option(version="0.1.0")
@click.option("--verbose", "-v", is_flag=True, help="Включить подробный вывод")
def cli(verbose):
    """🧠 Чистый Интеллект — локальный оркестратор для LLM"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


@cli.group()
def model():
    """Управление моделями."""
    pass


@model.command("list")
def model_list():
    """Показать доступные модели."""
    manager = ModelManager()
    downloaded = manager.list_downloaded()
    
    click.echo("\n📋 Доступные модели:\n")
    click.echo(f"  {'Модель':<30} {'Размер':<10} {'VRAM':<10} {'Статус'}")
    click.echo("  " + "─" * 70)
    
    for key, info in MODEL_REGISTRY.items():
        status = "✅ Скачана" if key in downloaded else "⬜ Не скачана"
        size = f"{info['size_gb']:.1f} GB"
        vram = f"{info['vram_gb']:.1f} GB"
        click.echo(f"  {key:<30} {size:<10} {vram:<10} {status}")
    
    click.echo(f"\nВсего моделей: {len(MODEL_REGISTRY)}")
    click.echo(f"Скачано: {len(downloaded)}\n")


@model.command()
@click.argument("model_key")
@click.option("--force", "-f", is_flag=True, help="Перескачать если уже есть")
def download(model_key, force):
    """Скачать модель с HuggingFace."""
    if model_key not in MODEL_REGISTRY:
        click.echo(f"❌ Неизвестная модель: {model_key}")
        click.echo(f"Доступные: {', '.join(MODEL_REGISTRY.keys())}")
        return
    
    manager = ModelManager()
    info = MODEL_REGISTRY[model_key]
    
    click.echo(f"\n📥 Скачиваю {info['name']}...")
    click.echo(f"   Размер: {info['size_gb']:.1f} GB")
    click.echo(f"   Репозиторий: {info['repo']}")
    click.echo(f"   Файл: {info['file']}\n")
    
    try:
        path = manager.download(model_key, force=force)
        click.echo(f"✅ Модель готова: {path}\n")
    except Exception as e:
        click.echo(f"❌ Ошибка: {e}\n")
        sys.exit(1)


@model.command()
@click.argument("model_key")
@click.option("--gpu-layers", "-g", default=-1, help="Количество GPU слоёв (-1 = все)")
def load(model_key, gpu_layers):
    """Загрузить модель в память."""
    if model_key not in MODEL_REGISTRY:
        click.echo(f"❌ Неизвестная модель: {model_key}")
        return
    
    manager = ModelManager()
    info = MODEL_REGISTRY[model_key]
    
    click.echo(f"\n⚙️  Загружаю {info['name']}...")
    click.echo(f"   GPU слоёв: {'все' if gpu_layers == -1 else gpu_layers}")
    
    try:
        llm = manager.load(model_key, n_gpu_layers=gpu_layers)
        click.echo(f"✅ Модель загружена!\n")
        
        # Тестовый запрос
        click.echo("🧪 Тестовый запрос...")
        response = manager.chat([
            {"role": "user", "content": "Привет! Напиши одно короткое предложение."}
        ], temperature=0.7)
        click.echo(f"   Ответ: {response[:100]}...\n")
        
    except Exception as e:
        click.echo(f"❌ Ошибка: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option("--host", "-h", default="0.0.0.0", help="Хост сервера")
@click.option("--port", "-p", default=8085, type=int, help="Порт сервера")
def serve(host, port):
    """Запустить сервер Оркестратора."""
    import uvicorn
    from .server import app
    
    click.echo(f"\n🚀 Запускаю сервер на {host}:{port}...")
    click.echo("   Нажмите Ctrl+C для остановки\n")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )


def main():
    cli()


if __name__ == "__main__":
    main()
