"""CLI entry point для Чистый Интеллект."""

import sys
import argparse
from pathlib import Path


def main():
    """Главная точка входа."""
    parser = argparse.ArgumentParser(
        prog="pure-intellect",
        description="🧠 Чистый Интеллект — локальный оркестратор для LLM",
    )

    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")

    # ─── serve ───
    serve_parser = subparsers.add_parser("serve", help="Запустить сервер Оркестратора")
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Хост (по умолчанию: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8085,
        help="Порт (по умолчанию: 8085)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Автоперезагрузка при изменениях (для разработки)",
    )

    # ─── index ───
    index_parser = subparsers.add_parser("index", help="Проиндексировать проект")
    index_parser.add_argument(
        "path",
        type=str,
        help="Путь к директории проекта",
    )

    # ─── status ───
    subparsers.add_parser("status", help="Показать статус системы")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "serve":
        _run_server(args)
    elif args.command == "index":
        _run_index(args)
    elif args.command == "status":
        _run_status()


def _run_server(args):
    """Запуск FastAPI сервера."""
    import uvicorn
    from pure_intellect.config import settings

    print("\n🧠 Чистый Интеллект — запуск Оркестратора...\n")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Ollama: {settings.ollama_url}")
    print(f"   Model: {settings.default_model}")
    print("\n" + "─" * 50 + "\n")

    uvicorn.run(
        "pure_intellect.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


def _run_index(args):
    """Индексация проекта."""
    project_path = Path(args.path)
    if not project_path.exists():
        print(f"❌ Путь не найден: {project_path}")
        sys.exit(1)

    print(f"\n📁 Индексация проекта: {project_path}\n")
    # TODO: реализовать индексацию
    print("⚠️  Индексация пока не реализована. Используйте API endpoint POST /index")


def _run_status():
    """Показать статус."""
    from pure_intellect.config import settings

    print("\n🧠 Чистый Интеллект — Статус\n")
    print(f"   Ollama URL:  {settings.ollama_url}")
    print(f"   Модель:      {settings.default_model}")
    print(f"   Контекст:    {settings.max_context_tokens} токенов")
    print(f"   Хранилище:   {settings.storage_dir}")
    print()


if __name__ == "__main__":
    main()
