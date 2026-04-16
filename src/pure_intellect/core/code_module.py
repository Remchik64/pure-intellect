"""CodeModule — опциональный модуль для работы с кодовыми проектами.

C1 дорожной карты: Code Indexing + Semantic Search.

Активируется когда сессия имеет тип 'project'.
Является тонкой обёрткой над существующими компонентами:
  - CardGenerator   — индексация файлов в ChromaDB
  - Retriever       — семантический поиск
  - GraphBuilder    — граф зависимостей

Использование:
  module = CodeModule(project_path='/path/to/project',
                      session_id='my_project')
  count = module.index_project()          # индексация
  results = module.search('функция evict') # поиск
  context = module.get_context_for_llm('как работает память') # для prompt
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CodeSearchResult:
    """Результат поиска по коду."""

    def __init__(
        self,
        entity_name: str,
        entity_type: str,
        file_path: str,
        start_line: int,
        end_line: int,
        summary: str,
        relevance: float = 0.0,
    ):
        self.entity_name = entity_name
        self.entity_type = entity_type
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.summary = summary
        self.relevance = relevance

    def to_dict(self) -> dict:
        return {
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "summary": self.summary,
            "relevance": round(self.relevance, 3),
        }

    def to_context_string(self) -> str:
        return (
            f"[{self.entity_type.upper()}] {self.entity_name}\n"
            f"  File: {self.file_path}:{self.start_line}-{self.end_line}\n"
            f"  {self.summary}"
        )


class CodeModule:
    """Модуль для работы с кодовыми проектами.

    Активируется при создании project-сессии.
    Инкапсулирует индексацию, поиск и граф зависимостей.
    """

    # Поддерживаемые расширения по умолчанию
    DEFAULT_EXTENSIONS = [".py"]

    def __init__(
        self,
        project_path: str,
        session_id: str = "default",
        chroma_dir: str = "./storage/chromadb",
        extensions: Optional[list] = None,
    ):
        self.project_path = Path(project_path)
        self.session_id = session_id
        self.chroma_dir = chroma_dir
        self.extensions = extensions or self.DEFAULT_EXTENSIONS

        self._card_generator = None
        self._retriever = None
        self._graph_builder = None
        self._indexed_files: int = 0
        self._last_indexed: Optional[str] = None
        self._is_indexed: bool = False

        logger.info(
            f"[code_module] Initialized for project: {self.project_path} "
            f"(session={session_id})"
        )

    # ── Lazy initialization ──────────────────────────────────

    def _get_card_generator(self):
        """Lazy-load CardGenerator."""
        if self._card_generator is None:
            try:
                from .card_generator import CardGenerator
                self._card_generator = CardGenerator(chroma_dir=self.chroma_dir)
                logger.info("[code_module] CardGenerator initialized")
            except Exception as e:
                logger.error(f"[code_module] CardGenerator init failed: {e}")
                raise
        return self._card_generator

    def _get_retriever(self):
        """Lazy-load Retriever."""
        if self._retriever is None:
            try:
                from .retriever import Retriever
                self._retriever = Retriever(chroma_dir=self.chroma_dir)
                logger.info("[code_module] Retriever initialized")
            except Exception as e:
                logger.error(f"[code_module] Retriever init failed: {e}")
                raise
        return self._retriever

    def _get_graph_builder(self):
        """Lazy-load GraphBuilder."""
        if self._graph_builder is None:
            try:
                from .graph_builder import GraphBuilder
                self._graph_builder = GraphBuilder()
                logger.info("[code_module] GraphBuilder initialized")
            except Exception as e:
                logger.warning(f"[code_module] GraphBuilder init failed (non-critical): {e}")
                return None
        return self._graph_builder

    # ── Публичный API ────────────────────────────────────────

    @property
    def is_indexed(self) -> bool:
        return self._is_indexed

    @property
    def indexed_files(self) -> int:
        return self._indexed_files

    def index_project(
        self,
        extensions: Optional[list] = None,
        force: bool = False,
    ) -> dict:
        """Проиндексировать папку проекта в ChromaDB.

        Args:
            extensions: список расширений (.py, .js, ...)
            force: переиндексировать даже если уже проиндексировано

        Returns:
            dict с результатами индексации
        """
        if self._is_indexed and not force:
            logger.info("[code_module] Already indexed, use force=True to re-index")
            return {
                "status": "already_indexed",
                "indexed_files": self._indexed_files,
                "last_indexed": self._last_indexed,
            }

        if not self.project_path.exists():
            return {
                "status": "error",
                "error": f"Project path not found: {self.project_path}",
            }

        exts = extensions or self.extensions
        start_time = time.time()

        try:
            generator = self._get_card_generator()
            count = generator.index_directory(
                directory=self.project_path,
                extensions=exts,
            )

            elapsed = round(time.time() - start_time, 2)
            self._indexed_files = count
            self._is_indexed = True

            import datetime
            self._last_indexed = datetime.datetime.now().isoformat()

            logger.info(
                f"[code_module] Indexed {count} entities from {self.project_path} "
                f"in {elapsed}s"
            )

            return {
                "status": "success",
                "indexed_files": count,
                "project_path": str(self.project_path),
                "extensions": exts,
                "elapsed_seconds": elapsed,
                "last_indexed": self._last_indexed,
            }

        except Exception as e:
            logger.error(f"[code_module] Indexing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "project_path": str(self.project_path),
            }

    def search(
        self,
        query: str,
        top_k: int = 5,
        entity_types: Optional[list] = None,
    ) -> list[CodeSearchResult]:
        """Семантический поиск по коду.

        Args:
            query: поисковый запрос на естественном языке
            top_k: количество результатов
            entity_types: фильтр по типам (function, class, method, ...)

        Returns:
            Список CodeSearchResult отсортированных по релевантности
        """
        if not self._is_indexed:
            logger.warning("[code_module] Project not indexed yet")
            return []

        try:
            retriever = self._get_retriever()
            raw_results = retriever.search(
                query=query,
                top_k=top_k,
            )

            results = []
            for r in raw_results:
                # Фильтрация по типу если указана
                if entity_types and r.entity_type not in entity_types:
                    continue
                results.append(
                    CodeSearchResult(
                        entity_name=r.entity_name,
                        entity_type=r.entity_type,
                        file_path=r.file_path,
                        start_line=r.start_line,
                        end_line=r.end_line,
                        summary=r.summary,
                        relevance=r.relevance_score,
                    )
                )

            logger.info(f"[code_module] Search '{query[:30]}...' → {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"[code_module] Search failed: {e}")
            return []

    def get_context_for_llm(
        self,
        query: str,
        top_k: int = 3,
        max_tokens: int = 1500,
    ) -> str:
        """Получить контекст кода для включения в LLM prompt.

        Args:
            query: запрос пользователя
            top_k: количество результатов
            max_tokens: максимальный размер контекста

        Returns:
            Форматированный контекст для system prompt
        """
        if not self._is_indexed:
            return ""

        try:
            retriever = self._get_retriever()
            raw_results = retriever.search(query=query, top_k=top_k)

            if not raw_results:
                return ""

            # Форматируем контекст
            context_parts = []
            total_len = 0
            for r in raw_results:
                part = (
                    f"\n[CODE: {r.entity_type} '{r.entity_name}']"
                    f" in {r.file_path}:{r.start_line}\n"
                    f"{r.summary}"
                )
                # Грубая оценка токенов: 1 токен ≈ 4 символа
                if total_len + len(part) // 4 > max_tokens:
                    break
                context_parts.append(part)
                total_len += len(part) // 4

            if not context_parts:
                return ""

            header = f"\n[КОНТЕКСТ ПРОЕКТА: {self.project_path.name}]"
            return header + "\n".join(context_parts)

        except Exception as e:
            logger.error(f"[code_module] get_context_for_llm failed: {e}")
            return ""

    def build_graph(self) -> dict:
        """Построить граф зависимостей проекта.

        Returns:
            dict со статистикой графа
        """
        try:
            builder = self._get_graph_builder()
            if builder is None:
                return {"status": "unavailable", "error": "GraphBuilder not available"}

            stats = builder.build_from_directory(
                directory=self.project_path,
                extensions=self.extensions,
            )
            logger.info(f"[code_module] Graph built: {stats}")
            return {"status": "success", **stats}
        except Exception as e:
            logger.error(f"[code_module] Graph build failed: {e}")
            return {"status": "error", "error": str(e)}

    def stats(self) -> dict:
        """Статистика CodeModule."""
        card_count = 0
        try:
            if self._card_generator is not None:
                card_count = self._card_generator.search_cards("function", top_k=1000)
                card_count = len(card_count)
        except Exception:
            pass

        return {
            "project_path": str(self.project_path),
            "session_id": self.session_id,
            "is_indexed": self._is_indexed,
            "indexed_files": self._indexed_files,
            "last_indexed": self._last_indexed,
            "extensions": self.extensions,
            "chroma_dir": self.chroma_dir,
        }

    def is_code_query(self, query: str) -> bool:
        """Определить является ли запрос вопросом о коде.

        Используется в orchestrator чтобы решить нужен ли RAG по коду.

        Returns:
            True если запрос похоже связан с кодом
        """
        code_keywords = [
            "функция", "метод", "класс", "модуль", "файл",
            "function", "method", "class", "module", "file",
            "код", "code", "реализация", "implementation",
            "как работает", "how does", "что делает", "what does",
            "найди", "find", "покажи", "show", "где", "where",
            "импорт", "import", "зависимост", "depend",
            "def ", "class ", "import ", "return ",
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in code_keywords)
