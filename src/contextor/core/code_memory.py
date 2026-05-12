"""CodeMemoryExtractor — извлечение фактов о коде в память.

C3 дорожной карты: Code-Aware Memory.

Когда пользователь обсуждает код:
  Вопрос: «Как работает evict_below_threshold?»
  CodeModule находит функцию в ChromaDB
  CodeMemoryExtractor извлекает факты:
    → «evict_below_threshold() в WorkingMemory:
       выгружает факты с attention < 0.2 при pressure > 0.8»
  Факт сохраняется в WorkingMemory с высоким весом
  При следующей сессии — модель помнит архитектурное решение
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class CodeFact:
    """Структурированный факт о коде для хранения в памяти."""

    def __init__(
        self,
        content: str,
        entity_name: Optional[str] = None,
        entity_type: Optional[str] = None,
        file_path: Optional[str] = None,
        importance: float = 0.7,
    ):
        self.content = content
        self.entity_name = entity_name
        self.entity_type = entity_type
        self.file_path = file_path
        self.importance = importance  # 0.0 - 1.0

    def __repr__(self) -> str:
        return f"CodeFact({self.entity_name!r}, importance={self.importance})"


class CodeMemoryExtractor:
    """Извлекает структурированные факты из кодовых обсуждений.

    Работает без LLM — правила + структура CodeModule результатов.
    Быстро (< 1ms) и надёжно.
    """

    # Минимальная важность для добавления в память
    MIN_IMPORTANCE = 0.5

    # Шаблоны для определения важных упоминаний
    IMPORTANT_PATTERNS = [
        r"(?:реализова|implement|добавлен|create|создан)",
        r"(?:исправлен|fix|починен|bug|ошибк)",
        r"(?:оптимизирован|optimiz|улучшен|improv)",
        r"(?:рефакторинг|refactor)",
        r"(?:архитектур|architect|design|паттерн|pattern)",
        r"(?:важно|important|критично|critical|ключевой|key)",
    ]

    def __init__(self):
        self._important_re = re.compile(
            "|".join(self.IMPORTANT_PATTERNS),
            re.IGNORECASE,
        )

    def extract_from_code_context(
        self,
        query: str,
        code_results: list,  # list[CodeSearchResult]
        response: str = "",
    ) -> list[CodeFact]:
        """Извлечь факты из результатов поиска по коду.

        Args:
            query: оригинальный вопрос пользователя
            code_results: результаты CodeModule.search()
            response: ответ LLM (опционально)

        Returns:
            Список CodeFact для добавления в WorkingMemory
        """
        facts = []

        for result in code_results[:3]:  # топ-3 результата
            fact = self._result_to_fact(result, query)
            if fact and fact.importance >= self.MIN_IMPORTANCE:
                facts.append(fact)

        # Дополнительный факт о самом запросе если он важный
        if self._important_re.search(query + " " + response[:200]):
            summary_fact = self._create_discussion_fact(query, code_results, response)
            if summary_fact:
                facts.append(summary_fact)

        logger.info(
            f"[code_memory] Extracted {len(facts)} facts from "
            f"{len(code_results)} code results"
        )
        return facts

    def format_for_working_memory(self, facts: list[CodeFact]) -> list[str]:
        """Форматировать CodeFact для добавления в WorkingMemory.

        Returns:
            Список строк готовых к Fact(content=...)
        """
        return [f.content for f in facts]

    # ── Внутренние методы ────────────────────────────────────

    def _result_to_fact(self, result, query: str) -> Optional[CodeFact]:
        """Преобразовать CodeSearchResult в CodeFact."""
        try:
            entity_name = result.entity_name
            entity_type = result.entity_type
            file_path = result.file_path
            summary = result.summary
            relevance = getattr(result, "relevance", 0.5)

            # Формируем содержательный факт
            file_short = self._shorten_path(file_path)

            content = (
                f"[КОД] {entity_type.upper()} `{entity_name}` "
                f"в {file_short}: {summary[:120]}"
            )

            # Важность = релевантность * базовый вес
            importance = min(0.9, relevance + 0.2) if relevance > 0 else 0.6

            return CodeFact(
                content=content,
                entity_name=entity_name,
                entity_type=entity_type,
                file_path=file_path,
                importance=importance,
            )
        except Exception as e:
            logger.warning(f"[code_memory] _result_to_fact failed: {e}")
            return None

    def _create_discussion_fact(
        self,
        query: str,
        code_results: list,
        response: str,
    ) -> Optional[CodeFact]:
        """Создать обобщающий факт об обсуждении."""
        if not code_results:
            return None

        # Собираем имена обсуждаемых сущностей
        entities = [r.entity_name for r in code_results[:3]]
        entities_str = ", ".join(f"`{e}`" for e in entities)

        # Краткое содержание запроса
        query_short = query[:60] + "..." if len(query) > 60 else query

        content = (
            f"[ОБСУЖДЕНИЕ] Вопрос о {entities_str}: "
            f"«{query_short}»"
        )

        return CodeFact(
            content=content,
            entity_name="discussion",
            entity_type="discussion",
            importance=0.75,  # обсуждения важнее обычных фактов
        )

    def _shorten_path(self, file_path: str) -> str:
        """Сократить путь к файлу для читаемости."""
        if not file_path:
            return "unknown"
        parts = file_path.replace("\\", "/").split("/")
        if len(parts) <= 2:
            return file_path
        # Показываем только последние 2 части пути
        return "/".join(parts[-2:])


class CodeAwareMemoryIntegration:
    """Связывает CodeModule, CodeMemoryExtractor и WorkingMemory.

    Используется в OrchestratorPipeline:
      integration = CodeAwareMemoryIntegration()
      integration.process_code_turn(
          query=message,
          code_module=self._code_module,
          working_memory=self.working_memory,
          response=llm_response,
      )
    """

    def __init__(self):
        self._extractor = CodeMemoryExtractor()

    def process_code_turn(
        self,
        query: str,
        code_module,  # CodeModule | None
        working_memory,  # WorkingMemory
        response: str = "",
        max_facts: int = 3,
    ) -> tuple[str, int]:
        """Обработать один turn с кодовым контекстом.

        Алгоритм:
        1. Если нет code_module или не кодовый запрос → пропуск
        2. Поиск релевантного кода через CodeModule
        3. Извлечение фактов через CodeMemoryExtractor
        4. Добавление фактов в WorkingMemory
        5. Возврат кодового контекста для prompt

        Returns:
            (code_context_for_prompt, facts_added_count)
        """
        if code_module is None or not code_module.is_indexed:
            return "", 0

        if not code_module.is_code_query(query):
            return "", 0

        try:
            # 1. Ищем код
            results = code_module.search(query=query, top_k=5)
            if not results:
                return "", 0

            # 2. Формируем контекст для LLM
            code_context = code_module.get_context_for_llm(
                query=query, top_k=3, max_tokens=1200
            )

            # 3. Извлекаем факты
            facts = self._extractor.extract_from_code_context(
                query=query,
                code_results=results,
                response=response,
            )

            # 4. Добавляем в WorkingMemory
            facts_added = 0
            from .memory import Fact
            for code_fact in facts[:max_facts]:
                # Проверяем что такого факта ещё нет
                if not self._fact_exists(working_memory, code_fact.entity_name):
                    wm_fact = Fact(
                        content=code_fact.content,
                        source="code_module",
                        attention_weight=code_fact.importance,
                        is_anchor=False,  # не anchor, но с высоким весом
                    )
                    working_memory.add(wm_fact)
                    facts_added += 1

            logger.info(
                f"[code_aware_memory] Turn processed: "
                f"{len(results)} found, {facts_added} facts added to WM"
            )
            return code_context, facts_added

        except Exception as e:
            logger.error(f"[code_aware_memory] process_code_turn failed: {e}")
            return "", 0

    def _fact_exists(self, working_memory, entity_name: str) -> bool:
        """Проверить есть ли уже факт об этой сущности в WM."""
        try:
            existing = working_memory.get_context()
            return entity_name.lower() in existing.lower()
        except Exception:
            return False
