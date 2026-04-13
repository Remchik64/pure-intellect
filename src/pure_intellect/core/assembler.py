"""Context Assembler — сборка контекста для LLM."""

from pure_intellect.config import settings
from pure_intellect.utils.tokenizer import count_tokens, truncate_to_tokens
from pure_intellect.utils.logger import get_logger

logger = get_logger("assembler")


class ContextAssembler:
    """Собирает оптимальный контекст для запроса."""

    def __init__(self, retriever, graph, archive):
        self.retriever = retriever
        self.graph = graph
        self.archive = archive

    def build_messages(
        self,
        query: str,
        mode: str = "chat",
        project: str = "default",
        conversation_id: str | None = None,
    ) -> list[dict]:
        """Собрать messages[] для отправки в LLM."""

        # 1. System prompt с инструкциями
        system_prompt = self._build_system_prompt(mode)

        # 2. История беседы (summary)
        history_summary = ""
        if conversation_id:
            history_summary = self.archive.get_conversation_summary(conversation_id)

        # 3. RAG поиск релевантных карточек
        rag_results = self.retriever.search(query, top_k=settings.max_rag_chunks)
        rag_cards = [r["document"] for r in rag_results]

        # 4. Граф: связанные сущности
        graph_context = self._get_graph_context(query, rag_results)

        # 5. Сборка system prompt с контекстом
        full_system = self._assemble_system(
            base_prompt=system_prompt,
            history=history_summary,
            rag_cards=rag_cards,
            graph_context=graph_context,
        )

        # 6. Формируем messages
        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": query},
        ]

        # 7. Обрезаем до бюджета токенов
        messages = self._fit_budget(messages)

        logger.info(
            f"Context: system={count_tokens(full_system)} tokens, "
            f"query={count_tokens(query)} tokens, "
            f"rag_cards={len(rag_cards)}"
        )

        return messages

    def _build_system_prompt(self, mode: str) -> str:
        """Базовый system prompt по режиму."""
        prompts = {
            "analyze": (
                "Ты Senior разработчик. Анализируй код, находи ошибки, "
                "предлагай конкретные решения с примерами."
            ),
            "code": (
                "Ты Senior разработчик. Пиши чистый, production-ready код. "
                "Используй type hints, docstrings, обработку ошибок."
            ),
            "explain": (
                "Ты Senior разработчик. Объясняй код простым языком, "
                "разбирай по шагам, приводи аналогии."
            ),
            "refactor": (
                "Ты Senior разработчик. Предлагай рефакторинг: SOLID, DRY, "
                "улучшение читаемости, производительности."
            ),
            "chat": (
                "Ты Senior разработчик. Отвечай на основе предоставленного контекста. "
                "Если информации недостаточно — честно скажи."
            ),
        }
        return prompts.get(mode, prompts["chat"])

    def _get_graph_context(self, query: str, rag_results: list[dict]) -> str:
        """Получить контекст из графа знаний."""
        # Собираем ID сущностей из RAG результатов
        entity_ids = set()
        for r in rag_results:
            metadata = r.get("metadata", {})
            entity_name = metadata.get("entity_name", "")
            if entity_name:
                # Ищем в графе по имени
                matches = self.graph.search_by_name(entity_name)
                for m in matches:
                    entity_ids.add(m["id"])

        # Получаем связанные сущности
        related = []
        for eid in list(entity_ids)[:3]:  # максимум 3 корневых узла
            related.extend(self.graph.get_related(eid, depth=1))

        if not related:
            return ""

        # Формируем контекст
        lines = []
        for item in related[:10]:  # максимум 10 связей
            name = item.get("name", "?")
            item_type = item.get("type", "?")
            summary = item.get("summary", "")
            lines.append(f"- {name} ({item_type}): {summary}")

        return "\n".join(lines)

    def _assemble_system(
        self,
        base_prompt: str,
        history: str,
        rag_cards: list[str],
        graph_context: str,
    ) -> str:
        """Собрать полный system prompt."""
        parts = [base_prompt]

        if history:
            parts.append(f"\n[ИСТОРИЯ БЕСЕДЫ]\n{history}")

        if graph_context:
            parts.append(f"\n[АРХИТЕКТУРА ПРОЕКТА]\n{graph_context}")

        if rag_cards:
            cards_text = "\n\n".join(rag_cards)
            parts.append(f"\n[КОД ИЗ ПРОЕКТА]\n{cards_text}")

        parts.append(
            "\nОтвечай строго на основе предоставленного контекста. "
            "Если нужно больше деталей — укажи конкретно, какой файл/функцию посмотреть."
        )

        return "\n".join(parts)

    def _fit_budget(self, messages: list[dict]) -> list[dict]:
        """Обрезать messages до бюджета токенов."""
        max_system = settings.max_system_prompt_tokens
        max_total = settings.max_context_tokens

        # Обрезаем system prompt если нужно
        for msg in messages:
            if msg["role"] == "system":
                tokens = count_tokens(msg["content"])
                if tokens > max_system:
                    msg["content"] = truncate_to_tokens(msg["content"], max_system)
                    logger.warning(f"System prompt truncated from {tokens} to {max_system} tokens")

        # Считаем общее количество токенов
        total = sum(count_tokens(m["content"]) for m in messages)

        # Если превышаем — обрезаем RAG карточки в system prompt
        if total > max_total:
            for msg in messages:
                if msg["role"] == "system":
                    # Ищем секцию [КОД ИЗ ПРОЕКТА] и обрезаем её
                    content = msg["content"]
                    if "[КОД ИЗ ПРОЕКТА]" in content:
                        start = content.index("[КОД ИЗ ПРОЕКТА]")
                        end = content.find("\n\n", start + 20)
                        if end == -1:
                            end = len(content)
                        before = content[:start]
                        after = content[end:]
                        # Оставляем только заголовок секции
                        truncated_section = "[КОД ИЗ ПРОЕКТА]\n(контекст обрезан для экономии токенов)"
                        msg["content"] = before + truncated_section + after

        return messages
