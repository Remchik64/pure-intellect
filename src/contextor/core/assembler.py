"""Context Assembler — сборка контекста для LLM."""

from typing import Optional, List
from ..config import settings
from ..utils.tokenizer import count_tokens, truncate_to_tokens
from ..utils.logger import get_logger
from .retriever import Retriever, RetrievalResult
from .intent import IntentResult

logger = get_logger("assembler")


class ContextAssembler:
    """Собирает оптимальный контекст для запроса."""
    
    def __init__(self, retriever: Optional[Retriever] = None):
        self.retriever = retriever or Retriever()
    
    def build_messages(
        self,
        query: str,
        intent_result: Optional[IntentResult] = None,
        conversation_id: Optional[str] = None,
    ) -> list[dict]:
        """Собрать messages[] для отправки в LLM."""
        
        # 1. Определяем режим на основе intent
        mode = intent_result.intent.value if intent_result else "chat"
        entities = intent_result.entities if intent_result else []
        
        # 2. System prompt с инструкциями
        system_prompt = self._build_system_prompt(mode)
        
        # 3. RAG поиск релевантных карточек
        if entities:
            rag_results = self.retriever.search_by_intent(mode, entities)
        else:
            rag_results = self.retriever.search(query, top_k=settings.max_rag_chunks)
        
        # 4. Сборка system prompt с контекстом
        full_system = self._assemble_system(
            base_prompt=system_prompt,
            rag_results=rag_results,
            intent_result=intent_result,
        )
        
        # 5. Формируем messages
        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": query},
        ]
        
        # 6. Обрезаем до бюджета токенов
        messages = self._fit_budget(messages)
        
        logger.info(
            f"Context: system={count_tokens(full_system)} tokens, "
            f"query={count_tokens(query)} tokens, "
            f"rag_cards={len(rag_results)}, mode={mode}"
        )
        
        return messages
    
    def _build_system_prompt(self, mode: str) -> str:
        """Базовый system prompt по режиму."""
        prompts = {
            "debug": (
                "Ты Senior разработчик. Анализируй код, находи ошибки, "
                "предлагай конкретные решения с примерами."
            ),
            "code_generation": (
                "Ты Senior разработчик. Пиши чистый, production-ready код. "
                "Используй type hints, docstrings, обработку ошибок."
            ),
            "code_explain": (
                "Ты Senior разработчик. Объясняй код простым языком, "
                "разбирай по шагам, приводи аналогии."
            ),
            "refactor": (
                "Ты Senior разработчик. Предлагай рефакторинг: SOLID, DRY, "
                "улучшение читаемости, производительности."
            ),
            "architecture": (
                "Ты Senior архитектор. Анализируй архитектуру, "
                "предлагай паттерны, масштабируемость."
            ),
            "chat": (
                "Ты Senior разработчик. Отвечай на основе предоставленного контекста. "
                "Если информации недостаточно — честно скажи."
            ),
        }
        return prompts.get(mode, prompts["chat"])
    
    def _assemble_system(
        self,
        base_prompt: str,
        rag_results: List[RetrievalResult],
        intent_result: Optional[IntentResult] = None,
    ) -> str:
        """Собрать полный system prompt."""
        parts = [base_prompt]
        
        # Добавляем intent информацию
        if intent_result:
            parts.append(f"\n[INTENT] Type: {intent_result.intent.value}, Confidence: {intent_result.confidence}")
            if intent_result.entities:
                parts.append(f"[ENTITIES] {', '.join(intent_result.entities)}")
        
        # Добавляем RAG контекст
        if rag_results:
            context_str = self.retriever.format_context(rag_results)
            parts.append(f"\n{context_str}")
        
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
        
        # Если превышаем — обрезаем RAG секцию
        if total > max_total:
            for msg in messages:
                if msg["role"] == "system":
                    content = msg["content"]
                    if "[RELEVANT CODE CONTEXT]" in content:
                        start = content.index("[RELEVANT CODE CONTEXT]")
                        end = content.find("[/RELEVANT CODE CONTEXT]")
                        if end != -1:
                            end = content.index("\n", end) + 1
                            before = content[:start]
                            after = content[end:]
                            truncated = "[RELEVANT CODE CONTEXT]\n(context truncated for token budget)\n[/RELEVANT CODE CONTEXT]"
                            msg["content"] = before + truncated + after
        
        return messages
    
    def assemble_and_respond(self, query: str, intent_result: Optional[IntentResult] = None) -> dict:
        """Собрать контекст и подготовить для отправки в LLM."""
        messages = self.build_messages(query, intent_result)
        
        return {
            "messages": messages,
            "mode": intent_result.intent.value if intent_result else "chat",
            "total_tokens": sum(count_tokens(m["content"]) for m in messages),
        }
