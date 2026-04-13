"""Summarizer — сжатие истории бесед."""

import json
from pathlib import Path
from pure_intellect.utils.logger import get_logger

logger = get_logger("summarizer")


class Summarizer:
    """Сжатие и группировка истории бесед."""

    def __init__(self, archive, llm_engine=None):
        self.archive = archive
        self.llm_engine = llm_engine

    async def compress_conversation(
        self,
        conversation_id: str,
        max_pairs: int = 50,
    ) -> str:
        """
        Сжать историю беседы в summary.
        
        Используется когда количество пар (query, response) превышает порог.
        """
        pairs = self.archive.get_pairs(conversation_id, limit=max_pairs * 2)
        
        if len(pairs) <= max_pairs:
            return ""  # Сжатие не требуется
        
        # Разделяем на старые (для сжатия) и новые (оставляем как есть)
        old_pairs = pairs[max_pairs:]
        
        if self.llm_engine:
            # LLM-базированное сжатие
            summary = await self._llm_compress(old_pairs)
        else:
            # Простое сжатие без LLM
            summary = self._simple_compress(old_pairs)
        
        # Удаляем старые пары, сохраняем summary
        self.archive.set_summary(conversation_id, summary)
        self.archive.trim_pairs(conversation_id, keep_last=max_pairs)
        
        logger.info(
            f"Summarizer: compressed {len(old_pairs)} pairs -> {len(summary)} chars"
        )
        
        return summary

    async def _llm_compress(self, pairs: list[dict]) -> str:
        """Сжатие через LLM."""
        if not self.llm_engine:
            return self._simple_compress(pairs)
        
        # Формируем текст для сжатия
        conversation_text = "\n".join([
            f"User: {p['query']}\nAssistant: {p['response'][:500]}..."
            for p in pairs
        ])
        
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — модуль сжатия контекста. Твоя задача: создать краткое summary "
                    "беседы, сохранив ВСЕ ключевые факты, решения, обсужденные темы. "
                    "Формат: маркированный список. Не добавляй информацию, которой нет в тексте."
                ),
            },
            {
                "role": "user",
                "content": f"Сожми эту беседу в краткое summary:\n\n{conversation_text}",
            },
        ]
        
        try:
            response = await self.llm_engine.chat(
                messages=messages,
                temperature=0.1,
                max_tokens=1000,
            )
            return response.content
        except Exception as e:
            logger.error(f"LLM compression failed: {e}")
            return self._simple_compress(pairs)

    def _simple_compress(self, pairs: list[dict]) -> str:
        """Простое сжатие без LLM."""
        topics = set()
        decisions = []
        files_mentioned = set()
        
        for p in pairs:
            query = p.get("query", "")
            response = p.get("response", "")
            
            # Извлекаем упоминания файлов
            import re
            file_matches = re.findall(r'\b[\w/]+\.[a-z]{1,4}\b', query + " " + response)
            files_mentioned.update(file_matches[:5])
            
            # Краткое описание запроса
            if len(query) > 20:
                topics.add(query[:100])
        
        # Формируем summary
        parts = ["=== SUMMARY БЕСЕДЫ ==="]
        
        if topics:
            parts.append("\nОбсуждённые темы:")
            for i, topic in enumerate(list(topics)[:10], 1):
                parts.append(f"  {i}. {topic}")
        
        if files_mentioned:
            parts.append("\nУпомянутые файлы:")
            for f in sorted(files_mentioned)[:20]:
                parts.append(f"  - {f}")
        
        parts.append(f"\nВсего обменов: {len(pairs)}")
        
        return "\n".join(parts)
