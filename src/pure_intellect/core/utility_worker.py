"""Воркер для работы с длинными текстами и интернетом через Utility модель."""

import logging
import httpx
from pure_intellect.utils.swap_manager import get_swap_manager
from pure_intellect.core.intent import IntentType, IntentResult
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

class UtilityWorker:
    def __init__(self, config):
        self.config = config
        self.swap_manager = get_swap_manager()
        self.ollama_base = "http://localhost:11434"

    def _chunk_text(self, text: str, chunk_size: int = 12000) -> list[str]:
        """Нарезает текст на куски примерно по 3000 токенов (12000 символов)."""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def perform_web_search(self, query: str) -> str:
        logger.info(f"[UtilityWorker] Запуск поиска в DDG: {query}")
        try:
            results = DDGS().text(query, max_results=3)
            if not results:
                return "Ничего не найдено в интернете."

            text_content = ""
            for res in results:
            text_content += f"Заголовок: {res.get('title', '')}\nСсылка: {res.get('href', '')}\nОписание: {res.get('body', '')}\n\n"
Ссылка: {res.get('href', '')}
Описание: {res.get('body', '')}

"
            return text_content
        except Exception as e:
            logger.error(f"[UtilityWorker] Ошибка поиска DDG: {e}")
            return f"Ошибка поиска: {e}"

    def read_local_file(self, path: str) -> str:
        logger.info(f"[UtilityWorker] Чтение файла: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"[UtilityWorker] Ошибка чтения файла: {e}")
            return f"Не удалось прочитать файл {path}: {e}"

    async def _ask_utility_model(self, prompt: str) -> str:
        utility_model_name = getattr(self.config, 'utility_model', "qwen2.5:7b")
        logger.info(f"[UtilityWorker] Запрос к {utility_model_name} (num_ctx=4096)")
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{self.ollama_base}/api/generate",
                json={
                    "model": utility_model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_ctx": 4096}
                }
            )
            if resp.status_code == 200:
                return resp.json().get('response', '')
            logger.error(f"[UtilityWorker] Ошибка генерации: {resp.status_code}")
            return "Ошибка генерации Утилитарной модели."

    async def run_map_reduce(self, intent_result: IntentResult) -> str:
        """Метод постепенной выжимки (Rolling Summary)."""
        raw_text = ""
        if intent_result.intent == IntentType.WEB_SEARCH:
            query = " ".join(intent_result.keywords) if intent_result.keywords else "новости"
            raw_text = self.perform_web_search(query)
        elif intent_result.intent == IntentType.READ_DOCUMENT:
            path = intent_result.entities[0] if intent_result.entities else ""
            raw_text = self.read_local_file(path)
        else:
            return ""

        if not raw_text or len(raw_text.strip()) < 100:
            return raw_text

        chunks = self._chunk_text(raw_text, 12000)
        utility_model_name = getattr(self.config, 'utility_model', "qwen2.5:7b")
        generator_model = getattr(self.config, 'chat_model', "qwen2.5:7b")

        logger.info(f"[UtilityWorker] Текст разбит на {len(chunks)} кусков. Начинаем Map-Reduce.")

        # 1. Захватываем контроль над VRAM (Убиваем Генератора)
        await self.swap_manager.acquire_utility(utility_model_name, generator_model)

        try:
            # 2. Итеративная адаптация (Rolling Memory)
            summary = ""
            for i, chunk in enumerate(chunks):
                logger.info(f"[UtilityWorker] Экстракция Части {i+1}/{len(chunks)}")
                if i == 0:
                    prompt = f"Сделай подробную выжимку следующего текста. Сохрани все факты:

{chunk}"
                else:
                    prompt = f"Вот текущая выжимка:
{summary}

А вот продолжение нового текста:
{chunk}

Обнови выжимку, аккуратно включив новые важные детали и факты."

                summary = await self._ask_utility_model(prompt)

            return summary
        finally:
            # 3. Возвращаем Генератора в VRAM
            await self.swap_manager.release_utility(utility_model_name, generator_model)
