"""Воркер для работы с длинными текстами и интернетом через Utility модель."""

import logging
import httpx
from contextor.utils.swap_manager import get_swap_manager
from contextor.core.intent import IntentType, IntentResult
from ddgs import DDGS
from contextor.config import settings

def _get_available_utility_model(preferred: str, fallback: str) -> str:
    """Query Ollama to verify model exists. Fall back to generator model if not."""
    try:
        import urllib.request, json as _json
        resp = urllib.request.urlopen(f"{settings.ollama_url}/api/tags", timeout=2)
        models = [m["name"] for m in _json.loads(resp.read()).get("models", [])]
        if preferred in models:
            return preferred
        # Try prefix match (e.g. 'qwen3.5:9b' matches 'qwen3.5:9b-instruct')
        for m in models:
            if m.startswith(preferred.split(":")[0]):
                return m
        return fallback
    except Exception:
        return fallback



logger = logging.getLogger(__name__)

class UtilityWorker:
    def __init__(self, config):
        self.config = config
        self.swap_manager = get_swap_manager()
        self.ollama_base = settings.ollama_url

    def _chunk_text(self, text: str, chunk_size: int = 12000) -> list[str]:
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def perform_web_search(self, query: str) -> str:
        logger.info(f"[UtilityWorker] Запуск поиска в DDG: {query}")
        try:
            results = DDGS().text(query, max_results=5)
            if not results:
                return "Ничего не найдено в интернете."

            text_content = ""
            for res in results:
                title = res.get("title", "")
                href = res.get("href", "")
                body = res.get("body", "")
                text_content += f"Заголовок: {title}\nСсылка: {href}\nОписание: {body}\n\n"
            return text_content
        except ImportError:
            logger.error("[UtilityWorker] ddgs не установлен! Установите: pip install duckduckgo-search")
            return "Ошибка: библиотека duckduckgo-search не установлена. Выполните: pip install duckduckgo-search"
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
        utility_model_name = _get_available_utility_model(getattr(self.config, 'utility_model', "qwen3.5:9b"), getattr(self.config, 'chat_model', "qwen3.5:9b"))
        # Читаем num_ctx из конфига
        try:
            from contextor.engines.config_loader import get_config
            num_ctx = get_config().memory.num_ctx
        except Exception:
            num_ctx = 4096  # fallback
        logger.info(f"[UtilityWorker] Запрос к {utility_model_name} (num_ctx={num_ctx})")
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{self.ollama_base}/api/generate",
                json={
                    "model": utility_model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_ctx": num_ctx}
                }
            )
            if resp.status_code == 200:
                return resp.json().get('response', '')
            logger.error(f"[UtilityWorker] Ошибка генерации: {resp.status_code}")
            return "Ошибка генерации Утилитарной модели."

    async def run_map_reduce(self, intent_result: IntentResult, query: str = "") -> str:
        raw_text = ""
        intent_val = getattr(intent_result.intent, 'value', str(intent_result.intent))

        if intent_val == "web_search":
            # Приоритет: keywords > извлечённый запрос > полный текст
            if intent_result.keywords:
                search_query = " ".join(intent_result.keywords)
            else:
                # Извлечь поисковый запрос из сообщения пользователя
                import re
                # Удалить типичные фразы-команды
                clean = re.sub(r'(поищи|найди|нагугли|поиск|ищи|спроси)+(в+)?(интернете|сети|вебе|онлайне|гугле|duckduckgo)*', '', query, flags=re.IGNORECASE)
                clean = re.sub(r'(выполни|сделай|помоги)+.*?[.!?]\s*', '', clean, flags=re.IGNORECASE)
                clean = clean.strip()
                search_query = clean if clean else query
            raw_text = self.perform_web_search(search_query)
        elif intent_val == "read_document":
            path = intent_result.entities[0] if intent_result.entities else ""
            raw_text = self.read_local_file(path)
        else:
            return ""

        if not raw_text or len(raw_text.strip()) < 100:
            return raw_text

        chunks = self._chunk_text(raw_text, 12000)
        utility_model_name = _get_available_utility_model(getattr(self.config, 'utility_model', "qwen3.5:9b"), getattr(self.config, 'chat_model', "qwen3.5:9b"))
        generator_model = getattr(self.config, 'chat_model', "qwen3.5:9b")

        logger.info(f"[UtilityWorker] Текст разбит на {len(chunks)} кусков. Начинаем Map-Reduce.")

        await self.swap_manager.acquire_utility(utility_model_name, generator_model)

        try:
            summary = ""
            for i, chunk in enumerate(chunks):
                logger.info(f"[UtilityWorker] Экстракция Части {i+1}/{len(chunks)}")
                if i == 0:
                    prompt = f"Сделай подробную выжимку следующего текста. Сохрани все факты:\n\n{chunk}"
                else:
                    prompt = f"Вот текущая выжимка:\n{summary}\n\nА вот продолжение нового текста:\n{chunk}\n\nОбнови выжимку, аккуратно включив новые важные детали и факты."

                summary = await self._ask_utility_model(prompt)

            return summary
        finally:
            await self.swap_manager.release_utility(utility_model_name, generator_model)
