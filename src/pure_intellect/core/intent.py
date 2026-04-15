"""Intent Detection — определение типа запроса пользователя."""

import logging
import json
import re
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """Типы запросов."""
    CHAT = "chat"                    # Общий вопрос, диалог
    CODE_GENERATION = "code_generation"  # Написать код
    CODE_REVIEW = "code_review"      # Ревью кода
    CODE_EXPLAIN = "code_explain"    # Объяснить код
    DEBUG = "debug"                  # Отладка, поиск ошибок
    REFACTOR = "refactor"            # Рефакторинг
    ARCHITECTURE = "architecture"     # Архитектурные вопросы
    SEARCH = "search"                # Поиск по проекту


@dataclass
class IntentResult:
    """Результат определения намерения."""
    intent: IntentType
    confidence: float               # 0.0 - 1.0
    entities: list[str]             # Извлечённые сущности (файлы, функции, классы)
    keywords: list[str]             # Ключевые слова
    reasoning: str                  # Объяснение почему такой intent
    suggested_context: list[str]    # Какой контекст нужен


class IntentDetector:
    """Детектор намерений пользователя."""
    
    # Правила для быстрого определения (без LLM)
    RULES = {
        IntentType.DEBUG: [
            r"почему", r"ошибка", r"баг", r"падает", r"не работает",
            r"исключение", r"exception", r"error", r"fix", r"починить",
            r"500", r"404", r"crash", r"segmentation", r"traceback",
        ],
        IntentType.CODE_GENERATION: [
            r"написать", r"создать", r"сгенерируй", r"напиши",
            r"сделай", r"реализуй", r"implement", r"write", r"create",
            r"generate", r"добавь функцию", r"добавь класс",
        ],
        IntentType.CODE_REVIEW: [
            r"ревью", r"проверь код", r"review", r"оценить",
            r"критика", r"улучшить", r"best practice",
        ],
        IntentType.CODE_EXPLAIN: [
            r"объясни", r"что делает", r"как работает", r"explain",
            r"разбер", r"разъясни", r"что значит",
        ],
        IntentType.REFACTOR: [
            r"рефактор", r"refactor", r"переписать", r"оптимизируй",
            r"улучши", r"упрости", r"clean", r"optimize",
        ],
        IntentType.ARCHITECTURE: [
            r"архитектур", r"architecture", r"дизайн", r"паттерн",
            r"структур", r"модуль", r"компоновка", r"схема",
        ],
        IntentType.SEARCH: [

            r"найди", r"поиск", r"где находится", r"search",
            r"locate", r"find", r"какой файл",
        ],
    }
    
    # Промпт для LLM-based определения
    LLM_PROMPT = """Ты — анализатор запросов разработчика. Определи тип запроса.

Типы запросов:
- chat: общий вопрос, диалог
- code_generation: написать новый код
- code_review: проверить существующий код
- code_explain: объяснить как работает код
- debug: найти и исправить ошибку
- refactor: улучшить/переписать код
- architecture: вопрос о структуре/дизайне
- search: найти что-то в проекте

Запрос: {query}

Ответь ТОЛЬКО JSON:
{{
  "intent": "<тип>",
  "confidence": <0.0-1.0>,
  "entities": ["<сущности из запроса: файлы, функции, классы>"],
  "keywords": ["<ключевые слова>"],
  "reasoning": "<почему такой тип>",
  "suggested_context": ["<какой контекст нужен для ответа>"]
}}"""
    
    def __init__(self, model_manager=None):
        self.model_manager = model_manager
    
    def detect_rules(self, query: str) -> Optional[IntentResult]:
        """Быстрое определение через правила (без LLM)."""
        query_lower = query.lower()
        scores = {}
        
        for intent_type, patterns in self.RULES.items():
            score = 0
            matched_keywords = []
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
                    matched_keywords.append(pattern)
            if score > 0:
                scores[intent_type] = (score, matched_keywords)
        
        if not scores:
            return None
        
        # Берём intent с наибольшим score
        best_intent = max(scores.keys(), key=lambda k: scores[k][0])
        score, keywords = scores[best_intent]
        
        # Нормализуем confidence
        confidence = min(score / 3.0, 1.0)  # 3+ совпадений = 100%
        
        # Извлекаем сущности (простые паттерны)
        entities = self._extract_entities(query)
        
        # Определяем нужный контекст
        context = self._suggest_context(best_intent, entities)
        
        return IntentResult(
            intent=best_intent,
            confidence=confidence,
            entities=entities,
            keywords=keywords,
            reasoning=f"Найдено {score} совпадений для {best_intent.value}",
            suggested_context=context,
        )
    
    def detect_llm(self, query: str) -> IntentResult:
        """Определение через LLM (Модель Оркестратора)."""
        if self.model_manager is None or self.model_manager.loaded_model is None:
            raise RuntimeError("Model not loaded for LLM-based intent detection")
        
        prompt = self.LLM_PROMPT.format(query=query)
        
        response = self.model_manager.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Низкая температура для точности
        )
        
        # Парсим JSON из ответа (robust вариант)
        data = self._parse_json_response(response)
        if data:
            try:
                return IntentResult(
                    intent=IntentType(data.get("intent", "chat")),
                    confidence=float(data.get("confidence", 0.5)),
                    entities=data.get("entities", []),
                    keywords=data.get("keywords", []),
                    reasoning=data.get("reasoning", ""),
                    suggested_context=data.get("suggested_context", []),
                )
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to build IntentResult from parsed JSON: {e}")
        
        # Fallback
        return IntentResult(
            intent=IntentType.CHAT,
            confidence=0.3,
            entities=[],
            keywords=[],
            reasoning="LLM не смог определить intent, fallback на CHAT",
            suggested_context=[],
        )
    
    def detect(self, query: str, use_llm: bool = False) -> IntentResult:
        """Основной метод определения намерения."""
        # Сначала пробуем правила
        rule_result = self.detect_rules(query)
        
        if rule_result and rule_result.confidence >= 0.7:
            logger.info(f"Intent detected by rules: {rule_result.intent.value} ({rule_result.confidence:.0%})")
            return rule_result
        
        # Если правила не уверены, используем LLM
        if use_llm and self.model_manager and self.model_manager.loaded_model:
            logger.info("Rules confidence low, using LLM...")
            llm_result = self.detect_llm(query)
            
            # Если LLM уверен, используем его результат
            if llm_result.confidence >= 0.6:
                return llm_result
        
        # Возвращаем лучший из rule-based или дефолт
        return rule_result or IntentResult(
            intent=IntentType.CHAT,
            confidence=0.5,
            entities=[],
            keywords=[],
            reasoning="Нет совпадений, дефолтный CHAT",
            suggested_context=["general"],
        )
    
    def _parse_json_response(self, response: str) -> dict | None:
        """Надёжный парсинг JSON из ответа LLM.
        
        Попытки (от строгого к мягкому):
        1. json.loads(response) — если LLM вернул чистый JSON
        2. Найти '{' ... '}' по границам — если JSON обёрнут в текст
        3. Вернуть None — если JSON не найден
        """
        if not response:
            return None
        
        # Попытка 1: весь ответ — JSON
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Попытка 2: найти JSON по первому '{' и последнему '}'
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response[start:end + 1])
            except json.JSONDecodeError:
                pass
        
        # Попытка 3: найти JSON-блок в markdown ```json ... ```
        import re as _re
        md_match = _re.search(r'```(?:json)?\s*({.*?})\s*```', response, _re.DOTALL)
        if md_match:
            try:
                return json.loads(md_match.group(1))
            except json.JSONDecodeError:
                pass
        
        logger.warning("Could not extract JSON from LLM response")
        return None
    
    def _extract_entities(self, query: str) -> list[str]:
        """Извлекает сущности из запроса (файлы, функции, классы)."""
        entities = []
        
        # Файлы: *.py, *.js, etc
        file_pattern = r'\b[\w/\\.-]+\.(?:py|js|ts|go|rs|java|cpp|c|h)\b'
        entities.extend(re.findall(file_pattern, query))
        
        # Функции/классы: camelCase или snake_case с заглавной
        func_pattern = r'\b[a-z][a-zA-Z0-9_]*\(["\']?[^)]*["\']?\)'
        entities.extend(re.findall(func_pattern, query))
        
        # Именованные сущности: class MyClass, function myFunc
        named_pattern = r'\b(?:class|function|def|func|method|module)\s+([A-Za-z_][A-Za-z0-9_]*)'
        entities.extend(re.findall(named_pattern, query, re.IGNORECASE))
        
        return list(set(entities))
    
    def _suggest_context(self, intent: IntentType, entities: list[str]) -> list[str]:
        """Предлагает какой контекст нужен для ответа."""
        context_map = {
            IntentType.CHAT: ["general"],
            IntentType.CODE_GENERATION: ["project_structure", "existing_patterns", "dependencies"],
            IntentType.CODE_REVIEW: ["target_files", "related_tests", "coding_standards"],
            IntentType.CODE_EXPLAIN: ["target_files", "call_graph"],
            IntentType.DEBUG: ["target_files", "error_logs", "recent_changes", "dependencies"],
            IntentType.REFACTOR: ["target_files", "call_graph", "test_coverage"],
            IntentType.ARCHITECTURE: ["project_structure", "dependencies", "patterns"],
            IntentType.SEARCH: ["project_index", "file_list"],
        }
        
        context = context_map.get(intent, ["general"])
        
        # Добавляем сущности в контекст
        if entities:
            context.extend([f"entity:{e}" for e in entities[:3]])
        
        return context

    def to_dict(self, result: IntentResult) -> dict:
        """Конвертирует IntentResult в словарь."""
        return asdict(result)
