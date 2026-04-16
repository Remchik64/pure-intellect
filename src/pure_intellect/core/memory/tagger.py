"""ImportanceTagger — LLM-based классификация важности фактов.

P3 улучшение: qwen2.5:3b анализирует каждый turn разговора и классифицирует
информацию на три категории:

- anchors:   Критически важные факты (имена, проекты, конфигурации)
             → add_anchor() — не decay, не evict
- facts:     Важные факты для текущей сессии
             → add_text() — обычный lifecycle
- transient: Временная информация (вопросы, рассуждения)
             → skip — не сохраняем

Fallback на rule-based extraction если Ollama недоступен.
"""

import json
import logging
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Ollama настройки
OLLAMA_BASE_URL = "http://localhost:11434"
TAGGER_MODEL = "qwen2.5:3b"
TAGGER_TIMEOUT = 12  # секунд

# Rule-based patterns для fallback
ANCHOR_PATTERNS = [
    r'(?:меня|его|её|нас|вас) зовут ([A-ZА-Яа-яёЁ][a-zа-яёЁ]+)',
    r'(?:мой|наш|её|его) (?:проект|сервис|продукт|система)[:\s]+([\w-]+)',
    r'(?:я|мы) использую?(?:ем)?\s+([A-Za-z][\w.]+(?:\s+[\d.]+)?)',
    r'GPU[:\s]+([A-Za-z0-9\s]+)',
    r'версия[:\s]+([\d.]+)',
]

FACT_PATTERNS = [
    r'([A-Za-z][a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)',  # технические термины
    r'(\d+(?:\.\d+)?\s*(?:GB|MB|TB|ms|s|%|K|M))',  # числа с единицами
]


@dataclass
class TaggingResult:
    """Результат классификации важности факта."""
    anchors: list[str] = field(default_factory=list)   # Критически важные
    facts: list[str] = field(default_factory=list)     # Важные факты
    transient: list[str] = field(default_factory=list) # Временная информация
    method: str = "llm"  # "llm" или "rule_based"
    
    @property
    def total(self) -> int:
        return len(self.anchors) + len(self.facts) + len(self.transient)


class ImportanceTagger:
    """Классифицирует важность информации из разговора.
    
    Использует LLM (qwen2.5:3b) для интеллектуальной классификации.
    При недоступности LLM — fallback на rule-based extraction.
    """
    
    def __init__(
        self,
        ollama_url: str = OLLAMA_BASE_URL,
        model: str = TAGGER_MODEL,
        timeout: int = TAGGER_TIMEOUT,
    ):
        self._ollama_url = ollama_url
        self._model = model
        self._timeout = timeout
        self._llm_available: bool | None = None
        self._llm_calls: int = 0
        self._fallback_calls: int = 0
    
    def _check_llm_available(self) -> bool:
        """Проверить доступность Ollama (кешируем результат)."""
        if self._llm_available is not None:
            return self._llm_available
        
        try:
            req = urllib.request.Request(
                f"{self._ollama_url}/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                self._llm_available = resp.status == 200
        except Exception:
            self._llm_available = False
        
        status = "available" if self._llm_available else "unavailable"
        logger.info(f"ImportanceTagger: Ollama {status}")
        return self._llm_available
    
    def tag(self, query: str, response: str) -> TaggingResult:
        """Классифицировать важность информации из turn разговора.
        
        Args:
            query: Запрос пользователя
            response: Ответ LLM
        
        Returns:
            TaggingResult с anchors, facts, transient
        """
        if self._check_llm_available():
            result = self._tag_llm(query, response)
            if result is not None:
                self._llm_calls += 1
                return result
        
        # Fallback на rule-based
        self._fallback_calls += 1
        return self._tag_rule_based(query, response)
    
    def _tag_llm(self, query: str, response: str) -> TaggingResult | None:
        """Классификация через Ollama LLM."""
        prompt = (
            "Проанализируй этот диалог и классифицируй информацию.\n"
            "Отвечай ТОЛЬКО JSON без пояснений.\n\n"
            f"USER: {query[:300]}\n"
            f"ASSISTANT: {response[:500]}\n\n"
            "Верни JSON:\n"
            "{\n"
            '  "anchors": ["критически важные факты: имена людей, названия проектов, версии, конфигурация"],\n'
            '  "facts": ["важные факты сессии: технологии, подходы, решения"],\n'
            '  "transient": ["временное: вопросы пользователя, рассуждения, примеры"]\n'
            "}\n\n"
            "Правила:\n"
            "- anchors: то что нужно помнить ВСЕГДА (имя пользователя, название проекта, GPU)\n"
            "- facts: важно для текущей сессии (технические термины, подходы)\n"
            "- transient: можно забыть через несколько turns\n"
            "- Пустые массивы [] если нечего добавить\n"
            "- Максимум 3 элемента в каждом массиве\n"
            "JSON:"
        )
        
        try:
            payload = json.dumps({
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Низкая температура для JSON
                    "num_predict": 200,
                }
            }).encode("utf-8")
            
            req = urllib.request.Request(
                f"{self._ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
                raw = data.get("response", "").strip()
                
                # Парсим JSON из ответа
                parsed = self._parse_json(raw)
                if parsed:
                    return TaggingResult(
                        anchors=parsed.get("anchors", [])[:3],
                        facts=parsed.get("facts", [])[:3],
                        transient=parsed.get("transient", [])[:3],
                        method="llm",
                    )
        except (urllib.error.URLError, TimeoutError, Exception) as e:
            logger.debug(f"ImportanceTagger LLM failed: {e}")
        
        return None
    
    def _parse_json(self, text: str) -> dict | None:
        """Извлечь JSON из текста LLM ответа."""
        # Стратегия 1: прямой парсинг
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Стратегия 2: найти границы {}
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Стратегия 3: markdown ```json ... ```
        try:
            match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except (json.JSONDecodeError, AttributeError):
            pass
        
        logger.debug(f"ImportanceTagger: failed to parse JSON from: {text[:100]}")
        return None
    
    def _tag_rule_based(self, query: str, response: str) -> TaggingResult:
        """Rule-based fallback для классификации."""
        text = f"{query} {response}"
        
        anchors = []
        facts = []
        
        # Ищем anchor patterns
        for pattern in ANCHOR_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:
                anchor = match.strip()
                if anchor and len(anchor) > 2 and anchor not in anchors:
                    anchors.append(anchor)
        
        # Ищем технические факты
        tech_terms = re.findall(
            r'\b([A-Z][a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)+|[A-Z]{2,}[a-zA-Z0-9]*)\b',
            response
        )
        for term in tech_terms[:5]:
            if len(term) > 2 and term not in facts and term not in anchors:
                facts.append(term)
        
        return TaggingResult(
            anchors=anchors[:3],
            facts=facts[:3],
            transient=[],
            method="rule_based",
        )
    
    def stats(self) -> dict:
        """Статистика работы tagger."""
        return {
            "llm_calls": self._llm_calls,
            "fallback_calls": self._fallback_calls,
            "llm_available": self._llm_available,
            "model": self._model,
        }
    
    def __repr__(self) -> str:
        return (
            f"ImportanceTagger(model={self._model}, "
            f"llm_calls={self._llm_calls}, "
            f"fallback_calls={self._fallback_calls})"
        )
