"""AttentionScorer — оценивает релевантность фактов на основе разговора.

Анализирует query и response LLM, находит упоминания фактов
и обновляет их attention_weight:
- Факт упомянут → touch() → вес растёт → остаётся в WorkingMemory
- Факт не упомянут → decay() → вес падает → уходит в Storage
"""

import re
import logging
from dataclasses import dataclass
from .fact import Fact

logger = logging.getLogger(__name__)

# Минимальная длина слова для matching (игнорируем предлоги и союзы)
MIN_WORD_LEN = 4

# Стоп-слова (не учитываем при matching)
STOP_WORDS = {
    # Русские
    "это", "как", "для", "что", "при", "или", "если", "когда",
    "чтобы", "тоже", "также", "можно", "нужно", "будет", "есть",
    "этот", "эта", "эти", "той", "того", "тому", "такой",
    # Английские
    "this", "that", "with", "from", "have", "been", "they",
    "will", "when", "then", "also", "some", "would", "could",
}


@dataclass
class ScoreResult:
    """Результат оценки факта."""
    fact_id: str
    matched: bool          # Факт упомянут в разговоре
    match_count: int       # Количество совпадений
    match_words: list[str] # Слова которые совпали
    weight_before: float
    weight_after: float


class AttentionScorer:
    """Оценивает релевантность фактов на основе текущего разговора.
    
    Принцип:
    1. Извлекаем ключевые слова из query + response
    2. Для каждого факта проверяем пересечение с ключевыми словами
    3. Если есть пересечение → fact.touch() → вес растёт
    4. Если нет → оставляем decay работать естественно
    """
    
    def __init__(
        self,
        min_word_len: int = MIN_WORD_LEN,
        stop_words: set[str] | None = None,
    ):
        self.min_word_len = min_word_len
        self.stop_words = stop_words or STOP_WORDS
    
    def score_facts(
        self,
        facts: list[Fact],
        query: str,
        response: str,
        turn: int,
    ) -> list[ScoreResult]:
        """Оценить все факты на основе query + response.
        
        Args:
            facts: Список фактов из WorkingMemory
            query: Запрос пользователя
            response: Ответ LLM
            turn: Номер текущего turn
        
        Returns:
            Список ScoreResult для каждого факта
        """
        # Извлекаем ключевые слова из разговора
        conversation_words = self._extract_keywords(query + " " + response)
        
        results = []
        matched_count = 0
        
        for fact in facts:
            weight_before = fact.attention_weight
            
            # Ключевые слова факта
            fact_words = self._extract_keywords(fact.content)
            
            # Пересечение
            common = fact_words & conversation_words
            
            if common:
                # Факт упомянут — увеличиваем вес
                fact.touch(turn=turn)
                matched_count += 1
                logger.debug(
                    f"Fact {fact.fact_id[:8]}... matched: {list(common)[:3]}, "
                    f"weight {weight_before:.2f} → {fact.attention_weight:.2f}"
                )
            
            results.append(ScoreResult(
                fact_id=fact.fact_id,
                matched=bool(common),
                match_count=len(common),
                match_words=list(common)[:5],
                weight_before=weight_before,
                weight_after=fact.attention_weight,
            ))
        
        if matched_count:
            logger.info(f"Turn {turn}: scored {matched_count}/{len(facts)} facts matched")
        
        return results
    
    def score_single(
        self,
        fact: Fact,
        query: str,
        response: str,
        turn: int,
    ) -> ScoreResult:
        """Оценить один факт."""
        results = self.score_facts([fact], query, response, turn)
        return results[0]
    
    def extract_facts_from_response(
        self,
        response: str,
        source: str = "llm_response",
    ) -> list[str]:
        """Извлечь потенциальные факты из ответа LLM.
        
        Простая эвристика: разбиваем на предложения.
        Будущее: LLM-based extraction.
        
        Returns:
            Список строк — кандидатов на факты.
        """
        # Разбиваем по предложениям (. ! ?)
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        
        # Фильтруем: только содержательные предложения
        facts = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Минимум 20 символов и 4 слова
            words = sentence.split()
            if len(sentence) >= 20 and len(words) >= 4:
                facts.append(sentence)
        
        return facts[:10]  # Не более 10 фактов за раз
    
    def _extract_keywords(self, text: str) -> set[str]:
        """Извлечь ключевые слова из текста.
        
        Убирает стоп-слова, короткие слова, знаки препинания.
        """
        # Токенизация: только буквы (включая кириллицу)
        words = re.findall(r'[а-яёА-ЯЁa-zA-Z][а-яёА-ЯЁa-zA-Z0-9_]*', text.lower())
        
        keywords = set()
        for word in words:
            if (
                len(word) >= self.min_word_len
                and word not in self.stop_words
            ):
                keywords.add(word)
        
        return keywords
