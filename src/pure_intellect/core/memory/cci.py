"""Context Coherence Index (CCI) — отслеживание связности контекста.

CCI измеряет насколько текущий запрос семантически связан
с предыдущими turns. Если coherence падает — система понимает
что разговор «сменил тему» и нужно восстановить контекст из L1.

Принцип:
- После каждого turn сохраняем CoherenceEntry
- При новом запросе считаем similarity с историей
- Если score < threshold → сигнал о потере coherence
"""

import re
import math
import logging
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

# Настройки по умолчанию
DEFAULT_HISTORY_SIZE = 10        # Количество turns в истории
DEFAULT_COHERENCE_THRESHOLD = 0.15  # Ниже — потеря coherence
MIN_WORD_LEN = 3                  # Минимальная длина слова для BM25

# BM25 параметры
BM25_K1 = 1.5
BM25_B = 0.75


@dataclass
class CoherenceEntry:
    """Запись об одном turn в истории CCI."""
    turn: int
    query: str
    response: str
    keywords: set[str] = field(default_factory=set)
    coherence_score: float = 1.0  # Оценка связности с предыдущим turn
    
    def __post_init__(self):
        if not self.keywords:
            self.keywords = _extract_keywords(self.query + " " + self.response)


@dataclass
class CoherenceResult:
    """Результат оценки coherence для нового запроса."""
    turn: int
    query: str
    score: float           # 0.0 (нет связи) - 1.0 (полная связь)
    is_coherent: bool      # True если score >= threshold
    top_matching_turns: list[int]  # Turns с наибольшей связью
    signal: str            # 'coherent', 'low_coherence', 'topic_switch'
    
    def needs_context_restore(self) -> bool:
        """Нужно ли восстанавливать контекст из L1 (памяти)."""
        return not self.is_coherent


def _extract_keywords(text: str, min_len: int = MIN_WORD_LEN) -> set[str]:
    """Извлечь ключевые слова из текста (без стоп-слов)."""
    STOP_WORDS = {
        "это", "как", "для", "что", "при", "или", "если", "когда",
        "that", "this", "with", "from", "have", "they", "will", "when",
    }
    words = re.findall(r'[а-яёА-ЯЁa-zA-Z][а-яёА-ЯЁa-zA-Z0-9_]*', text.lower())
    return {w for w in words if len(w) >= min_len and w not in STOP_WORDS}


def _bm25_score(
    query_keywords: set[str],
    doc_keywords: set[str],
    avg_doc_len: float,
    corpus_size: int = 1,
) -> float:
    """Упрощённый BM25 score между двумя наборами ключевых слов."""
    if not query_keywords or not doc_keywords:
        return 0.0
    
    doc_len = len(doc_keywords)
    score = 0.0
    
    # Частота термина в документе (TF)
    doc_counter = Counter(doc_keywords)
    query_counter = Counter(query_keywords)
    
    for term in query_keywords:
        if term not in doc_keywords:
            continue
        
        tf = doc_counter[term]
        # BM25 TF normalization
        tf_norm = (tf * (BM25_K1 + 1)) / (
            tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / max(avg_doc_len, 1))
        )
        # Простой IDF (без корпуса — равный для всех)
        idf = 1.0
        score += idf * tf_norm
    
    # Нормализация по длине запроса
    return score / max(len(query_keywords), 1)


class CCITracker:
    """Отслеживает Context Coherence Index между turns.
    
    Принцип работы:
    1. После каждого turn вызываем add_turn(query, response)
    2. Для нового запроса вызываем evaluate(query)
    3. evaluate() возвращает CoherenceResult с оценкой связности
    4. Если is_coherent=False → восстанавливаем контекст из памяти
    """
    
    def __init__(
        self,
        history_size: int = DEFAULT_HISTORY_SIZE,
        threshold: float = DEFAULT_COHERENCE_THRESHOLD,
    ):
        self.history_size = history_size
        self.threshold = threshold
        self._history: list[CoherenceEntry] = []
        self._current_turn: int = 0
    
    def add_turn(
        self,
        query: str,
        response: str,
        coherence_score: float = 1.0,
    ) -> CoherenceEntry:
        """Добавить turn в историю CCI.
        
        Вызывать ПОСЛЕ получения ответа от LLM.
        """
        self._current_turn += 1
        entry = CoherenceEntry(
            turn=self._current_turn,
            query=query,
            response=response,
            coherence_score=coherence_score,
        )
        self._history.append(entry)
        
        # Ограничиваем размер истории
        if len(self._history) > self.history_size:
            self._history.pop(0)
        
        logger.debug(f"CCI: turn {self._current_turn} added, history={len(self._history)}")
        return entry
    
    def evaluate(self, query: str) -> CoherenceResult:
        """Оценить coherence нового запроса с историей.
        
        Args:
            query: Новый запрос пользователя
        
        Returns:
            CoherenceResult с оценкой и сигналом
        """
        if not self._history:
            # Нет истории — первый запрос, всегда coherent
            return CoherenceResult(
                turn=self._current_turn + 1,
                query=query,
                score=1.0,
                is_coherent=True,
                top_matching_turns=[],
                signal="coherent",
            )
        
        query_keywords = _extract_keywords(query)
        
        # Считаем avg_doc_len для BM25
        avg_doc_len = sum(len(e.keywords) for e in self._history) / len(self._history)
        
        # Оцениваем similarity с каждым turn в истории
        scored_turns = []
        for entry in self._history:
            score = _bm25_score(
                query_keywords,
                entry.keywords,
                avg_doc_len,
                corpus_size=len(self._history),
            )
            scored_turns.append((score, entry.turn))
        
        scored_turns.sort(reverse=True)
        
        # Общий score = взвешенное среднее (недавние turn важнее)
        total_score = 0.0
        total_weight = 0.0
        
        for i, entry in enumerate(reversed(self._history)):
            # Экспоненциальное затухание: недавние важнее
            weight = math.exp(-0.3 * i)  # e^0, e^-0.3, e^-0.6, ...
            score = _bm25_score(query_keywords, entry.keywords, avg_doc_len)
            total_score += score * weight
            total_weight += weight
        
        final_score = total_score / max(total_weight, 1e-6)
        
        # Нормализуем в [0, 1]
        # BM25 может быть > 1, делаем soft cap
        final_score = min(1.0, final_score)
        
        is_coherent = final_score >= self.threshold
        top_turns = [turn for _, turn in scored_turns[:3] if _ > 0]
        
        # Определяем сигнал
        if final_score >= self.threshold:
            signal = "coherent"
        elif final_score >= self.threshold * 0.3:
            signal = "low_coherence"
        else:
            signal = "topic_switch"
        
        if not is_coherent:
            logger.info(
                f"CCI: low coherence detected! score={final_score:.3f} "
                f"(threshold={self.threshold}), signal={signal}"
            )
        
        return CoherenceResult(
            turn=self._current_turn + 1,
            query=query,
            score=final_score,
            is_coherent=is_coherent,
            top_matching_turns=top_turns,
            signal=signal,
        )
    
    def get_recent_keywords(self, n_turns: int = 3) -> set[str]:
        """Получить ключевые слова из последних N turns.
        
        Используется для восстановления контекста при low coherence.
        """
        recent = self._history[-n_turns:]
        keywords = set()
        for entry in recent:
            keywords.update(entry.keywords)
        return keywords
    
    def history_size_current(self) -> int:
        """Текущий размер истории."""
        return len(self._history)
    
    def stats(self) -> dict:
        """Статистика CCITracker."""
        if not self._history:
            return {
                "turns": 0,
                "avg_coherence": 0.0,
                "threshold": self.threshold,
                "history_capacity": self.history_size,
            }
        
        avg_coherence = sum(
            e.coherence_score for e in self._history
        ) / len(self._history)
        
        return {
            "turns": self._current_turn,
            "history_size": len(self._history),
            "avg_coherence": round(avg_coherence, 3),
            "threshold": self.threshold,
            "history_capacity": self.history_size,
        }
    
    def reset(self) -> None:
        """Сбросить историю (новая сессия)."""
        self._history.clear()
        self._current_turn = 0
        logger.info("CCITracker reset")
