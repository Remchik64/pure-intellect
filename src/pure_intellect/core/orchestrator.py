"""Главный пайплайн Оркестратора: Intent → RAG → Graph → Assembler → LLM."""

import logging
from typing import Optional, Generator
from dataclasses import dataclass, field

from ..engine.model_manager import ModelManager
from .intent import IntentDetector, IntentResult
from .retriever import Retriever
from .assembler import ContextAssembler
from .graph_builder import GraphBuilder
from .card_generator import CardGenerator
from .memory import WorkingMemory, MemoryStorage, MemoryOptimizer, AttentionScorer, CCITracker, ImportanceTagger

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Результат работы Оркестратора."""
    query: str
    intent: IntentResult
    context_cards: list = field(default_factory=list)
    graph_entities: list = field(default_factory=list)
    system_prompt: str = ""
    response: str = ""
    model_used: str = ""
    tokens_prompt: int = 0
    tokens_completion: int = 0
    coherence_score: float = 1.0
    coherence_signal: str = "coherent"

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "intent": {
                "type": self.intent.intent.value,
                "confidence": self.intent.confidence,
                "entities": self.intent.entities,
            },
            "context_cards": len(self.context_cards),
            "graph_entities": len(self.graph_entities),
            "response": self.response,
            "model_used": self.model_used,
            "tokens": {
                "prompt": self.tokens_prompt,
                "completion": self.tokens_completion,
            },
            "coherence": {
                "score": round(self.coherence_score, 3),
                "signal": self.coherence_signal,
            },
        }


class OrchestratorPipeline:
    """Главный пайплайн: связывает все модули Оркестратора."""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager(cache_dir="./models")
        self.intent_detector = IntentDetector(model_manager=self.model_manager)
        self.retriever = Retriever()
        self.assembler = ContextAssembler()
        self.graph_builder = GraphBuilder()
        self.card_generator = CardGenerator()
        
        # ── Memory Subsystem ──
        self.memory_storage = MemoryStorage()
        self.working_memory = WorkingMemory(
            token_budget=1500,
            storage=self.memory_storage,
        )
        self.memory_optimizer = MemoryOptimizer(
            hot_retrieval_threshold=3,
            cold_weight_threshold=0.1,
            run_every_n_turns=5,
        )
        self._scorer = AttentionScorer()
        self._tagger = ImportanceTagger()  # P3: LLM-based importance tagging
        self._turn: int = 0  # счётчик turns
        self._chat_history: list = []  # rolling window разговора
        self._context_window_size: int = 12  # max messages (6 turns)
        
        # ── CCI Tracker ──
        self.cci_tracker = CCITracker(
            history_size=10,
            threshold=0.15,
        )
    
    def _create_coordinate(self, chat_history: list) -> str:
        """Создать координату сессии через Ollama — дистиллят истории разговора.
        
        Координата = компактное резюме всего что важно сохранить при soft reset.
        """
        import json
        import urllib.request
        
        # Собираем текст истории
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}"
            for m in chat_history[-10:]  # Последние 10 сообщений
        )
        
        prompt = (
            "Проанализируй этот разговор и создай КРАТКУЮ координату (3-7 строк).\n"
            "Включи: имена участников, ключевые факты, текущую тему, открытые вопросы.\n"
            "Отвечай ТОЛЬКО на русском, без пояснений, только сама координата.\n\n"
            f"РАЗГОВОР:\n{history_text}\n\nКООРДИНАТА:"
        )
        
        try:
            payload = json.dumps({
                "model": "qwen2.5:3b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 300}
            }).encode("utf-8")
            
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                coordinate = data.get("response", "").strip()
                if coordinate:
                    logger.info(f"  [soft_reset] Coordinate created: {coordinate[:80]}...")
                    return coordinate
        except Exception as e:
            logger.warning(f"  [soft_reset] Coordinate creation failed: {e}")
        
        # Fallback: простая суммаризация без LLM
        key_messages = [m for m in chat_history if m["role"] == "user"][-3:]
        return "Контекст разговора: " + " | ".join(m["content"][:100] for m in key_messages)
    
    def _soft_reset(self) -> None:
        """Мягкий сброс контекста с сохранением координаты.
        
        1. Создаёт координату из текущей истории через LLM
        2. Сохраняет координату как anchor fact (не decay, не evict)
        3. Обрезает chat_history до последних 3 turns (6 messages)
        """
        if not self._chat_history:
            return
        
        logger.info(f"  [soft_reset] Triggered at turn {self._turn}, history={len(self._chat_history)} messages")
        
        # Шаг 1: Создаём координату
        coordinate = self._create_coordinate(self._chat_history)
        
        # Шаг 2: Сохраняем как anchor fact
        if coordinate:
            self.working_memory.add_anchor(
                content=coordinate,
                source=f"coordinate_turn_{self._turn}"
            )
        
        # Шаг 3: Обрезаем историю — оставляем последние 6 messages (3 turns)
        self._chat_history = self._chat_history[-6:]
        logger.info(f"  [soft_reset] History trimmed to {len(self._chat_history)} messages")

    def run(
        self,
        query: str,
        model_key: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_llm_intent: bool = False,
    ) -> OrchestrationResult:
        """Выполнить полный пайплайн Оркестратора."""
        
        logger.info(f"🔄 Orchestration started: {query[:80]}...")
        
        # ── Rolling Window: добавляем запрос в историю ──
        self._chat_history.append({"role": "user", "content": query})
        
        # ── Soft Reset: если история превысила лимит — создаём координату ──
        if len(self._chat_history) > self._context_window_size:
            self._soft_reset()
        
        
        # ── CCI: Оценка связности контекста ──
        coherence_result = self.cci_tracker.evaluate(query)
        logger.info(
            f"  [cci] score={coherence_result.score:.3f}, "
            f"signal={coherence_result.signal}"
        )
        
        # При потере coherence — подсказываем памяти что нужно восстановить контекст
        if coherence_result.needs_context_restore():
            logger.info("  [cci] Low coherence detected — restoring context from memory")
            recent_keywords = self.cci_tracker.get_recent_keywords(n_turns=3)
            if recent_keywords:
                # Ищем в storage факты связанные с предыдущим контекстом
                keyword_query = " ".join(list(recent_keywords)[:10])
                restored = self.memory_storage.retrieve(keyword_query, top_k=5)
                for fact in restored:
                    self.working_memory.add(fact)
                logger.info(f"  [cci] Restored {len(restored)} facts from storage")
        
        # ── Шаг 1: Определить Intent ──
        logger.info("  [1/5] Intent detection...")
        intent = self.intent_detector.detect(query, use_llm=use_llm_intent)
        logger.info(f"        Intent: {intent.intent.value} (confidence: {intent.confidence:.2f})")
        
        # ── Шаг 2: Извлечь контекст через RAG ──
        logger.info("  [2/5] RAG retrieval...")
        context_cards = []
        try:
            rag_results = self.retriever.search_by_intent(intent.intent.value)
            if rag_results:
                context_cards = rag_results
                logger.info(f"        Found {len(context_cards)} context cards")
            else:
                # Fallback: поиск по сущностям
                for entity in intent.entities[:3]:
                    cards = self.retriever.search(entity, top_k=2)
                    context_cards.extend(cards)
                logger.info(f"        Found {len(context_cards)} cards via entity search")
        except Exception as e:
            logger.warning(f"        RAG failed: {e}")
        
        # ── Шаг 3: Граф-связи ──
        logger.info("  [3/5] Graph search...")
        graph_entities = []
        try:
            if intent.entities:
                for entity in intent.entities[:3]:
                    results = self.graph_builder.search_nodes(entity, limit=3)
                    graph_entities.extend(results)
                logger.info(f"        Found {len(graph_entities)} graph entities")
        except Exception as e:
            logger.warning(f"        Graph search failed: {e}")
        
        # ── Шаг 4: Собрать контекст (Assembler) ──
        logger.info("  [4/5] Context assembly...")
        system_prompt = system or self._build_system_prompt(intent, context_cards, graph_entities)
        
        # Используем _build_messages (внутренний метод) вместо несуществующего assemble
        messages = self._build_messages(query, intent, context_cards, graph_entities, system)
        logger.info(f"        Assembled {len(messages)} messages")
        
        # ── Шаг 5: Генерация ответа ──
        logger.info("  [5/5] LLM generation...")
        if model_key is None:
            # Автовыбор модели по intent
            model_key = self._select_model(intent)
        
        response_text, tokens_prompt, tokens_completion = self._generate(
            messages=messages,
            model_key=model_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info(f"        Generated {tokens_completion} tokens")
        
        # ── Rolling Window: добавляем ответ в историю ──
        self._chat_history.append({"role": "assistant", "content": response_text})
        
        result = OrchestrationResult(
            query=query,
            intent=intent,
            context_cards=context_cards,
            graph_entities=graph_entities,
            system_prompt=system_prompt,
            response=response_text,
            model_used=model_key,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            coherence_score=coherence_result.score,
            coherence_signal=coherence_result.signal,
        )
        
        # ── Memory Update ──
        self._turn += 1
        try:
            # P3: LLM-based importance tagging
            tagging = self._tagger.tag(query, response_text)
            
            # Anchors → add_anchor() (не decay, не evict)
            for anchor_content in tagging.anchors:
                if anchor_content.strip():
                    self.working_memory.add_anchor(
                        content=anchor_content,
                        source=f"tagger_turn_{self._turn}"
                    )
            
            # Facts → add_text() (обычный lifecycle)
            for fact_content in tagging.facts:
                if fact_content.strip():
                    self.working_memory.add_text(
                        fact_content,
                        source=f"tagger_turn_{self._turn}"
                    )
            
            logger.debug(
                f"  [tagger/{tagging.method}] "
                f"anchors={len(tagging.anchors)}, "
                f"facts={len(tagging.facts)}, "
                f"transient={len(tagging.transient)}"
            )
            
            # Обновляем веса по тексту разговора
            self.working_memory.cleanup(
                turn=self._turn,
                query=query,
                response=response_text,
            )
            
            # Периодическая оптимизация
            self.memory_optimizer.run_if_needed(
                self.working_memory,
                self.memory_storage,
                current_turn=self._turn,
            )
            
            logger.info(
                f"  [memory] turn={self._turn}, "
                f"working={self.working_memory.size()}, "
                f"storage={self.memory_storage.size()}"
            )
            
            # CCI: фиксируем turn в истории связности
            self.cci_tracker.add_turn(
                query=query,
                response=response_text,
                coherence_score=coherence_result.score,
            )
        except Exception as e:
            logger.warning(f"  [memory] update failed: {e}")
        
        logger.info(f"✅ Orchestration complete: {tokens_completion} tokens generated")
        return result
    
    def run_stream(
        self,
        query: str,
        model_key: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """Стриминг-версия пайплайна."""
        
        # Шаги 1-4 (как в run)
        intent = self.intent_detector.detect(query, use_llm=False)
        
        context_cards = []
        try:
            context_cards = self.retriever.search_by_intent(intent.intent.value) or []
        except Exception:
            pass
        
        graph_entities = []
        try:
            if intent.entities:
                for entity in intent.entities[:3]:
                    results = self.graph_builder.search_nodes(entity, limit=3)
                    graph_entities.extend(results)
        except Exception:
            pass
        
        messages = self.assembler.build_messages(
            query=query,
            intent_result=intent,
        )
        
        # Шаг 5: Стриминг генерации
        if model_key is None:
            model_key = self._select_model(intent)
        
        # Загрузить модель если нужно
        if self.model_manager.loaded_model is None:
            self.model_manager.load(model_key, n_gpu_layers=-1)
        
        stream = self.model_manager.loaded_model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content
    
    def _select_model(self, intent: IntentResult) -> str:
        """Автовыбор модели по типу задачи."""
        from .intent import IntentType
        
        # Тяжёлые задачи → большая модель
        heavy_intents = {
            IntentType.CODE_GENERATION,
            IntentType.CODE_REVIEW,
            IntentType.REFACTOR,
            IntentType.ARCHITECTURE,
        }
        
        if intent.intent in heavy_intents:
            return "qwen2.5-coder-7b"  # Модель Чата
        else:
            return "qwen2.5-3b"  # Модель Оркестратора
    
    def _build_system_prompt(self, intent: IntentResult, cards: list, graph: list) -> str:
        """Построить system prompt из контекста."""
        parts = ["Ты — Чистый Интеллект, локальный AI-оркестратор для разработки."]
        
        # Intent
        if intent and hasattr(intent, 'intent') and intent.intent:
            parts.append(f"\n## Тип задачи: {intent.intent.value}")
        
        # Контекст кода
        if cards:
            parts.append("\n## Релевантный код:")
            for card in cards[:5]:
                try:
                    entity = getattr(card, 'entity', None)
                    name = getattr(entity, 'name', 'Unknown') or 'Unknown'
                    etype = getattr(entity, 'type', 'unknown') or 'unknown'
                    fpath = getattr(entity, 'file_path', 'Unknown') or 'Unknown'
                    summary = getattr(card, 'summary', '') or ''
                    code = getattr(entity, 'code', '') or ''
                    
                    parts.append(f"\n### {name} ({etype})")
                    parts.append(f"Файл: {fpath}")
                    if summary:
                        parts.append(f"Описание: {summary}")
                    if code:
                        parts.append(f"```python\n{code[:500]}\n```")
                except Exception:
                    continue
        
        # Граф-связи
        if graph:
            parts.append("\n## Связанные сущности:")
            for node in graph[:5]:
                if isinstance(node, dict):
                    parts.append(f"- {node.get('name', '?')} ({node.get('type', '?')})")
                else:
                    parts.append(f"- {str(node)}")
        
        # ── Память: активный контекст из WorkingMemory ──
        memory_context = self.working_memory.get_context(max_tokens=500)
        if memory_context:
            parts.append("\n## Контекст из памяти:")
            parts.append(memory_context)
        
        parts.append("\n## Инструкции:")
        parts.append("- Отвечай на русском языке")
        parts.append("- Используй контекст кода выше для точных ответов")
        parts.append("- При генерации кода следуй PEP 8 и SOLID")
        
        return "\n".join(parts)

    def _build_messages(self, query, intent, cards, graph_entities, system_override):
        """Собрать messages[] для LLM."""
        # System prompt
        if system_override:
            system = system_override
        else:
            system = self._build_system_prompt(intent, cards, graph_entities)
        
        # Rolling window: system + история + текущий запрос
        messages = [{"role": "system", "content": system}]
        
        # Добавляем историю (всё кроме текущего запроса который уже в chat_history[-1])
        if len(self._chat_history) > 1:
            messages.extend(self._chat_history[:-1])
        
        # Текущий запрос всегда последним
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def _generate(self, messages: list, model_key: str, temperature: float, max_tokens: int) -> tuple:
        """Сгенерировать ответ."""
        # Загрузить модель если нужно
        if self.model_manager.loaded_model is None:
            self.model_manager.load(model_key, n_gpu_layers=-1)
        
        response = self.model_manager.loaded_model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        text = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        tokens_prompt = usage.get("prompt_tokens", 0)
        tokens_completion = usage.get("completion_tokens", 0)
        
        return text, tokens_prompt, tokens_completion
    
    def memory_stats(self) -> dict:
        """Статистика состояния памяти."""
        return {
            "turn": self._turn,
            "working_memory": self.working_memory.stats(),
            "storage": self.memory_storage.stats(),
            "optimizer": self.memory_optimizer.optimizer_stats(),
        }
    
    def memory_clear(self) -> None:
        """Очистить рабочую память (переместить в storage)."""
        self.working_memory.clear()
        logger.info("Working memory cleared")
