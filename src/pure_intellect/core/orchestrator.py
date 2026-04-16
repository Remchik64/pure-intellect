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
from .memory import WorkingMemory, MemoryStorage, MemoryOptimizer, AttentionScorer, CCITracker, ImportanceTagger, MetaCoordinator
from .session import SessionPersistence
from .dual_model import DualModelRouter

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

        # ── P6: Dual Model Router ──
        self._router = DualModelRouter()

        # ── P5: Session Persistence ──
        self._session = SessionPersistence(
            base_dir="storage/sessions",
            session_id="default",
        )

        # ── R1: MetaCoordinator — управление ростом координат ──
        try:
            from pure_intellect.engines.config_loader import get_config as _get_cfg
            _meta_every = _get_cfg().memory.meta_coordinate_every
        except Exception:
            _meta_every = 4
        self._meta_coordinator = MetaCoordinator(
            session_dir=self._session.session_dir,
            meta_every=_meta_every,
        )

        # Загружаем сохранённую сессию если есть
        if self._session.exists:
            result = self._session.load(self.working_memory, self.memory_storage)
            if result["loaded"]:
                self._turn = result["turn"]
                self._chat_history = result["chat_history"]
                logger.info(
                    f"[session] Restored: turn={self._turn}, "
                    f"wm={self.working_memory.size()} facts, "
                    f"history={len(self._chat_history)} msgs"
                )
    
    def _create_coordinate(self, chat_history: list) -> str:
        """Создать координату сессии через coordinator (3B) — дистиллят истории разговора.

        P6: Использует DualModelRouter.coordinate() для структурированной задачи.
        Координата = компактное резюме всего что важно сохранить при soft reset.
        """
        # Собираем только USER сообщения (факты от пользователя)
        user_msgs = [m for m in chat_history if m['role'] == 'user']
        history_text = "\n".join(
            f"USER: {m['content'][:300]}"
            for m in user_msgs[-8:]  # Последние 8 пользовательских сообщений
        )

        # Включаем предыдущие координаты (anchor facts) чтобы сохранить накопленные знания
        existing_anchors = [
            f.content for f in self.working_memory._facts
            if getattr(f, 'is_anchor', False)
        ]
        prev_coords_text = ""
        if existing_anchors:
            prev_coords_text = (
                "ПРЕДЫДУЩИЕ КООРДИНАТЫ (используй эту информацию как базу):\n"
                + "\n---\n".join(existing_anchors[-2:])  # последние 2 координаты
                + "\n\n"
            )

        prompt = (
            "Заполни координату сессии, объединив предыдущие данные с новыми.\n"
            "ВАЖНО: Если в ПРЕДЫДУЩИХ КООРДИНАТАХ есть имя/проект — обязательно перенеси их!\n\n"
            f"{prev_coords_text}"
            f"НОВЫЕ СООБЩЕНИЯ:\n{history_text}\n\n"
            "Заполни шаблон (каждое поле — одна строка):\n"
            "УЧАСТНИК: [имя из предыдущих координат или новых сообщений]\n"
            "ПРОЕКТ: [название проекта из предыдущих координат или новых сообщений]\n"
            "ТЕХНОЛОГИИ: [языки/фреймворки/инструменты]\n"
            "ЦЕЛЬ: [главная задача пользователя]\n"
            "ОБОРУДОВАНИЕ: [GPU/CPU/RAM если упомянуто]\n"
            "ТЕМА: [о чём говорили в последних сообщениях]\n\n"
            "Координата:"
        )

        # P6: coordinator (3B) для структурированной задачи
        messages = [
            {"role": "system", "content": "Ты навигатор памяти. Создавай краткие координаты сессии."},
            {"role": "user", "content": prompt},
        ]
        coordinate = self._router.coordinate(
            messages=messages,
            temperature=0.3,
            max_tokens=300,
        )
        if coordinate.strip():
            logger.info(
                f"  [soft_reset] Coordinate via {self._router.coordinator_model}: "
                f"{coordinate[:80]}..."
            )
            return coordinate.strip()

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
        
        # Шаг 2: Передаём координату в MetaCoordinator (R1)
        if coordinate:
            self._meta_coordinator.add_coordinate(
                content=coordinate,
                turn=self._turn,
            )
            # Если накопилось достаточно координат → создаём мета-координату
            if self._meta_coordinator.needs_meta():
                meta_content = self._create_meta_coordinate()
                if meta_content:
                    self._meta_coordinator.consolidate(
                        meta_content=meta_content,
                        turn=self._turn,
                    )
                    logger.info(
                        f"  [soft_reset] Meta-coordinate created at turn {self._turn}"
                    )

        # Шаг 3: Обрезаем историю — оставляем последние 6 messages (3 turns)
        self._chat_history = self._chat_history[-6:]
        logger.info(f"  [soft_reset] History trimmed to {len(self._chat_history)} messages")

        # P5: Сохраняем сессию при каждом soft reset
        self._session.save(
            working_memory=self.working_memory,
            storage=self.memory_storage,
            chat_history=self._chat_history,
            turn=self._turn,
        )


    def _create_meta_coordinate(self) -> str:
        """Создать мета-координату из накопленных координат (R1).

        Вызывается когда MetaCoordinator.needs_meta() == True.
        3B модель объединяет все активные координаты в одну сжатую мета-координату.
        """
        all_contents = self._meta_coordinator.get_all_active_contents()
        if not all_contents:
            return ""

        history_text = "\n---\n".join(all_contents)
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — архивариус AI системы. Создай краткую мета-координату "
                    "объединяющую несколько координат сессии в одну запись. "
                    "Формат: УЧАСТНИК/ПРОЕКТ/ИСТОРИЯ/КЛЮЧЕВЫЕ РЕШЕНИЯ/ТЕКУЩИЙ ФОКУС. "
                    "Будь кратким — не более 300 слов."
                ),
            },
            {
                "role": "user",
                "content": "Объедини эти координаты в мета-координату:\n\n" + history_text,
            },
        ]

        try:
            meta = self._router.coordinate(
                messages=messages,
                temperature=0.2,
                max_tokens=400,
            )
            return meta.strip() if meta else ""
        except Exception as e:
            logger.error(f"[meta_coord] Failed to create meta-coordinate: {e}")
            return "МЕТА: " + " | ".join(c[:80] for c in all_contents[:3])

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

        # P5: Периодическое сохранение каждые 5 turns (не при soft reset — там уже сохраняем)
        if self._turn % 5 == 0:
            self._session.save(
                working_memory=self.working_memory,
                storage=self.memory_storage,
                chat_history=self._chat_history,
                turn=self._turn,
            )
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
        
        # ── R1: MetaCoordinator — стабильный контекст координат ──
        meta_context = self._meta_coordinator.get_context_for_prompt()
        if meta_context:
            parts.append("\n## Координаты сессии:")
            parts.append(meta_context)

        # ── Горячие факты из WorkingMemory ──
        memory_context = self.working_memory.get_context(max_tokens=400)
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
        """Сгенерировать ответ через DualModelRouter (P6).

        Использует generator (7B если доступен) или coordinator (3B fallback).
        Returns: (text, tokens_prompt, tokens_completion)
        """
        text, tokens_prompt, tokens_completion = self._router.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return text, tokens_prompt, tokens_completion

    def session_info(self) -> dict:
        """Информация о текущей сессии (для API endpoint)."""
        return self._session.info()

    def session_delete(self) -> None:
        """Удалить сохранённую сессию."""
        self._session.delete()
        self._turn = 0
        self._chat_history = []
        self.working_memory.clear()
        logger.info("[session] Session deleted and reset")

    def dual_model_stats(self) -> dict:
        """Статистика dual model router (P6)."""
        return self._router.stats()

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
