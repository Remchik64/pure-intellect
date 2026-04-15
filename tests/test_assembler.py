"""Тесты для ContextAssembler — сборка контекста для LLM."""

import pytest
from unittest.mock import MagicMock, patch
from pure_intellect.core.assembler import ContextAssembler
from pure_intellect.core.intent import IntentResult, IntentType
from pure_intellect.core.retriever import RetrievalResult


def make_intent(intent_type: IntentType = IntentType.CHAT, confidence: float = 0.8) -> IntentResult:
    """Вспомогательная функция для создания IntentResult."""
    return IntentResult(
        intent=intent_type,
        confidence=confidence,
        entities=[],
        keywords=[],
        reasoning="test",
        suggested_context=[],
    )


def make_retrieval_result(summary: str = "def example(): pass", score: float = 0.9) -> RetrievalResult:
    """Вспомогательная функция для создания RetrievalResult."""
    return RetrievalResult(
        card_id="test_card_001",
        entity_name="example",
        entity_type="function",
        file_path="example.py",
        start_line=1,
        end_line=5,
        summary=summary,
        distance=1.0 - score,
        relevance_score=score,
    )


@pytest.fixture
def mock_retriever():
    """Mock retriever — не обращается к ChromaDB."""
    retriever = MagicMock()
    retriever.search.return_value = []
    retriever.search_by_intent.return_value = []
    retriever.format_context.return_value = ""
    return retriever


@pytest.fixture
def assembler(mock_retriever):
    """Assembler с mock retriever."""
    return ContextAssembler(retriever=mock_retriever)


class TestBuildMessages:
    """Тесты build_messages() — основного метода Assembler."""

    def test_returns_list_of_messages(self, assembler):
        """build_messages возвращает список словарей."""
        messages = assembler.build_messages(query="привет")
        assert isinstance(messages, list)
        assert len(messages) > 0
        for msg in messages:
            assert "role" in msg
            assert "content" in msg

    def test_last_message_is_user(self, assembler):
        """Последнее сообщение всегда user."""
        messages = assembler.build_messages(query="объясни код")
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "объясни код"

    def test_system_message_present(self, assembler):
        """Первое сообщение — system prompt."""
        messages = assembler.build_messages(query="привет")
        assert messages[0]["role"] == "system"
        assert len(messages[0]["content"]) > 0

    def test_system_prompt_changes_by_intent(self, assembler):
        """System prompt меняется в зависимости от intent."""
        debug_intent = make_intent(IntentType.DEBUG)
        code_intent = make_intent(IntentType.CODE_GENERATION)

        msgs_debug = assembler.build_messages("ошибка", intent_result=debug_intent)
        msgs_code = assembler.build_messages("напиши", intent_result=code_intent)

        system_debug = msgs_debug[0]["content"]
        system_code = msgs_code[0]["content"]

        # Системные промпты должны отличаться
        assert system_debug != system_code

    def test_rag_context_injected(self, assembler, mock_retriever):
        """RAG результаты включаются в system prompt."""
        mock_retriever.search_by_intent.return_value = [make_retrieval_result()]
        mock_retriever.format_context.return_value = "[RELEVANT CODE CONTEXT]\ndef example(): pass\n[/RELEVANT CODE CONTEXT]"

        intent = make_intent(IntentType.CODE_GENERATION)
        intent.entities = ["example"]

        messages = assembler.build_messages("что делает example", intent_result=intent)
        system_content = messages[0]["content"]

        assert "RELEVANT CODE CONTEXT" in system_content

    def test_empty_query_handled(self, assembler):
        """Пустой запрос обрабатывается без исключений."""
        messages = assembler.build_messages(query="")
        assert isinstance(messages, list)

    def test_token_budget_not_exceeded(self, assembler):
        """Суммарный размер контекста не превышает max_context_tokens."""
        from pure_intellect.config import settings
        from pure_intellect.utils.tokenizer import count_tokens

        messages = assembler.build_messages(query="тест бюджета токенов")
        total = sum(count_tokens(m["content"]) for m in messages)
        assert total <= settings.max_context_tokens + 200  # небольшой допуск


class TestSystemPromptBuilding:
    """Тесты _build_system_prompt."""

    def test_known_modes(self, assembler):
        """Каждый режим имеет системный промпт."""
        modes = ["debug", "code_generation", "code_explain", "refactor", "architecture", "chat"]
        for mode in modes:
            prompt = assembler._build_system_prompt(mode)
            assert isinstance(prompt, str)
            assert len(prompt) > 10, f"Пустой промпт для режима: {mode}"

    def test_unknown_mode_fallback(self, assembler):
        """Неизвестный режим → дефолтный chat промпт."""
        prompt = assembler._build_system_prompt("unknown_mode_xyz")
        chat_prompt = assembler._build_system_prompt("chat")
        assert prompt == chat_prompt
