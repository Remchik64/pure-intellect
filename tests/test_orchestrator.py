"""Тесты для OrchestratorPipeline — основного пайплайна."""

import pytest
from unittest.mock import MagicMock, patch
from pure_intellect.core.orchestrator import OrchestratorPipeline, OrchestrationResult
from pure_intellect.core.intent import IntentResult, IntentType


def make_intent(intent_type: IntentType = IntentType.CHAT) -> IntentResult:
    return IntentResult(
        intent=intent_type,
        confidence=0.8,
        entities=[],
        keywords=[],
        reasoning="test",
        suggested_context=[],
    )


def make_mock_model_manager(response_text: str = "Test response"):
    """Создать mock ModelManager."""
    manager = MagicMock()
    manager.loaded_model = MagicMock()
    manager.loaded_model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": response_text}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20},
    }
    return manager


@pytest.fixture
def mock_pipeline():
    """OrchestratorPipeline со всеми зависимостями замокированными."""
    manager = make_mock_model_manager()

    with patch('pure_intellect.core.orchestrator.IntentDetector') as MockIntent, \
         patch('pure_intellect.core.orchestrator.Retriever') as MockRetriever, \
         patch('pure_intellect.core.orchestrator.ContextAssembler') as MockAssembler, \
         patch('pure_intellect.core.orchestrator.GraphBuilder') as MockGraph, \
         patch('pure_intellect.core.orchestrator.CardGenerator') as MockCard:

        # Настраиваем mock intent detector
        intent_instance = MockIntent.return_value
        intent_instance.detect.return_value = make_intent(IntentType.CHAT)

        # Настраиваем mock retriever
        retriever_instance = MockRetriever.return_value
        retriever_instance.search_by_intent.return_value = []
        retriever_instance.search.return_value = []

        # Настраиваем mock assembler
        assembler_instance = MockAssembler.return_value
        assembler_instance.build_messages.return_value = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "test query"},
        ]

        # Настраиваем mock graph builder
        graph_instance = MockGraph.return_value
        graph_instance.search_nodes.return_value = []

        pipeline = OrchestratorPipeline(model_manager=manager)
        yield pipeline


class TestOrchestrationResult:
    """Тесты dataclass OrchestrationResult."""

    def test_to_dict_returns_dict(self):
        """to_dict() возвращает словарь с нужными ключами."""
        result = OrchestrationResult(
            query="test",
            intent=make_intent(),
            context_cards=[],
            graph_entities=[],
            system_prompt="system",
            response="response",
            model_used="qwen2.5-3b",
            tokens_prompt=50,
            tokens_completion=20,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "query" in d
        assert "intent" in d
        assert "response" in d
        assert "tokens" in d

    def test_to_dict_intent_structure(self):
        """to_dict() содержит правильную структуру intent."""
        result = OrchestrationResult(
            query="test",
            intent=make_intent(IntentType.DEBUG),
            context_cards=[],
            graph_entities=[],
            system_prompt="",
            response="",
            model_used="",
            tokens_prompt=0,
            tokens_completion=0,
        )
        d = result.to_dict()
        assert d["intent"]["type"] == "debug"
        assert "confidence" in d["intent"]
        assert "entities" in d["intent"]

    def test_to_dict_tokens_structure(self):
        """to_dict() содержит правильную структуру tokens."""
        result = OrchestrationResult(
            query="test",
            intent=make_intent(),
            context_cards=[],
            graph_entities=[],
            system_prompt="",
            response="answer",
            model_used="model",
            tokens_prompt=100,
            tokens_completion=50,
        )
        d = result.to_dict()
        assert d["tokens"]["prompt"] == 100
        assert d["tokens"]["completion"] == 50


class TestPipelineInit:
    """Тесты инициализации OrchestratorPipeline."""

    def test_pipeline_creates_with_manager(self):
        """Pipeline создаётся с переданным model_manager."""
        manager = MagicMock()
        with patch('pure_intellect.core.orchestrator.IntentDetector'), \
             patch('pure_intellect.core.orchestrator.Retriever'), \
             patch('pure_intellect.core.orchestrator.ContextAssembler'), \
             patch('pure_intellect.core.orchestrator.GraphBuilder'), \
             patch('pure_intellect.core.orchestrator.CardGenerator'):
            pipeline = OrchestratorPipeline(model_manager=manager)
            assert pipeline.model_manager is manager

    def test_pipeline_has_required_components(self):
        """Pipeline имеет все необходимые компоненты."""
        manager = MagicMock()
        with patch('pure_intellect.core.orchestrator.IntentDetector'), \
             patch('pure_intellect.core.orchestrator.Retriever'), \
             patch('pure_intellect.core.orchestrator.ContextAssembler'), \
             patch('pure_intellect.core.orchestrator.GraphBuilder'), \
             patch('pure_intellect.core.orchestrator.CardGenerator'):
            pipeline = OrchestratorPipeline(model_manager=manager)
            assert hasattr(pipeline, 'intent_detector')
            assert hasattr(pipeline, 'retriever')
            assert hasattr(pipeline, 'assembler')
            assert hasattr(pipeline, 'graph_builder')
