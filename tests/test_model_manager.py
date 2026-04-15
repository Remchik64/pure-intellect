"""Тесты для ModelManager — lifecycle, singleton, thread-safety."""

import threading
import pytest
from unittest.mock import MagicMock, patch
from pure_intellect.engine.model_manager import ModelManager


@pytest.fixture(autouse=True)
def reset_singleton():
    """Сбрасывать singleton после каждого теста."""
    ModelManager._instance = None
    yield
    # Cleanup после теста
    if ModelManager._instance is not None:
        ModelManager._instance.dispose()
    ModelManager._instance = None


class TestModelManagerLifecycle:
    """Тесты lifecycle управления моделью."""

    def test_initial_state(self):
        """При создании модель не загружена."""
        manager = ModelManager()
        assert manager.is_loaded() is False
        assert manager.loaded_model_key() is None
        assert manager.loaded_model is None

    def test_dispose_when_not_loaded(self):
        """dispose() без загруженной модели не вызывает исключений."""
        manager = ModelManager()
        manager.dispose()  # не должно бросать исключений
        assert manager.is_loaded() is False

    def test_dispose_clears_state(self):
        """dispose() очищает loaded_model и _loaded_key."""
        manager = ModelManager()
        # Симулируем загруженную модель
        manager.loaded_model = MagicMock()
        manager._loaded_key = "qwen2.5-3b"

        assert manager.is_loaded() is True
        manager.dispose()
        assert manager.is_loaded() is False
        assert manager.loaded_model_key() is None

    def test_dispose_twice_is_safe(self):
        """Двойной dispose() безопасен."""
        manager = ModelManager()
        manager.loaded_model = MagicMock()
        manager._loaded_key = "qwen2.5-3b"

        manager.dispose()
        manager.dispose()  # второй вызов не должен падать
        assert manager.is_loaded() is False

    def test_is_loaded_returns_bool(self):
        """is_loaded() возвращает bool."""
        manager = ModelManager()
        result = manager.is_loaded()
        assert isinstance(result, bool)

    def test_loaded_model_key_returns_none_when_empty(self):
        """loaded_model_key() возвращает None когда модель не загружена."""
        manager = ModelManager()
        assert manager.loaded_model_key() is None


class TestModelManagerSingleton:
    """Тесты singleton поведения."""

    def test_get_instance_returns_same_object(self):
        """get_instance() всегда возвращает один и тот же объект."""
        instance1 = ModelManager.get_instance()
        instance2 = ModelManager.get_instance()
        assert instance1 is instance2

    def test_get_instance_is_model_manager(self):
        """get_instance() возвращает ModelManager."""
        instance = ModelManager.get_instance()
        assert isinstance(instance, ModelManager)

    def test_singleton_thread_safety(self):
        """get_instance() thread-safe при конкурентных вызовах."""
        instances = []
        errors = []

        def get_instance():
            try:
                instances.append(ModelManager.get_instance())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in threads: {errors}"
        assert len(instances) == 10
        # Все должны быть одним объектом
        first = instances[0]
        assert all(i is first for i in instances)


class TestModelManagerRegistry:
    """Тесты работы с registry."""

    def test_list_available_returns_dict(self):
        """list_available() возвращает словарь."""
        manager = ModelManager()
        models = manager.list_available()
        assert isinstance(models, dict)
        assert len(models) > 0

    def test_known_models_in_registry(self):
        """Известные модели присутствуют в registry."""
        manager = ModelManager()
        models = manager.list_available()
        assert "qwen2.5-3b" in models
        assert "qwen2.5-coder-7b" in models

    def test_list_downloaded_returns_list(self, tmp_path):
        """list_downloaded() возвращает список."""
        manager = ModelManager(cache_dir=str(tmp_path))
        downloaded = manager.list_downloaded()
        assert isinstance(downloaded, list)

    def test_list_downloaded_empty_when_no_models(self, tmp_path):
        """list_downloaded() пустой если модели не скачаны."""
        manager = ModelManager(cache_dir=str(tmp_path))
        downloaded = manager.list_downloaded()
        assert downloaded == []

    def test_chat_raises_without_model(self):
        """chat() вызывает RuntimeError если модель не загружена."""
        manager = ModelManager()
        with pytest.raises(RuntimeError, match="No model loaded"):
            manager.chat(messages=[{"role": "user", "content": "test"}])
