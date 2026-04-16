"""Тесты для config_loader и ProviderFactory (Шаг 1: гибкая система моделей)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── config_loader tests ──────────────────────────────────

from pure_intellect.engines.config_loader import (
    AppConfig,
    HardwareConfig,
    MemoryConfig,
    ModelConfig,
    _parse_model_config,
    load_config,
    reload_config,
)


class TestModelConfig:
    """Тесты ModelConfig dataclass."""

    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.provider == "ollama"
        assert cfg.model == "qwen2.5:3b"
        assert cfg.gpu_layers == "auto"
        assert cfg.temperature == 0.7

    def test_resolved_gpu_layers_auto_with_vram(self):
        cfg = ModelConfig(gpu_layers="auto")
        assert cfg.resolved_gpu_layers(available_vram_mb=8192) == -1   # всё на GPU
        assert cfg.resolved_gpu_layers(available_vram_mb=1024) == 20   # частичный
        assert cfg.resolved_gpu_layers(available_vram_mb=0) == 0       # CPU only

    def test_resolved_gpu_layers_explicit_int(self):
        cfg = ModelConfig(gpu_layers=-1)
        assert cfg.resolved_gpu_layers() == -1

        cfg = ModelConfig(gpu_layers=0)
        assert cfg.resolved_gpu_layers() == 0

        cfg = ModelConfig(gpu_layers=20)
        assert cfg.resolved_gpu_layers() == 20

    def test_resolved_gpu_layers_string_int(self):
        cfg = ModelConfig(gpu_layers="15")
        assert cfg.resolved_gpu_layers() == 15

    def test_resolved_gpu_layers_invalid_string(self):
        cfg = ModelConfig(gpu_layers="invalid")
        assert cfg.resolved_gpu_layers() == -1  # fallback to all GPU


class TestParseModelConfig:
    """Тесты _parse_model_config из YAML dict."""

    def test_full_config(self):
        data = {
            "provider": "ollama",
            "model": "qwen2.5:7b",
            "gpu_layers": "auto",
            "temperature": 0.5,
            "max_tokens": 1024,
            "timeout": 60,
            "fallback_model": "qwen2.5:3b",
        }
        cfg = _parse_model_config(data)
        assert cfg.provider == "ollama"
        assert cfg.model == "qwen2.5:7b"
        assert cfg.temperature == 0.5
        assert cfg.max_tokens == 1024
        assert cfg.fallback_model == "qwen2.5:3b"

    def test_minimal_config(self):
        cfg = _parse_model_config({"model": "mistral:7b"})
        assert cfg.model == "mistral:7b"
        assert cfg.provider == "ollama"  # default
        assert cfg.fallback_model is None

    def test_extra_fields_stored(self):
        cfg = _parse_model_config({"model": "test", "custom_param": "value"})
        assert cfg.extra.get("custom_param") == "value"


class TestLoadConfig:
    """Тесты загрузки config.yaml."""

    def test_load_defaults_no_file(self):
        """Без файла возвращаем defaults."""
        with patch("pure_intellect.engines.config_loader._find_config_yaml", return_value=None):
            cfg = load_config()
        assert isinstance(cfg, AppConfig)
        assert cfg.coordinator.model == "qwen2.5:3b"
        assert cfg.generator.model == "qwen2.5:7b"

    def test_load_from_yaml(self):
        """Загрузка из реального YAML файла."""
        yaml_content = """
models:
  coordinator:
    provider: ollama
    model: qwen2.5:3b
    temperature: 0.2
    max_tokens: 400
  generator:
    provider: ollama
    model: llama3.1:8b
    temperature: 0.8
    max_tokens: 2048
    fallback_model: qwen2.5:3b
hardware:
  auto_gpu_layers: true
  vram_reserve_mb: 1024
  vram_overflow_strategy: offload
memory:
  context_window_messages: 10
  keep_after_reset: 4
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp_path = Path(f.name)

        try:
            with patch("pure_intellect.engines.config_loader._find_config_yaml",
                       return_value=tmp_path):
                cfg = load_config()

            assert cfg.coordinator.model == "qwen2.5:3b"
            assert cfg.coordinator.temperature == 0.2
            assert cfg.generator.model == "llama3.1:8b"
            assert cfg.generator.fallback_model == "qwen2.5:3b"
            assert cfg.hardware.vram_reserve_mb == 1024
            assert cfg.hardware.vram_overflow_strategy == "offload"
            assert cfg.memory.context_window_messages == 10
            assert cfg.memory.keep_after_reset == 4
        finally:
            tmp_path.unlink()

    def test_load_invalid_yaml_returns_defaults(self):
        """При битом YAML возвращаем defaults а не crash."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("{ invalid yaml: [[[")
            tmp_path = Path(f.name)

        try:
            with patch("pure_intellect.engines.config_loader._find_config_yaml",
                       return_value=tmp_path):
                cfg = load_config()  # не должен упасть
            assert isinstance(cfg, AppConfig)
        finally:
            tmp_path.unlink()

    def test_load_partial_yaml(self):
        """Частичный YAML — missing sections используют defaults."""
        yaml_content = """
models:
  generator:
    model: mistral:7b
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp_path = Path(f.name)

        try:
            with patch("pure_intellect.engines.config_loader._find_config_yaml",
                       return_value=tmp_path):
                cfg = load_config()

            # coordinator не задан → defaults
            assert cfg.coordinator.model == "qwen2.5:3b"
            # generator задан
            assert cfg.generator.model == "mistral:7b"
            # hardware не задан → defaults
            assert cfg.hardware.auto_gpu_layers is True
        finally:
            tmp_path.unlink()


class TestAppConfigDefaults:
    """Тесты дефолтных значений AppConfig."""

    def test_hardware_defaults(self):
        hw = HardwareConfig()
        assert hw.auto_gpu_layers is True
        assert hw.vram_reserve_mb == 512
        assert hw.cpu_threads == 0
        assert hw.vram_overflow_strategy == "offload"

    def test_memory_defaults(self):
        mem = MemoryConfig()
        assert mem.context_window_messages == 12
        assert mem.keep_after_reset == 6
        assert mem.working_memory_tokens == 500
        assert mem.max_storage_facts == 1000


# ── ProviderFactory tests ─────────────────────────────────

from pure_intellect.engines.provider import (
    OllamaModelProvider,
    ProviderFactory,
    detect_free_vram_mb,
    detect_optimal_gpu_layers,
)


class TestOllamaModelProvider:
    """Тесты OllamaModelProvider."""

    def test_info_no_fallback(self):
        cfg = ModelConfig(provider="ollama", model="qwen2.5:3b")
        provider = OllamaModelProvider(cfg)
        info = provider.info()
        assert info["provider"] == "ollama"
        assert info["configured_model"] == "qwen2.5:3b"
        assert info["active_model"] == "qwen2.5:3b"
        assert info["using_fallback"] is False

    def test_info_with_fallback_config(self):
        cfg = ModelConfig(
            provider="ollama",
            model="qwen2.5:7b",
            fallback_model="qwen2.5:3b",
        )
        provider = OllamaModelProvider(cfg)
        info = provider.info()
        assert info["fallback_model"] == "qwen2.5:3b"
        assert info["using_fallback"] is False  # ещё не упали

    @pytest.mark.asyncio
    async def test_generate_success(self):
        cfg = ModelConfig(model="qwen2.5:3b", temperature=0.5, max_tokens=100)
        provider = OllamaModelProvider(cfg)

        mock_result = MagicMock()
        mock_result.content = "Test response"

        with patch.object(provider, "_get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.chat = AsyncMock(return_value=mock_result)
            mock_get_engine.return_value = mock_engine

            result = await provider.generate(
                messages=[{"role": "user", "content": "Hello"}]
            )

        assert result == "Test response"
        mock_engine.chat.assert_called_once_with(
            messages=[{"role": "user", "content": "Hello"}],
            model="qwen2.5:3b",
            temperature=0.5,
            max_tokens=100,
        )

    @pytest.mark.asyncio
    async def test_generate_temperature_override(self):
        cfg = ModelConfig(model="qwen2.5:3b", temperature=0.5)
        provider = OllamaModelProvider(cfg)

        mock_result = MagicMock(content="response")
        with patch.object(provider, "_get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.chat = AsyncMock(return_value=mock_result)
            mock_get_engine.return_value = mock_engine

            await provider.generate(
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.1,  # override
            )

        call_kwargs = mock_engine.chat.call_args
        assert call_kwargs.kwargs["temperature"] == 0.1  # должен использовать override

    @pytest.mark.asyncio
    async def test_generate_fallback_on_error(self):
        cfg = ModelConfig(
            model="qwen2.5:7b",
            fallback_model="qwen2.5:3b",
        )
        provider = OllamaModelProvider(cfg)

        call_count = 0
        mock_result = MagicMock(content="fallback response")

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs["model"] == "qwen2.5:7b":
                raise ConnectionError("7B unavailable")
            return mock_result

        with patch.object(provider, "_get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.chat = mock_chat
            mock_get_engine.return_value = mock_engine

            result = await provider.generate(
                messages=[{"role": "user", "content": "Hello"}]
            )

        assert result == "fallback response"
        assert provider._active_model == "qwen2.5:3b"  # переключился на fallback

    @pytest.mark.asyncio
    async def test_is_available_true(self):
        cfg = ModelConfig(model="qwen2.5:3b")
        provider = OllamaModelProvider(cfg)

        with patch.object(provider, "_get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.list_models = AsyncMock(return_value=["qwen2.5:3b", "qwen2.5:7b"])
            mock_get_engine.return_value = mock_engine

            assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_false(self):
        cfg = ModelConfig(model="llama3.1:8b")
        provider = OllamaModelProvider(cfg)

        with patch.object(provider, "_get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.list_models = AsyncMock(return_value=["qwen2.5:3b"])
            mock_get_engine.return_value = mock_engine

            assert await provider.is_available() is False


class TestProviderFactory:
    """Тесты ProviderFactory."""

    def setup_method(self):
        ProviderFactory.reset()

    def test_create_ollama_provider(self):
        cfg = ModelConfig(provider="ollama", model="qwen2.5:3b")
        provider = ProviderFactory.create(cfg)
        assert isinstance(provider, OllamaModelProvider)

    def test_create_auto_provider(self):
        """auto провайдер → OllamaModelProvider."""
        cfg = ModelConfig(provider="auto", model="qwen2.5:3b")
        provider = ProviderFactory.create(cfg)
        assert isinstance(provider, OllamaModelProvider)

    def test_create_unknown_provider_fallback(self):
        """Неизвестный провайдер → OllamaModelProvider с warning."""
        cfg = ModelConfig(provider="unknown_future_provider", model="test")
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            provider = ProviderFactory.create(cfg)
        assert isinstance(provider, OllamaModelProvider)
        assert len(w) == 1
        assert "Unknown provider" in str(w[0].message)

    def test_get_coordinator_singleton(self):
        """get_coordinator() возвращает один и тот же объект."""
        p1 = ProviderFactory.get_coordinator()
        p2 = ProviderFactory.get_coordinator()
        assert p1 is p2

    def test_get_generator_singleton(self):
        g1 = ProviderFactory.get_generator()
        g2 = ProviderFactory.get_generator()
        assert g1 is g2

    def test_reset_clears_instances(self):
        p1 = ProviderFactory.get_coordinator()
        ProviderFactory.reset()
        p2 = ProviderFactory.get_coordinator()
        assert p1 is not p2  # новый объект после reset


class TestHardwareDetection:
    """Тесты hardware detection."""

    def test_detect_free_vram_no_gpu(self):
        """Без GPU возвращает 0, не падает."""
        with patch("subprocess.run", side_effect=Exception("No GPU")):
            with patch("torch.cuda.is_available", return_value=False):
                result = detect_free_vram_mb()
        assert result == 0

    def test_detect_optimal_layers_no_vram(self):
        """Нет VRAM → CPU only (0)."""
        with patch("pure_intellect.engines.provider.detect_free_vram_mb", return_value=0):
            layers = detect_optimal_gpu_layers(model_size_gb=7.0)
        assert layers == 0

    def test_detect_optimal_layers_full_vram(self):
        """Много VRAM → все слои на GPU (-1)."""
        with patch("pure_intellect.engines.provider.detect_free_vram_mb", return_value=12000):
            layers = detect_optimal_gpu_layers(model_size_gb=4.7)
        assert layers == -1

    def test_detect_optimal_layers_partial_vram(self):
        """Частичный VRAM → offload часть слоёв."""
        with patch("pure_intellect.engines.provider.detect_free_vram_mb", return_value=3000):
            layers = detect_optimal_gpu_layers(model_size_gb=4.7)
        assert 0 < layers < 32  # частичный offload


class TestDualModelConfigDriven:
    """Тесты config-driven поведения DualModelRouter."""

    def test_loads_from_config_by_default(self):
        """DualModelRouter без аргументов читает из config.yaml."""
        from pure_intellect.core.dual_model import DualModelRouter

        with patch("pure_intellect.core.dual_model._load_models_from_config",
                   return_value=("test-coordinator:3b", "test-generator:7b")):
            router = DualModelRouter()

        assert router.coordinator_model == "test-coordinator:3b"
        assert router.generator_model == "test-generator:7b"

    def test_explicit_args_override_config(self):
        """Явные аргументы override значения из config."""
        from pure_intellect.core.dual_model import DualModelRouter

        router = DualModelRouter(
            coordinator_model="custom:3b",
            generator_model="custom:7b",
        )
        assert router.coordinator_model == "custom:3b"
        assert router.generator_model == "custom:7b"

    def test_reload_from_config(self):
        """reload_from_config() обновляет модели без перезапуска."""
        from pure_intellect.core.dual_model import DualModelRouter

        router = DualModelRouter(
            coordinator_model="old:3b",
            generator_model="old:7b",
        )

        with patch("pure_intellect.core.dual_model._load_models_from_config",
                   return_value=("new:3b", "new:7b")):
            with patch("pure_intellect.engines.config_loader.reload_config"):
                router.reload_from_config()

        assert router.coordinator_model == "new:3b"
        assert router.generator_model == "new:7b"
        assert router._generator_available is None  # кеш сброшен

    def test_config_info_returns_dict(self):
        """config_info() возвращает словарь с информацией."""
        from pure_intellect.core.dual_model import DualModelRouter

        router = DualModelRouter(
            coordinator_model="qwen2.5:3b",
            generator_model="qwen2.5:7b",
        )
        info = router.config_info()
        assert "coordinator" in info
        assert "generator" in info
        assert info["coordinator"]["model"] == "qwen2.5:3b"
        assert info["generator"]["model"] == "qwen2.5:7b"
