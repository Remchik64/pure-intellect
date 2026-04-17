"""
Тесты для HardwareDetector.
Проверяем логику рекомендаций без реального железа.
"""
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock

from pure_intellect.utils.hardware_detector import (
    HardwareDetector,
    HardwareInfo,
    GPUInfo,
    ModelRecommendation,
    detect_hardware,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_info(
    vram_mb: int = 0,
    vendor: str = "none",
    ram_mb: int = 16 * 1024,
) -> HardwareInfo:
    info = HardwareInfo(ram_mb=ram_mb)
    if vendor != "none" and vram_mb > 0:
        info.gpu = GPUInfo(
            name=f"Test GPU ({vendor})",
            vram_mb=vram_mb,
            vendor=vendor,
            cuda_available=(vendor == "nvidia"),
        )
    return info


# ── Рекомендации по сценариям ─────────────────────────────────────────────────

class TestRecommendations:
    def setup_method(self):
        self.d = HardwareDetector()

    def test_nvidia_12gb_vram_full_mode(self):
        """RTX 3060 12GB → GPU FULL, qwen2.5:7b генератор."""
        info = make_info(vram_mb=12 * 1024, vendor="nvidia")
        rec = self.d.recommend(info)
        assert rec.mode == "GPU FULL"
        assert rec.generator == "qwen2.5:7b"
        assert rec.coordinator == "qwen2.5:3b"
        assert rec.status == "✅"
        assert rec.num_gpu == 999

    def test_nvidia_6gb_vram_split_mode(self):
        """RTX 3060 6GB → GPU SPLIT, частично на CPU."""
        info = make_info(vram_mb=6 * 1024, vendor="nvidia")
        rec = self.d.recommend(info)
        assert rec.mode == "GPU SPLIT"
        assert rec.generator == "qwen2.5:7b"
        assert rec.num_gpu == 20
        assert len(rec.warnings) > 0

    def test_nvidia_4gb_vram_limited(self):
        """GPU 4GB → GPU LIMITED, оба 3B."""
        info = make_info(vram_mb=4 * 1024, vendor="nvidia")
        rec = self.d.recommend(info)
        assert rec.mode == "GPU LIMITED"
        assert rec.generator == "qwen2.5:3b"
        assert rec.coordinator == "qwen2.5:3b"

    def test_nvidia_2gb_vram_minimal(self):
        """GPU 2GB → GPU MINIMAL, phi3:mini."""
        info = make_info(vram_mb=2 * 1024, vendor="nvidia")
        rec = self.d.recommend(info)
        assert rec.mode == "GPU MINIMAL"
        assert rec.coordinator == "phi3:mini"
        assert rec.generator == "phi3:mini"
        assert rec.status == "⚠️"

    def test_no_gpu_16gb_ram_cpu_only(self):
        """Нет GPU, 16GB RAM → CPU ONLY."""
        info = make_info(vram_mb=0, vendor="none", ram_mb=16 * 1024)
        rec = self.d.recommend(info)
        assert rec.mode == "CPU ONLY"
        assert rec.num_gpu == 0
        assert any("GPU" in w for w in rec.warnings)

    def test_no_gpu_8gb_ram_cpu_minimal(self):
        """Нет GPU, 8GB RAM → CPU MINIMAL, phi3:mini."""
        info = make_info(vram_mb=0, vendor="none", ram_mb=8 * 1024)
        rec = self.d.recommend(info)
        assert rec.mode == "CPU MINIMAL"
        assert rec.coordinator == "phi3:mini"
        assert rec.status == "❌"
        assert len(rec.warnings) >= 2

    def test_apple_silicon_16gb(self):
        """Apple M2 16GB unified → Apple Silicon FULL."""
        info = make_info(vram_mb=16 * 1024, vendor="apple")
        rec = self.d.recommend(info)
        assert rec.mode == "Apple Silicon FULL"
        assert rec.generator == "qwen2.5:7b"
        assert rec.status == "✅"

    def test_apple_silicon_8gb(self):
        """Apple M1 8GB → Apple Silicon (без FULL)."""
        info = make_info(vram_mb=8 * 1024, vendor="apple")
        rec = self.d.recommend(info)
        assert rec.mode == "Apple Silicon"
        assert rec.generator == "qwen2.5:3b"


# ── GPUInfo свойства ──────────────────────────────────────────────────────────

class TestGPUInfo:
    def test_vram_gb_conversion(self):
        gpu = GPUInfo(vram_mb=12288)
        assert gpu.vram_gb == 12.0

    def test_vram_gb_rounding(self):
        gpu = GPUInfo(vram_mb=6144)
        assert gpu.vram_gb == 6.0

    def test_vram_gb_zero(self):
        gpu = GPUInfo(vram_mb=0)
        assert gpu.vram_gb == 0.0


# ── HardwareInfo свойства ─────────────────────────────────────────────────────

class TestHardwareInfo:
    def test_ram_gb_conversion(self):
        info = HardwareInfo(ram_mb=32 * 1024)
        assert info.ram_gb == 32.0

    def test_ram_gb_zero(self):
        info = HardwareInfo(ram_mb=0)
        assert info.ram_gb == 0.0


# ── NVIDIA detection mock ─────────────────────────────────────────────────────

class TestNvidiaDetection:
    def test_detect_nvidia_success(self):
        """Мокируем nvidia-smi и проверяем парсинг."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 3060, 12288, 535.183.01\n"

        with patch("subprocess.run", return_value=mock_result):
            d = HardwareDetector()
            gpu = d._detect_nvidia()

        assert gpu is not None
        assert gpu.vendor == "nvidia"
        assert gpu.vram_mb == 12288
        assert gpu.cuda_available is True
        assert "RTX 3060" in gpu.name

    def test_detect_nvidia_not_found(self):
        """nvidia-smi не найден — возвращает None."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            d = HardwareDetector()
            gpu = d._detect_nvidia()
        assert gpu is None

    def test_detect_nvidia_timeout(self):
        """nvidia-smi завис — возвращает None."""
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 10)):
            d = HardwareDetector()
            gpu = d._detect_nvidia()
        assert gpu is None

    def test_detect_nvidia_error_code(self):
        """nvidia-smi вернул ненулевой код — None."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            d = HardwareDetector()
            gpu = d._detect_nvidia()
        assert gpu is None


# ── detect_and_recommend output format ───────────────────────────────────────

class TestDetectAndRecommend:
    def test_output_structure(self):
        """detect_and_recommend возвращает нужные ключи."""
        d = HardwareDetector()
        # Мокируем detect чтобы не зависеть от реального железа
        with patch.object(d, "detect", return_value=make_info(vram_mb=12*1024, vendor="nvidia")):
            result = d.detect_and_recommend()

        assert "hardware" in result
        assert "recommendation" in result
        assert "errors" in result

        hw = result["hardware"]
        assert "os" in hw
        assert "ram_gb" in hw
        assert "gpu" in hw

        rec = result["recommendation"]
        assert "coordinator" in rec
        assert "generator" in rec
        assert "mode" in rec
        assert "speed_estimate" in rec
        assert "status" in rec
        assert "warnings" in rec

    def test_no_gpu_output(self):
        """Без GPU gpu=None в hardware секции."""
        d = HardwareDetector()
        with patch.object(d, "detect", return_value=make_info(vram_mb=0, vendor="none")):
            result = d.detect_and_recommend()
        assert result["hardware"]["gpu"] is None

    def test_with_gpu_output(self):
        """С GPU: gpu — строка с именем, gpu_info — dict с деталями."""
        d = HardwareDetector()
        with patch.object(d, "detect", return_value=make_info(vram_mb=8*1024, vendor="nvidia")):
            result = d.detect_and_recommend()
        hw = result["hardware"]
        # gpu должен быть строкой (имя GPU), а не объектом
        assert hw["gpu"] is not None
        assert isinstance(hw["gpu"], str), f"Expected str, got {type(hw['gpu'])}: {hw['gpu']}"
        # gpu_info содержит детальную информацию
        gpu_info = hw.get("gpu_info")
        assert gpu_info is not None
        assert "name" in gpu_info
        assert "vram_gb" in gpu_info
        assert "vendor" in gpu_info
        assert "cuda_available" in gpu_info
        # vram_gb доступен на верхнем уровне hardware
        assert isinstance(hw["vram_gb"], (int, float))
        assert hw["vram_gb"] > 0


# ── detect_hardware() convenience function ────────────────────────────────────

class TestDetectHardwareFunction:
    def test_returns_dict(self):
        """detect_hardware() должна вернуть dict без исключений."""
        result = detect_hardware()
        assert isinstance(result, dict)
        assert "hardware" in result
        assert "recommendation" in result

    def test_recommendation_has_valid_model(self):
        """Рекомендованные модели должны быть непустыми строками."""
        result = detect_hardware()
        rec = result["recommendation"]
        assert isinstance(rec["coordinator"], str)
        assert len(rec["coordinator"]) > 0
        assert isinstance(rec["generator"], str)
        assert len(rec["generator"]) > 0
