"""
Hardware Detector — определяет железо пользователя и даёт рекомендации.
Определяет GPU, VRAM, RAM, CPU и подбирает оптимальные модели.
"""
from __future__ import annotations

import platform
import subprocess
import shutil
from dataclasses import dataclass, field
from typing import Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class GPUInfo:
    name: str = "Unknown"
    vram_mb: int = 0
    vendor: str = "unknown"   # nvidia / amd / apple / intel / none
    cuda_available: bool = False
    driver_version: str = ""

    @property
    def vram_gb(self) -> float:
        return round(self.vram_mb / 1024, 1)


@dataclass
class HardwareInfo:
    os: str = ""
    os_version: str = ""
    cpu: str = ""
    cpu_cores: int = 0
    ram_mb: int = 0
    gpu: Optional[GPUInfo] = None
    ollama_available: bool = False
    python_version: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def ram_gb(self) -> float:
        return round(self.ram_mb / 1024, 1)


@dataclass
class ModelRecommendation:
    coordinator: str = "qwen2.5:3b"
    generator: str = "qwen2.5:3b"
    mode: str = "CPU ONLY"
    speed_estimate: str = "~2 tok/sec"
    status: str = "⚠️"
    status_label: str = "Ограниченно"
    num_gpu: int = 0
    warnings: list[str] = field(default_factory=list)
    notes: str = ""


class HardwareDetector:
    """Определяет железо и даёт рекомендации по моделям."""

    def detect(self) -> HardwareInfo:
        info = HardwareInfo()
        info.os = platform.system()
        info.os_version = platform.version()
        info.python_version = platform.python_version()

        # CPU
        try:
            info.cpu = platform.processor() or "Unknown CPU"
            if HAS_PSUTIL:
                info.cpu_cores = psutil.cpu_count(logical=False) or 0
        except Exception as e:
            info.errors.append(f"CPU detection error: {e}")

        # RAM
        try:
            if HAS_PSUTIL:
                info.ram_mb = psutil.virtual_memory().total // (1024 * 1024)
            else:
                # Fallback — /proc/meminfo на Linux
                if info.os == "Linux":
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if line.startswith("MemTotal:"):
                                info.ram_mb = int(line.split()[1]) // 1024
                                break
        except Exception as e:
            info.errors.append(f"RAM detection error: {e}")

        # GPU
        info.gpu = self._detect_gpu(info.os)

        # Ollama
        info.ollama_available = shutil.which("ollama") is not None

        return info

    def _detect_gpu(self, os_name: str) -> Optional[GPUInfo]:
        """Пробуем разные методы определения GPU."""
        # 1. NVIDIA — nvidia-smi
        gpu = self._detect_nvidia()
        if gpu:
            return gpu

        # 2. Apple Silicon — platform
        if os_name == "Darwin":
            gpu = self._detect_apple_silicon()
            if gpu:
                return gpu

        # 3. AMD — rocm-smi (Linux)
        if os_name == "Linux":
            gpu = self._detect_amd()
            if gpu:
                return gpu

        return None

    def _detect_nvidia(self) -> Optional[GPUInfo]:
        import sys
        # Список путей для nvidia-smi (включая типичные Windows пути)
        nvidia_smi_paths = ["nvidia-smi"]
        if sys.platform == "win32":
            nvidia_smi_paths += [
                r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
                r"C:\Windows\System32\nvidia-smi.exe",
            ]

        for smi_path in nvidia_smi_paths:
            try:
                result = subprocess.run(
                    [
                        smi_path,
                        "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split("\n")
                    parts = lines[0].split(",")
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        vram_mb = int(parts[1].strip())
                        driver = parts[2].strip() if len(parts) > 2 else ""
                        return GPUInfo(
                            name=name,
                            vram_mb=vram_mb,
                            vendor="nvidia",
                            cuda_available=True,
                            driver_version=driver,
                        )
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
                continue

        # Ollama fallback: если Ollama использует GPU — берём инфо оттуда
        try:
            import urllib.request, json as _json
            req = urllib.request.urlopen("http://localhost:11434/api/ps", timeout=3)
            data = _json.loads(req.read())
            models = data.get("models", [])
            for m in models:
                vram = m.get("size_vram", 0)
                if vram and vram > 0:
                    return GPUInfo(
                        name="NVIDIA GPU (via Ollama)",
                        vram_mb=int(vram / 1024 / 1024),
                        vendor="nvidia",
                        cuda_available=True,
                        driver_version="",
                    )
        except Exception:
            pass

        return None

    def _detect_apple_silicon(self) -> Optional[GPUInfo]:
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "Apple" in result.stdout:
                # Apple Silicon — unified memory, читаем RAM как VRAM
                ram_mb = 0
                if HAS_PSUTIL:
                    ram_mb = psutil.virtual_memory().total // (1024 * 1024)
                chip = result.stdout.strip()
                return GPUInfo(
                    name=chip,
                    vram_mb=ram_mb,  # unified memory
                    vendor="apple",
                    cuda_available=False,
                    driver_version="Metal",
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None

    def _detect_amd(self) -> Optional[GPUInfo]:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return GPUInfo(
                    name="AMD GPU (ROCm)",
                    vram_mb=0,
                    vendor="amd",
                    cuda_available=False,
                    driver_version="ROCm",
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None

    def recommend(self, info: HardwareInfo) -> ModelRecommendation:
        """На основе железа даёт рекомендации по моделям."""
        rec = ModelRecommendation()
        vram_gb = info.gpu.vram_gb if info.gpu else 0
        ram_gb = info.ram_gb
        vendor = info.gpu.vendor if info.gpu else "none"

        # Apple Silicon — unified memory, отличная производительность
        if vendor == "apple":
            if vram_gb >= 16:
                rec.coordinator = "qwen2.5:3b"
                rec.generator = "qwen2.5:7b"
                rec.mode = "Apple Silicon FULL"
                rec.speed_estimate = "~20 tok/sec"
                rec.status = "✅"
                rec.status_label = "Отлично"
                rec.num_gpu = 999
            else:
                rec.coordinator = "qwen2.5:3b"
                rec.generator = "qwen2.5:3b"
                rec.mode = "Apple Silicon"
                rec.speed_estimate = "~12 tok/sec"
                rec.status = "✅"
                rec.status_label = "Хорошо"
                rec.num_gpu = 999
            return rec

        # NVIDIA / AMD GPU
        if vram_gb >= 10:
            rec.coordinator = "qwen2.5:3b"
            rec.generator = "qwen2.5:7b"
            rec.mode = "GPU FULL"
            rec.speed_estimate = "~15 tok/sec"
            rec.status = "✅"
            rec.status_label = "Отлично"
            rec.num_gpu = 999
            rec.notes = "Обе модели полностью в VRAM"

        elif vram_gb >= 6:
            rec.coordinator = "qwen2.5:3b"
            rec.generator = "qwen2.5:7b"
            rec.mode = "GPU SPLIT"
            rec.speed_estimate = "~8 tok/sec"
            rec.status = "✅"
            rec.status_label = "Хорошо"
            rec.num_gpu = 20  # ~20 слоёв на GPU из 28
            rec.notes = "Генератор частично на CPU"
            rec.warnings.append("7B модель будет частично на CPU — это нормально")

        elif vram_gb >= 3:
            rec.coordinator = "qwen2.5:3b"
            rec.generator = "qwen2.5:3b"
            rec.mode = "GPU LIMITED"
            rec.speed_estimate = "~6 tok/sec"
            rec.status = "✅"
            rec.status_label = "Удовлетворительно"
            rec.num_gpu = 999
            rec.notes = "Рекомендуем 3B модели для обоих ролей"
            rec.warnings.append("7B модель не рекомендуется с < 4GB VRAM")

        elif vram_gb > 0:
            rec.coordinator = "phi3:mini"
            rec.generator = "phi3:mini"
            rec.mode = "GPU MINIMAL"
            rec.speed_estimate = "~4 tok/sec"
            rec.status = "⚠️"
            rec.status_label = "Ограниченно"
            rec.num_gpu = 999
            rec.warnings.append("Мало VRAM — используем phi3:mini (2.2GB)")

        # Только CPU
        elif ram_gb >= 16:
            rec.coordinator = "qwen2.5:3b"
            rec.generator = "qwen2.5:3b"
            rec.mode = "CPU ONLY"
            rec.speed_estimate = "~2 tok/sec"
            rec.status = "⚠️"
            rec.status_label = "Медленно"
            rec.num_gpu = 0
            rec.warnings.append("GPU не найден — работаем на CPU")
            rec.notes = "Ответы будут медленнее чем с GPU"

        else:
            rec.coordinator = "phi3:mini"
            rec.generator = "phi3:mini"
            rec.mode = "CPU MINIMAL"
            rec.speed_estimate = "~1 tok/sec"
            rec.status = "❌"
            rec.status_label = "Очень медленно"
            rec.num_gpu = 0
            rec.warnings.append("Мало RAM и нет GPU — система будет работать медленно")
            rec.warnings.append("Рекомендуется минимум 16GB RAM или GPU с 4GB VRAM")

        return rec

    def detect_and_recommend(self) -> dict:
        """Полный цикл: определение + рекомендации. Возвращает dict для API.

        ФОРМАТ HARDWARE:
          gpu       — str  | None  — имя GPU ("NVIDIA GeForce RTX 3060") или None
          gpu_info  — dict | None  — детали GPU (vram_gb, vendor, cuda...)
          vram_gb   — float        — VRAM в GB (0 если GPU не найден)
          ram_gb    — float        — RAM в GB
        """
        info = self.detect()
        rec = self.recommend(info)

        # gpu → строка с именем (или None), gpu_info → полный dict
        gpu_name: str | None = None
        vram_gb: float = 0.0
        gpu_info = None
        if info.gpu:
            gpu_name = info.gpu.name        # строка!
            vram_gb = info.gpu.vram_gb      # число!
            gpu_info = {
                "name": info.gpu.name,
                "vram_gb": info.gpu.vram_gb,
                "vendor": info.gpu.vendor,
                "cuda_available": info.gpu.cuda_available,
                "driver_version": info.gpu.driver_version,
            }

        return {
            "hardware": {
                "os": info.os,
                "os_version": info.os_version,
                "python_version": info.python_version,
                "cpu": info.cpu,
                "cpu_cores": info.cpu_cores,
                "ram_gb": info.ram_gb,      # число!
                "gpu": gpu_name,            # строка или None — НЕ объект!
                "gpu_info": gpu_info,        # полный dict для детальной инфо
                "vram_gb": vram_gb,          # число на верхнем уровне
                "ollama_available": info.ollama_available,
            },
            "recommendation": {
                "coordinator": rec.coordinator,
                "generator": rec.generator,
                "mode": rec.mode,
                "speed_estimate": rec.speed_estimate,
                "status": rec.status,
                "status_label": rec.status_label,
                "num_gpu": rec.num_gpu,
                "warnings": rec.warnings,
                "notes": rec.notes,
            },
            "errors": info.errors,
        }


# Singleton для переиспользования
_detector: Optional[HardwareDetector] = None


def get_detector() -> HardwareDetector:
    global _detector
    if _detector is None:
        _detector = HardwareDetector()
    return _detector


def detect_hardware() -> dict:
    """Быстрый вызов для API."""
    return get_detector().detect_and_recommend()
