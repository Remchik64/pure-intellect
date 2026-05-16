"""Model management endpoints."""

import asyncio
import logging
import httpx
from fastapi import APIRouter, HTTPException
from ..api.state import get_pipeline, download_progress

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/models/status")
async def models_status():
    """Полный статус всех моделей: скачанные + активные в VRAM.

    Возвращает:
    - downloaded: все скачанные модели (из /api/tags)
    - active_in_vram: модели загруженные в VRAM прямо сейчас (из /api/ps)
    - coordinator/generator: статус назначенных моделей

    Статусы:
    - 'active'  — загружена в VRAM прямо сейчас 🔥
    - 'ready'   — скачана, готова к запуску ✅
    - 'offline' — не скачана, нужно скачать ❌
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Все скачанные модели
            tags_resp = await client.get("http://host.docker.internal:11434/api/tags")
            tags_resp.raise_for_status()
            downloaded = [m["name"] for m in tags_resp.json().get("models", [])]

            # Активные в VRAM прямо сейчас
            try:
                ps_resp = await client.get("http://host.docker.internal:11434/api/ps")
                active = [m["name"] for m in ps_resp.json().get("models", [])]
            except Exception:
                active = []

            # Назначенные модели из пайплайна
            try:
                pipeline = get_pipeline()
                coordinator = pipeline._router.coordinator_model
                generator = pipeline._router.generator_model
            except Exception:
                coordinator = "qwen3.5:2b"
                generator = "qwen3.5:9b"

            def get_status(model_name: str) -> str:
                if model_name in active:
                    return "active"   # загружена в VRAM
                elif model_name in downloaded:
                    return "ready"    # скачана, готова
                else:
                    return "offline"  # не скачана

            return {
                "downloaded": downloaded,
                "active_in_vram": active,
                "coordinator": {
                    "model": coordinator,
                    "status": get_status(coordinator),
                },
                "generator": {
                    "model": generator,
                    "status": get_status(generator),
                },
            }
    except Exception as e:
        logger.warning(f"Models status failed: {e}")
        return {"error": str(e), "downloaded": [], "active_in_vram": [],
                "coordinator": {"model": "—", "status": "offline"},
                "generator": {"model": "—", "status": "offline"}}


@router.post("/models/switch")
async def switch_model(req: dict):
    """Переключить модель координатора или генератора без перезапуска.

    Body: {"role": "coordinator" | "generator", "model": "qwen3.5:9b"}
    """
    try:
        pipeline = get_pipeline()
        role = req.get("role")
        model = req.get("model")
        if role not in ("coordinator", "generator"):
            raise HTTPException(status_code=400, detail="role must be coordinator or generator")
        if not model:
            raise HTTPException(status_code=400, detail="model is required")

        router_obj = pipeline._router
        if role == "coordinator":
            router_obj.coordinator_model = model
            logger.info(f"[admin] Coordinator switched to {model}")
        else:
            router_obj.generator_model = model
            logger.info(f"[admin] Generator switched to {model}")

        # Сохраняем в config.yaml — чтобы выбор сохранялся после перезапуска
        try:
            from contextor.engines.config_loader import save_model_to_config
            saved = save_model_to_config(role, model)
            save_status = "saved to config.yaml" if saved else "runtime only (config save failed)"
            logger.info(f"[admin] Config {'saved' if saved else 'not saved'}: {role}={model}")
        except Exception as e:
            save_status = f"runtime only ({e})"
            logger.warning(f"[admin] Could not save config: {e}")

        return {"status": "switched", "role": role, "model": model, "persistence": save_status}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Switch model failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_name:path}")
async def delete_model(model_name: str):
    """Удалить модель из Ollama полностью (освобождает место на диске)."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(
                method="DELETE",
                url="http://host.docker.internal:11434/api/delete",
                json={"name": model_name},
            )
            if resp.status_code == 200:
                logger.info(f"[admin] Model deleted: {model_name}")
                return {"status": "deleted", "model": model_name}
            else:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Ollama error: {resp.text[:200]}"
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete model failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/download")
async def download_model(req: dict):
    """Скачать модель через Ollama со streaming прогрессом.

    Body: {"model": "qwen3.5:2b"}
    Прогресс доступен через GET /models/download/check/{model}
    """
    import json as _json
    import time as _time

    model = req.get("model", "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    # Если уже качается — не запускаем повторно
    if download_progress.get(model, {}).get("status") == "downloading":
        return {"status": "already_downloading", "model": model}

    download_progress[model] = {
        "status": "starting",
        "percent": 0,
        "speed": "",
        "error": None,
        "started_at": _time.time(),
    }

    async def _pull_with_progress():
        """Streaming pull через Ollama HTTP API с парсингом прогресса."""
        download_progress[model]["status"] = "downloading"
        last_speed_check = _time.time()
        last_completed = 0

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    "http://host.docker.internal:11434/api/pull",
                    json={"name": model, "stream": True},
                    timeout=None,
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue

                        status_msg = data.get("status", "")
                        completed = data.get("completed", 0)
                        total = data.get("total", 0)

                        # Вычисляем процент
                        percent = 0
                        if total and total > 0:
                            percent = int(completed / total * 100)

                        # Вычисляем скорость (bytes/sec)
                        now = _time.time()
                        elapsed = now - last_speed_check
                        speed_str = ""
                        if elapsed >= 1.0 and completed > last_completed:
                            bytes_per_sec = (completed - last_completed) / elapsed
                            if bytes_per_sec >= 1_048_576:
                                speed_str = f"{bytes_per_sec / 1_048_576:.1f} MB/s"
                            elif bytes_per_sec >= 1024:
                                speed_str = f"{bytes_per_sec / 1024:.1f} KB/s"
                            else:
                                speed_str = f"{bytes_per_sec:.0f} B/s"
                            last_speed_check = now
                            last_completed = completed

                        download_progress[model].update({
                            "status": "downloading",
                            "percent": percent,
                            "speed": speed_str,
                            "status_msg": status_msg,
                            "completed": completed,
                            "total": total,
                            "error": None,
                        })

                        # Ollama сигнализирует об успехе через status == "success"
                        if status_msg == "success":
                            break

            download_progress[model]["status"] = "done"
            download_progress[model]["percent"] = 100
            download_progress[model]["speed"] = ""
            logger.info(f"[model_download] {model} downloaded successfully")

        except Exception as e:
            download_progress[model]["status"] = "error"
            download_progress[model]["error"] = str(e)
            logger.error(f"[model_download] {model} failed: {e}")

    asyncio.create_task(_pull_with_progress())
    return {"status": "downloading", "model": model, "message": f"Скачивание {model} запущено"}


@router.get("/models/download/check/{model_name:path}")
async def check_model_downloaded(model_name: str):
    """Прогресс скачивания и статус модели в Ollama.

    Возвращает реальный прогресс из download_progress,
    а также проверяет наличие модели в Ollama tags.
    """
    # Сначала отдаём прогресс если есть активное скачивание
    progress = download_progress.get(model_name)

    # Проверяем готовность через Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://host.docker.internal:11434/api/tags")
            data = resp.json()
            available = [m["name"] for m in data.get("models", [])]
            is_ready = any(
                model_name == m or m.startswith(model_name)
                for m in available
            )
    except Exception:
        available = []
        is_ready = False

    if progress is not None:
        return {
            "model": model_name,
            "ready": is_ready or progress.get("status") == "done",
            "status": progress.get("status", "unknown"),
            "percent": progress.get("percent", 0),
            "speed": progress.get("speed", ""),
            "status_msg": progress.get("status_msg", ""),
            "error": progress.get("error"),
            "available_models": available,
        }

    # Нет записи в прогрессе — просто проверяем наличие
    return {
        "model": model_name,
        "ready": is_ready,
        "status": "ready" if is_ready else "not_downloaded",
        "percent": 100 if is_ready else 0,
        "speed": "",
        "status_msg": "",
        "error": None,
        "available_models": available,
    }


@router.get("/dual-model/stats")
async def dual_model_stats():
    """Статистика Dual Model Router (P6): coordinator vs generator."""
    try:
        pipeline = get_pipeline()
        return pipeline.dual_model_stats()
    except Exception as e:
        logger.error(f"Dual model stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ollama/models")
async def ollama_models_proxy():
    """Прокси к Ollama API — список доступных моделей.

    Решает CORS проблему браузера при обращении к localhost:11434.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://host.docker.internal:11434/api/tags")
            return resp.json()
    except Exception as e:
        logger.warning(f"Ollama not available: {e}")
        return {"models": [], "error": str(e)}
