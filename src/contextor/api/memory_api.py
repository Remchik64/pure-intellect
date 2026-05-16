"""Memory endpoints."""

import logging
from fastapi import APIRouter, HTTPException
from ..api.state import get_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/memory/stats")
async def memory_stats():
    """Статистика самообновляемой памяти."""
    try:
        pipeline = get_pipeline()
        return pipeline.memory_stats()
    except Exception as e:
        logger.error(f"Memory stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/facts")
async def memory_facts():
    """Получить все факты из памяти (working + storage)."""
    try:
        pipeline = get_pipeline()
        def _fact_to_dict(f):
            if isinstance(f, dict):
                return f
            if hasattr(f, '__dict__'):
                return f.__dict__
            return {"content": str(f)}

        wm_facts = [_fact_to_dict(f) for f in pipeline.working_memory._facts]
        storage_facts = [_fact_to_dict(f) for f in pipeline.memory_storage._facts]
        return {"facts": wm_facts + storage_facts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/clear")
async def memory_clear():
    """Очистить рабочую память (переместить все факты в долгосрочное хранилище)."""
    try:
        pipeline = get_pipeline()
        pipeline.memory_clear()
        return {"status": "cleared", "message": "Working memory cleared, facts moved to storage"}
    except Exception as e:
        logger.error(f"Memory clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
