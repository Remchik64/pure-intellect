"""API route registry — connects all domain routers."""

from fastapi import APIRouter
from .chat import router as chat_router
from .session import router as session_router
from .memory_api import router as memory_router
from .models_api import router as models_router
from .system import router as system_router, openai_router

router = APIRouter()

# ── Include all domain routers ──────────────────────────────────────────────────
router.include_router(chat_router)
router.include_router(session_router)
router.include_router(memory_router)
router.include_router(models_router)
router.include_router(system_router)
