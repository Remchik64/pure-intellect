"""Session CRUD endpoints — multi-session management for MVP v0.3."""

import logging
from fastapi import APIRouter, HTTPException
from ..api.schemas import CreateSessionRequest, RenameSessionRequest
from ..api.state import get_session_manager, get_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/sessions")
async def create_session(request: CreateSessionRequest):
    """Создать новый чат.

    Body: {"display_name": "My Chat" (optional), "session_type": "chat" (optional)}
    """
    try:
        mgr = get_session_manager()
        info = mgr.create_session(
            display_name=request.display_name,
            session_type=request.session_type or "chat",
        )
        return {"status": "created", **info.to_dict()}
    except Exception as e:
        logger.error(f"Create session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions():
    """Список всех чатов."""
    try:
        mgr = get_session_manager()
        sessions = mgr.list_sessions()
        return {
            "active_session_id": mgr.active_session_id,
            "sessions": [s.to_dict() for s in sessions],
            "total": len(sessions),
        }
    except Exception as e:
        logger.error(f"List sessions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Информация о конкретном чате."""
    try:
        mgr = get_session_manager()
        info = mgr.get_session(session_id)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        return info.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Удалить чат.

    Нельзя удалить активный чат или default.
    """
    try:
        mgr = get_session_manager()
        # Проверяем существование
        if not mgr.session_exists(session_id):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        deleted = mgr.delete_session(session_id)
        if not deleted:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete session '{session_id}' (active or default)",
            )
        return {"status": "deleted", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/sessions/{session_id}")
async def rename_session(session_id: str, request: RenameSessionRequest):
    """Переименовать чат."""
    try:
        mgr = get_session_manager()
        if not mgr.session_exists(session_id):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        renamed = mgr.rename_session(session_id, request.display_name)
        if not renamed:
            raise HTTPException(status_code=500, detail="Rename failed")

        info = mgr.get_session(session_id)
        return {"status": "renamed", **info.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rename session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/switch")
async def switch_session(session_id: str):
    """Переключить активный чат.

    Обновляет SessionManager И переключает пайплайн на новую сессию.
    Возвращает chat_history для обновления UI.
    """
    try:
        mgr = get_session_manager()
        info = mgr.switch_to(session_id)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        # Switch pipeline to new session (saves current, loads new)
        pipeline = get_pipeline()
        switch_result = pipeline.switch_session(session_id)

        return {
            "status": "switched",
            **info.to_dict(),
            "chat_history": switch_result.get("chat_history", []),
            "turn": switch_result.get("turn", 0),
            "loaded": switch_result.get("loaded", False),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Switch session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Получить историю чата для указанной сессии.

    Позволяет загрузить историю при переключении без полного pipeline reload.
    """
    try:
        mgr = get_session_manager()
        if not mgr.session_exists(session_id):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        # Load chat_history directly from session directory
        from pathlib import Path
        history_path = Path("storage/sessions") / session_id / "chat_history.json"
        if not history_path.exists():
            return {"session_id": session_id, "chat_history": [], "count": 0}

        import json
        try:
            data = json.loads(history_path.read_text(encoding="utf-8", errors="replace"))
            messages = data.get("messages", [])
            return {
                "session_id": session_id,
                "chat_history": messages,
                "count": len(messages),
            }
        except Exception as e:
            logger.error(f"Failed to read chat history for session {session_id}: {e}")
            return {"session_id": session_id, "chat_history": [], "count": 0}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Legacy endpoints (backward compatible) ──────────────────────────────────────

@router.get("/session/info")
async def session_info():
    """Информация о текущей сохранённой сессии (legacy)."""
    try:
        pipeline = get_pipeline()
        return pipeline.session_info()
    except Exception as e:
        logger.error(f"Session info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session")
async def session_delete():
    """Удалить сохранённую сессию и сбросить состояние (legacy)."""
    try:
        pipeline = get_pipeline()
        pipeline.session_delete()
        return {"status": "deleted", "message": "Session deleted and state reset"}
    except Exception as e:
        logger.error(f"Session delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
