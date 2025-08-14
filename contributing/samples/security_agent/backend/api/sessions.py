"""
ADK Session Management API following thin client best practices.
Provides centralized session management for the ADK Security Agent.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Import the enhanced chat manager
try:
    import sys
    import os
    backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_path not in sys.path:
        sys.path.append(backend_path)
    from chat_manager import chat_manager
    CHAT_MANAGER_AVAILABLE = True
    logger.info("✅ Chat manager available for session management")
except ImportError as e:
    CHAT_MANAGER_AVAILABLE = False
    logger.warning(f"⚠️ Chat manager not available: {e}")

class SessionCreateRequest(BaseModel):
    """Request model for session creation."""
    user_id: str
    metadata: Optional[Dict[str, Any]] = None
    project_id: Optional[str] = None

class SessionResponse(BaseModel):
    """Response model for session operations."""
    success: bool
    session_id: str
    user_id: str
    status: Optional[str] = None
    created_at: Optional[str] = None
    last_activity: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MessageResponse(BaseModel):
    """Response model for session messages."""
    success: bool
    session_id: str
    messages: List[Dict[str, Any]]
    total_count: int
    page: Optional[int] = 1
    limit: Optional[int] = 50

@router.post("/create", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """
    Create a new ADK session following thin client best practices.
    This centralizes session management on the backend.
    """
    try:
        metadata = request.metadata or {}
        metadata.update({
            "source": "adk_thin_client",
            "project_id": request.project_id,
            "created_via": "sessions_api"
        })
        
        if CHAT_MANAGER_AVAILABLE:
            session_id = chat_manager.create_session(request.user_id, metadata)
            session = chat_manager.get_session(session_id)
            
            return SessionResponse(
                success=True,
                session_id=session_id,
                user_id=request.user_id,
                status=session.status.value if session else "active",
                created_at=session.created_at.isoformat() if session else datetime.now().isoformat(),
                metadata=metadata
            )
        else:
            # Fallback session creation
            session_id = f"{request.user_id}_{int(time.time())}"
            
            return SessionResponse(
                success=True,
                session_id=session_id,
                user_id=request.user_id,
                status="active",
                created_at=datetime.now().isoformat(),
                metadata=metadata
            )
            
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session details."""
    try:
        if CHAT_MANAGER_AVAILABLE:
            session = chat_manager.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            return SessionResponse(
                success=True,
                session_id=session_id,
                user_id=session.user_id,
                status=session.status.value,
                created_at=session.created_at.isoformat(),
                last_activity=session.last_activity.isoformat(),
                metadata=session.metadata
            )
        else:
            raise HTTPException(status_code=503, detail="Session management not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}/messages", response_model=MessageResponse)
async def get_session_messages(
    session_id: str,
    limit: int = Query(50, ge=1, le=500),
    page: int = Query(1, ge=1)
):
    """
    Get messages for a session with pagination.
    Following ADK best practices for thin client data retrieval.
    """
    try:
        if CHAT_MANAGER_AVAILABLE:
            # Get all messages
            all_messages = chat_manager.get_conversation_history(session_id)
            
            # Calculate pagination
            total_count = len(all_messages)
            start_idx = (page - 1) * limit
            end_idx = start_idx + limit
            
            # Get paginated messages
            messages = all_messages[start_idx:end_idx]
            
            # Convert to response format
            message_list = []
            for msg in messages:
                message_list.append({
                    "id": msg.id,
                    "content": msg.content,
                    "sender_type": msg.sender_type,
                    "timestamp": msg.timestamp.isoformat(),
                    "agent_used": msg.agent_used,
                    "delegation_path": msg.delegation_path,
                    "metadata": msg.metadata
                })
            
            return MessageResponse(
                success=True,
                session_id=session_id,
                messages=message_list,
                total_count=total_count,
                page=page,
                limit=limit
            )
        else:
            return MessageResponse(
                success=True,
                session_id=session_id,
                messages=[],
                total_count=0,
                page=page,
                limit=limit
            )
            
    except Exception as e:
        logger.error(f"Failed to get messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{session_id}/close")
async def close_session(session_id: str):
    """Close a session."""
    try:
        if CHAT_MANAGER_AVAILABLE:
            chat_manager.close_session(session_id)
            
        return {
            "success": True,
            "session_id": session_id,
            "status": "closed",
            "closed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to close session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}/analytics")
async def get_session_analytics(session_id: str):
    """Get analytics for a session."""
    try:
        if CHAT_MANAGER_AVAILABLE:
            analytics = chat_manager.get_session_analytics(session_id)
            return {
                "success": True,
                "session_id": session_id,
                "analytics": analytics
            }
        else:
            return {
                "success": True,
                "session_id": session_id,
                "analytics": {}
            }
            
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}/suggestions")
async def get_session_suggestions(session_id: str):
    """Get contextual suggestions for a session."""
    try:
        if CHAT_MANAGER_AVAILABLE:
            suggestions = chat_manager.get_contextual_suggestions(session_id)
            return {
                "success": True,
                "session_id": session_id,
                "suggestions": suggestions
            }
        else:
            return {
                "success": True,
                "session_id": session_id,
                "suggestions": []
            }
            
    except Exception as e:
        logger.error(f"Failed to get suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}/sessions")
async def get_user_sessions(user_id: str, active_only: bool = Query(False)):
    """Get all sessions for a user."""
    try:
        if CHAT_MANAGER_AVAILABLE:
            sessions = chat_manager.get_user_sessions(user_id, active_only)
            
            session_list = []
            for session in sessions:
                session_list.append({
                    "session_id": session.session_id,
                    "user_id": session.user_id,
                    "status": session.status.value,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "message_count": sum(len(conv) for conv in session.conversations.values())
                })
            
            return {
                "success": True,
                "user_id": user_id,
                "sessions": session_list,
                "total_count": len(session_list)
            }
        else:
            return {
                "success": True,
                "user_id": user_id,
                "sessions": [],
                "total_count": 0
            }
            
    except Exception as e:
        logger.error(f"Failed to get user sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{session_id}/restore")
async def restore_session(session_id: str):
    """
    Restore a session for continued use.
    Following ADK best practices for session persistence.
    """
    try:
        if CHAT_MANAGER_AVAILABLE:
            session = chat_manager.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Update session status and activity
            session.status = "active"
            session.last_activity = datetime.now()
            
            return {
                "success": True,
                "session_id": session_id,
                "status": "active",
                "restored_at": datetime.now().isoformat(),
                "message_count": sum(len(conv) for conv in session.conversations.values())
            }
        else:
            raise HTTPException(status_code=503, detail="Session management not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restore session: {e}")
        raise HTTPException(status_code=500, detail=str(e))