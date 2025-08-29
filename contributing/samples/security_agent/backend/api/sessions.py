"""
Session Management API Endpoints

Provides RESTful API for managing conversation sessions.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Path as FastApiPath
from pydantic import BaseModel, Field
from pathlib import Path

from cache import cached
from services.session_manager import SessionManager, Session, Message

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])

from pathlib import Path
db_path = Path(__file__).parent.parent / "data" / "sessions.db"
db_path.parent.mkdir(parents=True, exist_ok=True)
session_manager = SessionManager(db_path=str(db_path), session_ttl_hours=24)


class CreateSessionRequest(BaseModel):
    """Request model for creating a session"""
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UpdateSessionRequest(BaseModel):
    """Request model for updating a session"""
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class AddMessageRequest(BaseModel):
    """Request model for adding a message"""
    role: str = Field(..., description="Message role (user or assistant)")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    """Response model for session data"""
    id: str
    user_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    context: Dict[str, Any]
    metadata: Dict[str, Any]


class MessageResponse(BaseModel):
    """Response model for message data"""
    id: str
    session_id: str
    role: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]


@router.post("/create", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    print("Creating session")
    """
    Create a new conversation session
    
    - **user_id**: Optional user identifier
    - **metadata**: Optional session metadata
    """
    try:
        session = session_manager.create_session(
            user_id=request.user_id,
            metadata=request.metadata
        )
        return SessionResponse(
            id=session.id,
            user_id=session.user_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            expires_at=session.expires_at,
            is_active=session.is_active,
            context=session.context,
            metadata=session.metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}", response_model=SessionResponse)
@cached
async def get_session(session_id: str = FastApiPath(..., description="Session ID")):
    """
    Get session details by ID
    
    - **session_id**: Session identifier
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    return SessionResponse(
        id=session.id,
        user_id=session.user_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        expires_at=session.expires_at,
        is_active=session.is_active,
        context=session.context,
        metadata=session.metadata
    )


@router.put("/{session_id}/update")
async def update_session(
    session_id: str,
    request: UpdateSessionRequest
):
    """
    Update session context and metadata
    
    - **session_id**: Session identifier
    - **context**: New context to merge
    - **metadata**: New metadata to merge
    """
    success = session_manager.update_session(
        session_id=session_id,
        context=request.context,
        metadata=request.metadata
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    return {"success": True, "session_id": session_id}


@router.post("/{session_id}/messages", response_model=MessageResponse)
async def add_message(
    session_id: str,
    request: AddMessageRequest
):
    """
    Add a message to a session
    
    - **session_id**: Session identifier
    - **role**: Message role (user or assistant)
    - **content**: Message content
    - **metadata**: Optional message metadata
    """
    try:
        message = session_manager.add_message(
            session_id=session_id,
            role=request.role,
            content=request.content,
            metadata=request.metadata
        )
        return MessageResponse(
            id=message.id,
            session_id=message.session_id,
            role=message.role,
            content=message.content,
            timestamp=message.timestamp,
            metadata=message.metadata
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/messages", response_model=List[MessageResponse])
async def get_conversation_history(
    session_id: str,
    limit: Optional[int] = Query(None, description="Maximum number of messages to return")
):
    """
    Get conversation history for a session
    
    - **session_id**: Session identifier
    - **limit**: Maximum number of messages to return
    """
    messages = session_manager.get_conversation_history(session_id, limit)
    
    return [
        MessageResponse(
            id=msg.id,
            session_id=msg.session_id,
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp,
            metadata=msg.metadata
        )
        for msg in messages
    ]


@router.get("/user/{user_id}", response_model=List[SessionResponse])
async def get_user_sessions(
    user_id: str,
    active_only: bool = Query(True, description="Return only active sessions")
):
    """
    Get all sessions for a user
    
    - **user_id**: User identifier
    - **active_only**: Whether to return only active sessions
    """
    sessions = session_manager.get_user_sessions(user_id, active_only)
    
    return [
        SessionResponse(
            id=session.id,
            user_id=session.user_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            expires_at=session.expires_at,
            is_active=session.is_active,
            context=session.context,
            metadata=session.metadata
        )
        for session in sessions
    ]


@router.delete("/{session_id}/expire")
async def expire_session(session_id: str):
    """
    Mark a session as expired/inactive
    
    - **session_id**: Session identifier
    """
    success = session_manager.expire_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"success": True, "session_id": session_id}


@router.post("/cleanup")
async def cleanup_expired_sessions():
    """
    Clean up expired sessions
    
    Marks expired sessions as inactive and optionally deletes old data
    """
    count = session_manager.cleanup_expired_sessions()
    return {
        "success": True,
        "expired_sessions": count,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/{session_id}/summary")
async def get_session_summary(session_id: str):
    """
    Get a summary of a session with statistics
    
    - **session_id**: Session identifier
    """
    summary = session_manager.get_session_summary(session_id)
    
    if "error" in summary:
        raise HTTPException(status_code=404, detail=summary["error"])
    
    return summary


@router.get("/test")
async def test_endpoint():
    return {"message": "hello"}