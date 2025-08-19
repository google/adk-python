# STORY-013: Session Management Service

**Epic**: SEC-001 - GCP Security Agent Platform  
**Story ID**: STORY-013  
**Title**: SQLite-based Session Management Service  
**Status**: Ready for Implementation  
**Priority**: P0 (Critical)  
**Size**: M (5 Story Points)  
**Sprint**: Current  

## User Story

**As a** System Administrator  
**I want to** have persistent session management using SQLite  
**So that** conversation history and context are maintained across interactions  

## Background

The GCP Security Agent needs to maintain conversation context and history across multiple interactions. Users should be able to resume previous conversations, reference earlier findings, and maintain context about their security posture over time. SQLite provides a lightweight, serverless solution perfect for POC and small-to-medium deployments.

## Acceptance Criteria

### Functional Requirements

1. **Database Schema**
   - [ ] Sessions table with unique session IDs
   - [ ] Messages table for conversation history
   - [ ] Context table for security findings cache
   - [ ] Users table for multi-user support
   - [ ] Metadata table for session configuration

2. **Session Service Operations**
   - [ ] Create new session with unique ID
   - [ ] Retrieve existing session by ID
   - [ ] Update session with new messages
   - [ ] Delete expired sessions
   - [ ] List all active sessions
   - [ ] Search sessions by user or date

3. **Conversation History**
   - [ ] Store all user queries
   - [ ] Store all agent responses
   - [ ] Maintain message ordering
   - [ ] Support message threading
   - [ ] Enable history export

4. **Context Management**
   - [ ] Cache security scan results
   - [ ] Store asset inventory snapshots
   - [ ] Maintain risk score history
   - [ ] Track remediation actions
   - [ ] Remember user preferences

5. **Session Lifecycle**
   - [ ] Auto-create session on first interaction
   - [ ] Session timeout after inactivity (configurable)
   - [ ] Session expiry after max age (30 days default)
   - [ ] Graceful session recovery
   - [ ] Clean shutdown handling

### Non-Functional Requirements

1. **Performance**
   - [ ] Sub-100ms query response time
   - [ ] Support 100+ concurrent sessions
   - [ ] Efficient pagination for history

2. **Data Management**
   - [ ] Automatic database migrations
   - [ ] Data retention policies
   - [ ] VACUUM operations for optimization
   - [ ] Backup and restore capability

3. **Security**
   - [ ] Session token validation
   - [ ] SQL injection prevention
   - [ ] Encrypted sensitive data
   - [ ] Access control per session

## Technical Design

### Database Schema

```sql
-- Sessions table
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    project_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSON
);

-- Messages table
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tool_calls JSON,
    metadata JSON,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Context cache table
CREATE TABLE context_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    context_type TEXT NOT NULL, -- 'assets', 'security_findings', 'risk_scores'
    data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Indexes for performance
CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_messages_session ON messages(session_id);
CREATE INDEX idx_context_session ON context_cache(session_id);
```

### Session Service Implementation

**File**: `backend/services/session_service.py`

```python
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

class SessionService:
    def __init__(self, db_path: str = "backend/data/chat_sessions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database with schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA_SQL)
    
    def create_session(self, user_id: str, project_id: str) -> str:
        """Create new session and return session ID"""
        session_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(days=30)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions (id, user_id, project_id, expires_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, user_id, project_id, expires_at))
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve session with messages and context"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get session
            session = conn.execute(
                "SELECT * FROM sessions WHERE id = ? AND is_active = TRUE",
                (session_id,)
            ).fetchone()
            
            if not session:
                return None
            
            # Get messages
            messages = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            ).fetchall()
            
            # Get context
            context = conn.execute(
                "SELECT * FROM context_cache WHERE session_id = ? AND expires_at > datetime('now')",
                (session_id,)
            ).fetchall()
            
            return {
                "session": dict(session),
                "messages": [dict(m) for m in messages],
                "context": [dict(c) for c in context]
            }
    
    def add_message(self, session_id: str, role: str, content: str, 
                   tool_calls: Optional[List] = None):
        """Add message to session history"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO messages (session_id, role, content, tool_calls)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, content, json.dumps(tool_calls) if tool_calls else None))
            
            # Update session timestamp
            conn.execute(
                "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?",
                (session_id,)
            )
    
    def cache_context(self, session_id: str, context_type: str, 
                     data: Dict, ttl_hours: int = 1):
        """Cache context data for session"""
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO context_cache (session_id, context_type, data, expires_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, context_type, json.dumps(data), expires_at))
    
    def cleanup_expired(self):
        """Clean up expired sessions and context"""
        with sqlite3.connect(self.db_path) as conn:
            # Mark expired sessions as inactive
            conn.execute("""
                UPDATE sessions 
                SET is_active = FALSE 
                WHERE expires_at < datetime('now') AND is_active = TRUE
            """)
            
            # Delete old context cache
            conn.execute("""
                DELETE FROM context_cache 
                WHERE expires_at < datetime('now')
            """)
            
            # VACUUM to optimize database
            conn.execute("VACUUM")
```

### API Integration

**File**: `backend/api/session.py`

```python
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from ..services.session_service import SessionService

router = APIRouter()
session_service = SessionService()

class SessionCreateRequest(BaseModel):
    user_id: str
    project_id: str

class MessageRequest(BaseModel):
    session_id: str
    role: str
    content: str
    tool_calls: Optional[List] = None

@router.post("/create")
async def create_session(request: SessionCreateRequest):
    """Create new chat session"""
    session_id = session_service.create_session(
        request.user_id, 
        request.project_id
    )
    return {"session_id": session_id, "status": "created"}

@router.get("/get/{session_id}")
async def get_session(session_id: str):
    """Get session with history and context"""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.post("/message")
async def add_message(request: MessageRequest):
    """Add message to session"""
    session_service.add_message(
        request.session_id,
        request.role,
        request.content,
        request.tool_calls
    )
    return {"status": "message_added"}

@router.post("/cleanup")
async def cleanup_expired():
    """Clean up expired sessions"""
    session_service.cleanup_expired()
    return {"status": "cleanup_completed"}
```

### Frontend Integration

**File**: `frontend/session_manager.py`

```python
import streamlit as st
import httpx
from typing import Optional

class SessionManager:
    def __init__(self, backend_url: str):
        self.backend_url = backend_url
    
    def get_or_create_session(self) -> str:
        """Get existing or create new session"""
        if 'session_id' not in st.session_state:
            # Create new session
            response = httpx.post(
                f"{self.backend_url}/api/v1/session/create",
                json={
                    "user_id": st.session_state.get('user_id', 'anonymous'),
                    "project_id": st.session_state.get('project_id', 'default')
                }
            )
            st.session_state.session_id = response.json()['session_id']
        
        return st.session_state.session_id
    
    def load_history(self, session_id: str):
        """Load conversation history"""
        response = httpx.get(
            f"{self.backend_url}/api/v1/session/get/{session_id}"
        )
        return response.json()
    
    def save_message(self, role: str, content: str, tool_calls=None):
        """Save message to session"""
        httpx.post(
            f"{self.backend_url}/api/v1/session/message",
            json={
                "session_id": st.session_state.session_id,
                "role": role,
                "content": content,
                "tool_calls": tool_calls
            }
        )
```

## Testing Strategy

### Unit Tests
- Test database operations (CRUD)
- Test session expiry logic
- Test context caching
- Test data serialization

### Integration Tests
- Test API endpoints
- Test session lifecycle
- Test concurrent sessions
- Test database performance

### Example Test
```python
def test_session_lifecycle():
    service = SessionService(":memory:")
    
    # Create session
    session_id = service.create_session("user1", "project1")
    assert session_id is not None
    
    # Add messages
    service.add_message(session_id, "user", "Check my security")
    service.add_message(session_id, "assistant", "Scanning resources...")
    
    # Retrieve session
    session = service.get_session(session_id)
    assert len(session['messages']) == 2
    assert session['session']['user_id'] == "user1"
```

## Definition of Done

- [ ] Database schema implemented
- [ ] Session service with all CRUD operations
- [ ] API endpoints integrated
- [ ] Frontend session management
- [ ] Message history persistence
- [ ] Context caching working
- [ ] Session expiry and cleanup
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Documentation updated

## Dependencies

- SQLite3 (included in Python)
- Database file location configured
- Backend API running
- Frontend integrated

## Notes

- SQLite is perfect for POC and small deployments
- For production, consider PostgreSQL or Firestore
- Session IDs should be cryptographically secure
- Consider implementing session encryption for sensitive data
- Add session analytics for usage tracking