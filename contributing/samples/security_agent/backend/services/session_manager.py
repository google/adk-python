"""
Session Management Service for ADK Security Agent

Provides persistent session storage using SQLite for conversation history,
context retention, and multi-user support.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Message(BaseModel):
    """Individual message in a conversation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


class Session(BaseModel):
    """Session model with conversation history"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    is_active: bool = True
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionManager:
    """Manages conversation sessions with SQLite persistence"""
    
    def __init__(self, db_path: str = "sessions.db", session_ttl_hours: int = 24):
        """
        Initialize session manager
        
        Args:
            db_path: Path to SQLite database file
            session_ttl_hours: Session time-to-live in hours
        """
        self.db_path = Path(db_path)
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self._init_database()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    context TEXT,
                    metadata TEXT
                )
            """)
            
            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_user_id 
                ON sessions(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_expires_at 
                ON sessions(expires_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_id 
                ON messages(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                ON messages(timestamp)
            """)
    
    def create_session(self, user_id: Optional[str] = None, 
                      metadata: Optional[Dict[str, Any]] = None) -> Session:
        """
        Create a new session
        
        Args:
            user_id: Optional user identifier
            metadata: Optional session metadata
            
        Returns:
            Created session
        """
        session = Session(
            user_id=user_id,
            expires_at=datetime.utcnow() + self.session_ttl,
            metadata=metadata or {}
        )
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions 
                (id, user_id, created_at, updated_at, expires_at, is_active, context, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.id,
                session.user_id,
                session.created_at,
                session.updated_at,
                session.expires_at,
                session.is_active,
                json.dumps(session.context),
                json.dumps(session.metadata)
            ))
        
        logger.info(f"Created session {session.id} for user {user_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Retrieve a session by ID
        
        Args:
            session_id: Session ID
            
        Returns:
            Session if found and active, None otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sessions 
                WHERE id = ? AND is_active = 1
            """, (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Check if session is expired
            expires_at = datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None
            if expires_at and expires_at < datetime.utcnow():
                self.expire_session(session_id)
                return None
            
            return Session(
                id=row['id'],
                user_id=row['user_id'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                expires_at=expires_at,
                is_active=bool(row['is_active']),
                context=json.loads(row['context'] or '{}'),
                metadata=json.loads(row['metadata'] or '{}')
            )
    
    def update_session(self, session_id: str, 
                      context: Optional[Dict[str, Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update session context and metadata
        
        Args:
            session_id: Session ID
            context: New context to merge
            metadata: New metadata to merge
            
        Returns:
            True if updated successfully
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        if context:
            session.context.update(context)
        if metadata:
            session.metadata.update(metadata)
        
        session.updated_at = datetime.utcnow()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions 
                SET context = ?, metadata = ?, updated_at = ?
                WHERE id = ?
            """, (
                json.dumps(session.context),
                json.dumps(session.metadata),
                session.updated_at,
                session_id
            ))
        
        return True
    
    def add_message(self, session_id: str, role: str, content: str,
                   metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add a message to a session
        
        Args:
            session_id: Session ID
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            Created message
        """
        # Verify session exists and is active
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found or expired")
        
        message = Message(
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages 
                (id, session_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                message.id,
                message.session_id,
                message.role,
                message.content,
                message.timestamp,
                json.dumps(message.metadata)
            ))
            
            # Update session's updated_at timestamp
            cursor.execute("""
                UPDATE sessions 
                SET updated_at = ? 
                WHERE id = ?
            """, (datetime.utcnow(), session_id))
        
        return message
    
    def get_conversation_history(self, session_id: str, 
                                limit: Optional[int] = None) -> List[Message]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session ID
            limit: Maximum number of messages to return
            
        Returns:
            List of messages in chronological order
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM messages 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (session_id,))
            rows = cursor.fetchall()
            
            messages = []
            for row in rows:
                messages.append(Message(
                    id=row['id'],
                    session_id=row['session_id'],
                    role=row['role'],
                    content=row['content'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    metadata=json.loads(row['metadata'] or '{}')
                ))
            
            return messages
    
    def get_user_sessions(self, user_id: str, 
                         active_only: bool = True) -> List[Session]:
        """
        Get all sessions for a user
        
        Args:
            user_id: User ID
            active_only: Whether to return only active sessions
            
        Returns:
            List of sessions
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM sessions WHERE user_id = ?"
            if active_only:
                query += " AND is_active = 1"
            query += " ORDER BY updated_at DESC"
            
            cursor.execute(query, (user_id,))
            rows = cursor.fetchall()
            
            sessions = []
            for row in rows:
                expires_at = datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None
                
                # Skip expired sessions if active_only
                if active_only and expires_at and expires_at < datetime.utcnow():
                    continue
                
                sessions.append(Session(
                    id=row['id'],
                    user_id=row['user_id'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    expires_at=expires_at,
                    is_active=bool(row['is_active']),
                    context=json.loads(row['context'] or '{}'),
                    metadata=json.loads(row['metadata'] or '{}')
                ))
            
            return sessions
    
    def expire_session(self, session_id: str) -> bool:
        """
        Mark a session as expired/inactive
        
        Args:
            session_id: Session ID
            
        Returns:
            True if expired successfully
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions 
                SET is_active = 0, updated_at = ?
                WHERE id = ?
            """, (datetime.utcnow(), session_id))
            
            return cursor.rowcount > 0
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Mark expired sessions as inactive
            cursor.execute("""
                UPDATE sessions 
                SET is_active = 0 
                WHERE expires_at < ? AND is_active = 1
            """, (datetime.utcnow(),))
            
            expired_count = cursor.rowcount
            
            # Optionally delete old inactive sessions (older than 30 days)
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            cursor.execute("""
                DELETE FROM messages 
                WHERE session_id IN (
                    SELECT id FROM sessions 
                    WHERE is_active = 0 AND updated_at < ?
                )
            """, (cutoff_date,))
            
            cursor.execute("""
                DELETE FROM sessions 
                WHERE is_active = 0 AND updated_at < ?
            """, (cutoff_date,))
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired sessions")
            
            return expired_count
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of a session
        
        Args:
            session_id: Session ID
            
        Returns:
            Session summary with statistics
        """
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found or expired"}
        
        messages = self.get_conversation_history(session_id)
        
        return {
            "session_id": session.id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "expires_at": session.expires_at.isoformat() if session.expires_at else None,
            "is_active": session.is_active,
            "message_count": len(messages),
            "user_messages": len([m for m in messages if m.role == 'user']),
            "assistant_messages": len([m for m in messages if m.role == 'assistant']),
            "context": session.context,
            "metadata": session.metadata
        }
    
    def search_messages(self, query: str, session_id: Optional[str] = None,
                       user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search messages across sessions
        
        Args:
            query: Search query
            session_id: Optional session ID to limit search
            user_id: Optional user ID to limit search
            
        Returns:
            List of matching messages with session info
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query_parts = ["SELECT m.*, s.user_id FROM messages m JOIN sessions s ON m.session_id = s.id"]
            conditions = ["m.content LIKE ?"]
            params = [f"%{query}%"]
            
            if session_id:
                conditions.append("m.session_id = ?")
                params.append(session_id)
            
            if user_id:
                conditions.append("s.user_id = ?")
                params.append(user_id)
            
            query_sql = f"{' '.join(query_parts)} WHERE {' AND '.join(conditions)} ORDER BY m.timestamp DESC"
            cursor.execute(query_sql, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "message_id": row['id'],
                    "session_id": row['session_id'],
                    "user_id": row['user_id'],
                    "role": row['role'],
                    "content": row['content'],
                    "timestamp": row['timestamp'],
                    "metadata": json.loads(row['metadata'] or '{}')
                })
            
            return results