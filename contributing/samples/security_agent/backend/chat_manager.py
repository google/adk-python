"""Enhanced Chat Manager for real-time multi-session conversation management with persistence."""

import asyncio
import json
import logging
import time
import sqlite3
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class MessageType(Enum):
    CHAT = "chat"
    SYSTEM = "system"
    DELEGATION = "delegation"
    STATUS_UPDATE = "status_update"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR = "error"
    TYPING_INDICATOR = "typing_indicator"

class SessionStatus(Enum):
    ACTIVE = "active"
    IDLE = "idle"
    CLOSED = "closed"
    ARCHIVED = "archived"

@dataclass
class ChatMessage:
    """Represents a chat message with metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    sender_type: str = "user"  # user, assistant, system
    timestamp: datetime = field(default_factory=datetime.now)
    message_type: MessageType = MessageType.CHAT
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_used: Optional[str] = None
    delegation_path: List[str] = field(default_factory=list)
    context_references: List[str] = field(default_factory=list)
    performance_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationTopic:
    """Represents a conversation topic with context."""
    name: str
    confidence: float
    keywords: List[str] = field(default_factory=list)
    context_data: Dict[str, Any] = field(default_factory=dict)
    first_mentioned: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ChatSession:
    """Represents a complete chat session."""
    session_id: str
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    status: SessionStatus = SessionStatus.ACTIVE
    conversations: Dict[str, List[ChatMessage]] = field(default_factory=dict)
    topics: List[ConversationTopic] = field(default_factory=list)
    user_context: Dict[str, Any] = field(default_factory=dict)
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConversationAnalyzer:
    """Analyzes conversations for topic detection and context understanding."""
    
    def __init__(self):
        self.topic_keywords = {
            "security": ["security", "vulnerability", "threat", "risk", "breach", "attack", "malware"],
            "iam": ["iam", "permissions", "roles", "access", "identity", "authentication", "authorization"],
            "storage": ["storage", "bucket", "gcs", "cloud storage", "data", "backup"],
            "compliance": ["compliance", "audit", "regulation", "policy", "framework", "standard"],
            "monitoring": ["monitoring", "metrics", "performance", "alerts", "dashboard", "logs"],
            "incidents": ["incident", "response", "investigation", "forensics", "containment"]
        }
    
    def detect_topics(self, messages: List[ChatMessage]) -> List[ConversationTopic]:
        """Detect conversation topics from messages."""
        topics = {}
        
        for message in messages:
            content_lower = message.content.lower()
            
            for topic_name, keywords in self.topic_keywords.items():
                matches = [kw for kw in keywords if kw in content_lower]
                
                if matches:
                    if topic_name not in topics:
                        topics[topic_name] = ConversationTopic(
                            name=topic_name,
                            confidence=0.0,
                            keywords=matches,
                            first_mentioned=message.timestamp
                        )
                    
                    # Update confidence and keywords
                    topics[topic_name].confidence += len(matches) * 0.1
                    topics[topic_name].keywords.extend(matches)
                    topics[topic_name].keywords = list(set(topics[topic_name].keywords))
                    topics[topic_name].last_updated = message.timestamp
        
        # Normalize confidence scores
        for topic in topics.values():
            topic.confidence = min(1.0, topic.confidence)
        
        return list(topics.values())
    
    def generate_context_suggestions(self, topics: List[ConversationTopic]) -> List[str]:
        """Generate contextual suggestions based on detected topics."""
        suggestions = []
        
        topic_suggestions = {
            "security": [
                "Show me detailed security findings",
                "How can I improve my security posture?",
                "What are the current security threats?"
            ],
            "iam": [
                "Analyze IAM permissions",
                "Review user access rights", 
                "Show overprivileged accounts"
            ],
            "storage": [
                "Check storage security",
                "Optimize storage costs",
                "Review bucket permissions"
            ],
            "compliance": [
                "Check compliance status",
                "Generate compliance report",
                "Fix compliance gaps"
            ],
            "monitoring": [
                "Show performance metrics",
                "Set up monitoring alerts",
                "Analyze system health"
            ],
            "incidents": [
                "Show active incidents",
                "Create incident response plan",
                "Investigate security events"
            ]
        }
        
        for topic in sorted(topics, key=lambda t: t.confidence, reverse=True)[:3]:
            if topic.name in topic_suggestions:
                suggestions.extend(topic_suggestions[topic.name][:2])
        
        return suggestions[:5]  # Return top 5 suggestions

class EnhancedChatManager:
    """Enhanced chat manager with real-time capabilities, multi-session support, and persistence."""
    
    def __init__(self, db_path: str = None):
        self.sessions: Dict[str, ChatSession] = {}
        self.active_connections: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
        self.conversation_analyzer = ConversationAnalyzer()
        self.performance_tracker = PerformanceTracker()
        
        # Database setup for persistence
        if db_path is None:
            db_dir = Path(__file__).parent / "data"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "chat_sessions.db")
        
        self.db_path = db_path
        self._init_database()
        self._load_sessions()
        
        # Background task for cleanup
        self._cleanup_task = None
        
    async def start(self):
        """Start the chat manager and background tasks."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_old_sessions())
    
    async def stop(self):
        """Stop the chat manager and cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        self._save_all_sessions()
    
    def create_session(self, user_id: str, metadata: Dict[str, Any] = None, session_id: str = None) -> str:
        """Create a new chat session."""
        if not session_id:
            session_id = f"{user_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        self.sessions[session_id] = session
        self._save_session(session_id, session)
        logger.info(f"Created new session {session_id} for user {user_id}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)
    
    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        if session_id in self.sessions:
            return True
        # Check database as well
        self._load_single_session(session_id)
        return session_id in self.sessions
    
    async def session_exists_async(self, session_id: str) -> bool:
        """Async version of session_exists for compatibility."""
        return self.session_exists(session_id)
    
    def get_user_sessions(self, user_id: str, active_only: bool = False) -> List[ChatSession]:
        """Get all sessions for a user."""
        sessions = [
            session for session in self.sessions.values()
            if session.user_id == user_id
        ]
        
        if active_only:
            sessions = [s for s in sessions if s.status == SessionStatus.ACTIVE]
        
        return sorted(sessions, key=lambda s: s.last_activity, reverse=True)
    
    async def create_session_async(self, user_id: str, metadata: Dict[str, Any] = None, session_id: str = None) -> str:
        """Async version of create_session for compatibility."""
        return self.create_session(user_id, metadata, session_id)
    
    async def add_message(
        self, 
        session_id: str, 
        content: str, 
        sender_type: str = "user",
        agent_used: str = None,
        delegation_path: List[str] = None,
        performance_data: Dict[str, Any] = None
    ) -> ChatMessage:
        """Add a message to a session."""
        session = self.get_session(session_id)
        if not session:
            # Try to reload session from database if not in memory
            self._load_single_session(session_id)
            session = self.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
        
        # Create conversation if it doesn't exist
        conversation_id = "main"
        if conversation_id not in session.conversations:
            session.conversations[conversation_id] = []
        
        # Create message
        message = ChatMessage(
            content=content,
            sender_type=sender_type,
            agent_used=agent_used,
            delegation_path=delegation_path or [],
            performance_data=performance_data or {}
        )
        
        # Add to conversation
        session.conversations[conversation_id].append(message)
        session.last_activity = datetime.now()
        
        # Update topics
        all_messages = session.conversations[conversation_id]
        session.topics = self.conversation_analyzer.detect_topics(all_messages)
        
        # Track performance
        if performance_data:
            self.performance_tracker.record_metrics(session_id, performance_data)
        
        # Save session to database
        self._save_session(session_id, session)
        
        logger.info(f"Added message to session {session_id}")
        return message
    
    def get_conversation_history(
        self, 
        session_id: str, 
        conversation_id: str = "main",
        limit: int = None
    ) -> List[ChatMessage]:
        """Get conversation history."""
        session = self.get_session(session_id)
        if not session:
            return []
        
        messages = session.conversations.get(conversation_id, [])
        
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_contextual_suggestions(self, session_id: str) -> List[str]:
        """Get contextual suggestions for a session."""
        session = self.get_session(session_id)
        if not session:
            return []
        
        return self.conversation_analyzer.generate_context_suggestions(session.topics)
    
    async def update_user_context(self, session_id: str, context_updates: Dict[str, Any]):
        """Update user context for a session."""
        session = self.get_session(session_id)
        if session:
            session.user_context.update(context_updates)
            session.last_activity = datetime.now()
            self._save_session(session_id, session)
    
    def get_user_context(self, session_id: str) -> Dict[str, Any]:
        """Get user context for a session."""
        session = self.get_session(session_id)
        return session.user_context if session else {}
    
    def close_session(self, session_id: str):
        """Close a session."""
        session = self.get_session(session_id)
        if session:
            session.status = SessionStatus.CLOSED
            session.metadata["closed_at"] = datetime.now().isoformat()
            self._save_session(session_id, session)
            logger.info(f"Closed session {session_id}")
    
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for a session."""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        total_messages = sum(len(conv) for conv in session.conversations.values())
        user_messages = sum(
            len([m for m in conv if m.sender_type == "user"])
            for conv in session.conversations.values()
        )
        
        return {
            "session_id": session_id,
            "total_messages": total_messages,
            "user_messages": user_messages,
            "assistant_messages": total_messages - user_messages,
            "conversation_count": len(session.conversations),
            "topics": [{"name": t.name, "confidence": t.confidence} for t in session.topics],
            "duration_minutes": (session.last_activity - session.created_at).total_seconds() / 60,
            "status": session.status.value,
            "performance_metrics": self.performance_tracker.get_session_summary(session_id)
        }
    
    def _init_database(self):
        """Initialize the SQLite database for session persistence."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_data BLOB NOT NULL,
                created_at TIMESTAMP,
                last_activity TIMESTAMP,
                status TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON sessions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_last_activity ON sessions(last_activity)")
        
        conn.commit()
        conn.close()
        logger.info(f"Initialized database at {self.db_path}")
    
    def _load_single_session(self, session_id: str):
        """Load a single session from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT session_data 
                FROM sessions 
                WHERE session_id = ? AND status IN ('active', 'idle')
            """, (session_id,))
            
            result = cursor.fetchone()
            if result:
                session_data = result[0]
                session = pickle.loads(session_data)
                self.sessions[session_id] = session
                logger.info(f"Loaded session {session_id} from database")
            else:
                logger.warning(f"Session {session_id} not found in database")
                
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
        finally:
            if 'conn' in locals():
                conn.close()

    def _load_sessions(self):
        """Load all sessions from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load active and idle sessions (not archived or closed)
            cursor.execute("""
                SELECT session_id, session_data 
                FROM sessions 
                WHERE status IN ('active', 'idle')
                ORDER BY last_activity DESC
            """)
            
            rows = cursor.fetchall()
            for session_id, session_data in rows:
                try:
                    session = pickle.loads(session_data)
                    self.sessions[session_id] = session
                    logger.info(f"Loaded session {session_id}")
                except Exception as e:
                    logger.error(f"Failed to load session {session_id}: {e}")
            
            conn.close()
            logger.info(f"Loaded {len(self.sessions)} sessions from database")
            
        except Exception as e:
            logger.error(f"Failed to load sessions from database: {e}")
    
    def _save_session(self, session_id: str, session: ChatSession):
        """Save a single session to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            session_data = pickle.dumps(session)
            
            cursor.execute("""
                INSERT OR REPLACE INTO sessions 
                (session_id, user_id, session_data, created_at, last_activity, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                session.user_id,
                session_data,
                session.created_at,
                session.last_activity,
                session.status.value
            ))
            
            conn.commit()
            conn.close()
            logger.debug(f"Saved session {session_id} to database")
            
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
    
    def _save_all_sessions(self):
        """Save all sessions to the database."""
        for session_id, session in self.sessions.items():
            self._save_session(session_id, session)
        logger.info(f"Saved {len(self.sessions)} sessions to database")
    
    async def _cleanup_old_sessions(self):
        """Background task to cleanup old sessions."""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(days=7)  # Archive after 7 days
                
                for session_id, session in list(self.sessions.items()):
                    if session.last_activity < cutoff_time and session.status != SessionStatus.ARCHIVED:
                        session.status = SessionStatus.ARCHIVED
                        self._save_session(session_id, session)
                        # Remove from memory after archiving
                        del self.sessions[session_id]
                        logger.info(f"Archived old session {session_id}")
                
                # Also cleanup database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Delete very old archived sessions (> 30 days)
                old_cutoff = datetime.now() - timedelta(days=30)
                cursor.execute("""
                    DELETE FROM sessions 
                    WHERE status = 'archived' AND last_activity < ?
                """, (old_cutoff,))
                
                deleted = cursor.rowcount
                if deleted > 0:
                    logger.info(f"Deleted {deleted} old archived sessions from database")
                
                conn.commit()
                conn.close()
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(3600)

class PerformanceTracker:
    """Tracks performance metrics for chat sessions."""
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_metrics(self, session_id: str, metrics: Dict[str, Any]):
        """Record performance metrics for a session."""
        if session_id not in self.metrics:
            self.metrics[session_id] = []
        
        metrics["timestamp"] = datetime.now().isoformat()
        self.metrics[session_id].append(metrics)
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get performance summary for a session."""
        session_metrics = self.metrics.get(session_id, [])
        
        if not session_metrics:
            return {}
        
        response_times = [m.get("response_time_ms", 0) for m in session_metrics]
        
        return {
            "total_interactions": len(session_metrics),
            "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "agents_used": list(set(m.get("agent_used") for m in session_metrics if m.get("agent_used"))),
            "last_updated": session_metrics[-1]["timestamp"] if session_metrics else None
        }
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global performance metrics across all sessions."""
        all_metrics = []
        for session_metrics in self.metrics.values():
            all_metrics.extend(session_metrics)
        
        if not all_metrics:
            return {}
        
        response_times = [m.get("response_time_ms", 0) for m in all_metrics]
        
        return {
            "total_sessions": len(self.metrics),
            "total_interactions": len(all_metrics),
            "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "last_24h_interactions": len([
                m for m in all_metrics 
                if datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00")) > datetime.now() - timedelta(days=1)
            ])
        }

# Global chat manager instance
chat_manager = EnhancedChatManager()