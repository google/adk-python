"""
Conversation Memory Manager for ADK Security Agent
Maintains conversation context and history for chat-centric interface
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConversationMessage:
    """Individual message in a conversation"""
    id: str
    role: str  # 'user' or 'assistant' or 'system'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationMessage':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class ConversationContext:
    """Context information extracted from conversation"""
    topic: Optional[str] = None
    entities: List[str] = None  # e.g., ['bucket-name', 'project-id']
    analysis_results: Dict[str, Any] = None  # e.g., bucket analysis results
    recommendations: List[str] = None
    agent_routing: List[str] = None  # Which agents were used
    security_context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.analysis_results is None:
            self.analysis_results = {}
        if self.recommendations is None:
            self.recommendations = []
        if self.agent_routing is None:
            self.agent_routing = []
        if self.security_context is None:
            self.security_context = {}

@dataclass
class ConversationSession:
    """Complete conversation session"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    messages: List[ConversationMessage]
    context: ConversationContext
    status: str = 'active'
    
    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'messages': [msg.to_dict() for msg in self.messages],
            'context': asdict(self.context),
            'status': self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationSession':
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_activity'] = datetime.fromisoformat(data['last_activity'])
        data['messages'] = [ConversationMessage.from_dict(msg) for msg in data['messages']]
        data['context'] = ConversationContext(**data['context'])
        return cls(**data)

class ConversationMemoryManager:
    """Manages conversation memory and context for chat interface"""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
        self.memory_ttl = timedelta(hours=24)  # Session memory retention
    
    def create_session(self, user_id: str) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            messages=[],
            context=ConversationContext()
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created conversation session {session_id} for user {user_id}")
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None) -> str:
        """Add a message to the conversation"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        message_id = str(uuid.uuid4())
        message = ConversationMessage(
            id=message_id,
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        session = self.sessions[session_id]
        session.messages.append(message)
        session.last_activity = datetime.now()
        
        # Update context based on message content
        self._update_context(session, message)
        
        logger.info(f"Added {role} message to session {session_id}: {content[:100]}...")
        return message_id
    
    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[ConversationMessage]:
        """Get recent conversation history"""
        if session_id not in self.sessions:
            return []
        
        messages = self.sessions[session_id].messages
        return messages[-limit:] if limit else messages
    
    def get_conversation_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get the current conversation context"""
        if session_id not in self.sessions:
            return None
        
        return self.sessions[session_id].context
    
    def update_context(self, session_id: str, **context_updates) -> None:
        """Update conversation context with new information"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        context = session.context
        
        for key, value in context_updates.items():
            if hasattr(context, key):
                if isinstance(getattr(context, key), list) and isinstance(value, list):
                    # Merge lists
                    current_list = getattr(context, key)
                    current_list.extend(value)
                    setattr(context, key, list(set(current_list)))  # Remove duplicates
                elif isinstance(getattr(context, key), dict) and isinstance(value, dict):
                    # Merge dictionaries
                    current_dict = getattr(context, key)
                    current_dict.update(value)
                else:
                    setattr(context, key, value)
        
        session.last_activity = datetime.now()
        logger.info(f"Updated context for session {session_id}: {context_updates}")
    
    def _update_context(self, session: ConversationSession, message: ConversationMessage):
        """Update conversation context based on message content"""
        content_lower = message.content.lower()
        
        # Detect topic
        if 'bucket' in content_lower:
            session.context.topic = 'storage_analysis'
            if 'buckets' not in session.context.entities:
                session.context.entities.append('buckets')
        elif 'policy' in content_lower:
            session.context.topic = 'policy_analysis'
            if 'policies' not in session.context.entities:
                session.context.entities.append('policies')
        elif 'vpc' in content_lower or 'network' in content_lower:
            session.context.topic = 'network_analysis'
            if 'networking' not in session.context.entities:
                session.context.entities.append('networking')
        
        # Extract agent routing information from metadata
        if message.metadata and 'agent_used' in message.metadata:
            agent_name = message.metadata['agent_used']
            if agent_name not in session.context.agent_routing:
                session.context.agent_routing.append(agent_name)
    
    def get_context_for_agent_routing(self, session_id: str) -> Dict[str, Any]:
        """Get context information for intelligent agent routing"""
        context = self.get_conversation_context(session_id)
        if not context:
            return {}
        
        return {
            'topic': context.topic,
            'entities': context.entities,
            'previous_agents': context.agent_routing,
            'has_analysis_results': bool(context.analysis_results),
            'security_context': context.security_context
        }
    
    def cleanup_expired_sessions(self):
        """Remove expired conversation sessions"""
        cutoff_time = datetime.now() - self.memory_ttl
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.last_activity < cutoff_time
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session {session_id}")
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the conversation session"""
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        return {
            'session_id': session_id,
            'message_count': len(session.messages),
            'duration_minutes': (session.last_activity - session.created_at).total_seconds() / 60,
            'topic': session.context.topic,
            'entities': session.context.entities,
            'agents_used': session.context.agent_routing,
            'has_recommendations': bool(session.context.recommendations)
        }

# Global instance for the application
conversation_memory = ConversationMemoryManager()