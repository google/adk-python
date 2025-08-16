"""
Simple conversation context manager for maintaining state between queries.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ConversationContext:
    """Manages conversation context and history."""
    
    def __init__(self):
        """Initialize conversation context storage."""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.max_history = 10  # Keep last 10 messages
        self.session_timeout = timedelta(hours=1)  # Sessions expire after 1 hour
    
    def get_or_create_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """Get existing session or create new one."""
        now = datetime.now()
        
        # Clean up expired sessions
        expired = []
        for sid, session in self.sessions.items():
            if now - session['last_activity'] > self.session_timeout:
                expired.append(sid)
        for sid in expired:
            del self.sessions[sid]
            logger.info(f"Expired session: {sid}")
        
        # Get or create session
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'user_id': user_id,
                'created': now,
                'last_activity': now,
                'history': [],
                'context': {}
            }
            logger.info(f"Created new session: {session_id}")
        else:
            self.sessions[session_id]['last_activity'] = now
        
        return self.sessions[session_id]
    
    def add_to_history(self, session_id: str, query: str, response: str):
        """Add query and response to session history."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session['history'].append({
                'timestamp': datetime.now(),
                'query': query,
                'response': response[:500]  # Limit response size
            })
            
            # Keep only recent history
            if len(session['history']) > self.max_history:
                session['history'] = session['history'][-self.max_history:]
    
    def get_context(self, session_id: str) -> str:
        """Get conversation context for the session."""
        if session_id not in self.sessions:
            return ""
        
        session = self.sessions[session_id]
        if not session['history']:
            return ""
        
        # Build context from recent history
        context_parts = ["Previous conversation:"]
        for item in session['history'][-3:]:  # Last 3 exchanges
            context_parts.append(f"User: {item['query']}")
            context_parts.append(f"Assistant: {item['response'][:200]}...")
        
        return "\n".join(context_parts)
    
    def update_context(self, session_id: str, key: str, value: Any):
        """Update session context with specific information."""
        if session_id in self.sessions:
            self.sessions[session_id]['context'][key] = value
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information."""
        if session_id not in self.sessions:
            return {'exists': False}
        
        session = self.sessions[session_id]
        return {
            'exists': True,
            'user_id': session['user_id'],
            'created': session['created'],
            'last_activity': session['last_activity'],
            'history_count': len(session['history']),
            'context_keys': list(session['context'].keys())
        }


# Global instance
conversation_manager = ConversationContext()