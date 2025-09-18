"""
Frontend Agent Service
======================

Service to integrate frontend agents with the chat widget.
Coordinates between FrontendRouterAgent and LocalLookupAgent.
Handles conversation context management.
"""

import logging
from typing import Dict, Any, List, Optional
from frontend.agents.frontend_router import FrontendRouterAgent, LocalLookupAgent, QueryAnalysis
from frontend.services.adk_service import send_message
from frontend.utils.config import FrontendConfig

logger = logging.getLogger(__name__)

class FrontendAgentService:
    """
    Orchestrates frontend agents for intelligent query preprocessing.
    
    This service:
    1. Analyzes queries with conversation context
    2. Checks local cache first
    3. Enhances queries before sending to backend
    4. Manages conversation state
    """
    
    def __init__(self):
        """Initialize the frontend agent service."""
        self.router_agent = FrontendRouterAgent()
        self.local_agent = LocalLookupAgent()
        self.config = FrontendConfig()
        
        # Configuration for agents
        self.max_context_messages = 4  # Last 4 messages for context
        self.enable_local_cache = True
        self.enable_query_enhancement = True
        
        logger.info("Frontend agent service initialized")
        logger.info(f"Router enabled: {self.router_agent.enabled}")
        logger.info(f"Local cache enabled: {self.enable_local_cache}")
    
    def process_query(self, 
                     user_query: str,
                     conversation_history: List[Dict[str, str]] = None,
                     session_id: str = None) -> Dict[str, Any]:
        """
        Process a user query through the frontend agent pipeline.
        
        Args:
            user_query: The user's query
            conversation_history: Recent conversation messages
            session_id: Session identifier for tracking
            
        Returns:
            Response with success status, content, and metadata
        """
        try:
            logger.info(f"Processing query: {user_query[:100]}...")
            
            # Step 1: Check local cache first
            if self.enable_local_cache and self.local_agent.can_handle_locally(user_query):
                local_response = self.local_agent.handle_query(user_query)
                if local_response['success']:
                    logger.info("Query handled by local cache")
                    
                    # If local response suggests backend followup, continue with enhanced query
                    if local_response.get('needs_backend', False):
                        return self._process_with_backend(
                            user_query, 
                            conversation_history, 
                            session_id,
                            local_preface=local_response['response']
                        )
                    else:
                        return {
                            'success': True,
                            'response': local_response['response'],
                            'source': 'local_cache',
                            'metadata': {
                                'cache_hit': True,
                                'enhanced': False
                            }
                        }
            
            # Step 2: Process with backend (with potential enhancement)
            return self._process_with_backend(user_query, conversation_history, session_id)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'success': False,
                'error': f"Frontend agent error: {str(e)}",
                'response': "I encountered an error processing your query. Please try again."
            }
    
    def _process_with_backend(self,
                            user_query: str,
                            conversation_history: List[Dict[str, str]] = None,
                            session_id: str = None,
                            local_preface: str = None) -> Dict[str, Any]:
        """
        Process query with backend, potentially with enhancement.
        
        Args:
            user_query: Original user query
            conversation_history: Conversation context
            session_id: Session identifier
            local_preface: Optional local response to prepend
            
        Returns:
            Backend response with enhancement metadata
        """
        enhanced_query = user_query
        query_metadata = {
            'original_query': user_query,
            'enhanced': False,
            'analysis': None,
            'cache_hit': False
        }
        
        # Step 1: Analyze and potentially enhance query
        if self.enable_query_enhancement and self.router_agent.enabled:
            try:
                # Get recent context for analysis
                context = self._get_recent_context(conversation_history)
                
                # Analyze the query
                analysis = self.router_agent.analyze_query(user_query, context)
                query_metadata['analysis'] = {
                    'query_type': analysis.query_type,
                    'confidence': analysis.confidence,
                    'suggested_tool': analysis.suggested_tool
                }
                
                # Enhance query for backend
                if analysis.needs_backend:
                    enhanced_query = self.router_agent.enhance_for_backend(user_query, analysis)
                    query_metadata['enhanced'] = enhanced_query != user_query
                    
                    if query_metadata['enhanced']:
                        logger.info(f"Query enhanced from: {user_query[:50]}...")
                        logger.info(f"To: {enhanced_query[:50]}...")
                
            except Exception as e:
                logger.warning(f"Query enhancement failed: {e}")
                # Continue with original query
        
        # Step 2: Send to backend
        try:
            backend_response = send_message(enhanced_query, session_id)
            
            if backend_response['success']:
                response_content = backend_response['response']
                
                # If we have a local preface, prepend it
                if local_preface:
                    response_content = f"{local_preface}\n\n---\n\n{response_content}"
                
                return {
                    'success': True,
                    'response': response_content,
                    'source': 'backend',
                    'metadata': query_metadata
                }
            else:
                return {
                    'success': False,
                    'error': backend_response.get('error', 'Backend error'),
                    'response': "I'm having trouble connecting to the backend. Please try again.",
                    'metadata': query_metadata
                }
                
        except Exception as e:
            logger.error(f"Backend communication error: {e}")
            return {
                'success': False,
                'error': f"Backend communication failed: {str(e)}",
                'response': "I'm having trouble connecting to the backend. Please try again.",
                'metadata': query_metadata
            }
    
    def _get_recent_context(self, conversation_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Get recent conversation context for query analysis.
        
        Args:
            conversation_history: Full conversation history
            
        Returns:
            Recent messages for context
        """
        if not conversation_history:
            return []
        
        # Return last N messages, excluding the current query
        return conversation_history[-self.max_context_messages:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about frontend agent usage.
        
        Returns:
            Usage statistics
        """
        return {
            'router_enabled': self.router_agent.enabled,
            'local_cache_enabled': self.enable_local_cache,
            'enhancement_enabled': self.enable_query_enhancement,
            'max_context_messages': self.max_context_messages,
            'local_knowledge_topics': list(self.local_agent.local_knowledge.keys())
        }


# Global instance for use across the frontend
_agent_service = None

def get_agent_service() -> FrontendAgentService:
    """
    Get the global frontend agent service instance.
    
    Returns:
        FrontendAgentService instance
    """
    global _agent_service
    if _agent_service is None:
        _agent_service = FrontendAgentService()
    return _agent_service


# Convenience function for direct use
def process_user_query(user_query: str,
                      conversation_history: List[Dict[str, str]] = None,
                      session_id: str = None) -> Dict[str, Any]:
    """
    Convenience function to process a user query through frontend agents.
    
    Args:
        user_query: The user's query
        conversation_history: Recent conversation messages
        session_id: Session identifier
        
    Returns:
        Processed response
    """
    service = get_agent_service()
    return service.process_query(user_query, conversation_history, session_id)
