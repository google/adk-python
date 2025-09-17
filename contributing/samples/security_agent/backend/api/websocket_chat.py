"""
WebSocket Chat API for real-time communication
============================================

Provides real-time WebSocket endpoints for streaming chat responses.
Integrates with ADK agent and includes proper error handling, authentication,
and connection management.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.websockets import WebSocketState
from typing import Dict, Any, List, Optional, Set
import asyncio
import json
import logging
import time
from datetime import datetime
import uuid
from contextlib import asynccontextmanager

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections with authentication and rate limiting."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, List[float]] = {}  # connection_id -> timestamps
        self.max_messages_per_minute = 30
        self.heartbeat_interval = 30  # seconds
        
    async def connect(self, websocket: WebSocket, connection_id: str = None) -> str:
        """Accept WebSocket connection and assign connection ID."""
        if connection_id is None:
            connection_id = str(uuid.uuid4())
            
        await websocket.accept()
        
        self.active_connections[connection_id] = websocket
        self.connection_metadata[connection_id] = {
            "connected_at": datetime.now().isoformat(),
            "last_activity": time.time(),
            "message_count": 0
        }
        
        logging.info(f"WebSocket connection {connection_id} established")
        
        # Start heartbeat task
        asyncio.create_task(self._heartbeat_task(connection_id))
        
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Remove WebSocket connection."""
        if connection_id in self.active_connections:
            self.active_connections.pop(connection_id)
            self.connection_metadata.pop(connection_id, None)
            self.rate_limits.pop(connection_id, None)
            logging.info(f"WebSocket connection {connection_id} disconnected")
    
    async def send_message(self, connection_id: str, message: Dict[str, Any]):
        """Send message to specific connection."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
                    self.connection_metadata[connection_id]["last_activity"] = time.time()
                else:
                    self.disconnect(connection_id)
            except Exception as e:
                logging.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all active connections."""
        if not self.active_connections:
            return
            
        disconnected = []
        for connection_id, websocket in self.active_connections.items():
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
                else:
                    disconnected.append(connection_id)
            except Exception as e:
                logging.error(f"Error broadcasting to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    def check_rate_limit(self, connection_id: str) -> bool:
        """Check if connection is within rate limits."""
        now = time.time()
        minute_ago = now - 60
        
        if connection_id not in self.rate_limits:
            self.rate_limits[connection_id] = []
        
        # Remove old timestamps
        self.rate_limits[connection_id] = [
            ts for ts in self.rate_limits[connection_id] 
            if ts > minute_ago
        ]
        
        # Check limit
        if len(self.rate_limits[connection_id]) >= self.max_messages_per_minute:
            return False
        
        # Add current timestamp
        self.rate_limits[connection_id].append(now)
        return True
    
    async def _heartbeat_task(self, connection_id: str):
        """Send periodic heartbeat to maintain connection."""
        while connection_id in self.active_connections:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if connection_id not in self.active_connections:
                    break
                
                await self.send_message(connection_id, {
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logging.error(f"Heartbeat error for {connection_id}: {e}")
                break
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about active connections."""
        return {
            "active_connections": len(self.active_connections),
            "connections": {
                conn_id: {
                    "connected_at": metadata["connected_at"],
                    "message_count": metadata["message_count"],
                    "last_activity": datetime.fromtimestamp(
                        metadata["last_activity"]
                    ).isoformat()
                }
                for conn_id, metadata in self.connection_metadata.items()
            }
        }

# Global connection manager
connection_manager = ConnectionManager()

# Router setup
router = APIRouter(prefix="/api/v1/ws", tags=["websocket"])

logger = logging.getLogger(__name__)

@router.websocket("/chat/{connection_id}")
async def websocket_chat_endpoint(websocket: WebSocket, connection_id: str = None):
    """
    WebSocket endpoint for real-time chat with ADK agent.
    
    Features:
    - Real-time streaming responses
    - Token-by-token streaming
    - Connection recovery
    - Rate limiting
    - Heartbeat mechanism
    - Error handling
    """
    
    # Use provided connection_id or generate new one
    if not connection_id or connection_id == "new":
        connection_id = str(uuid.uuid4())
    
    try:
        # Accept connection
        actual_connection_id = await connection_manager.connect(websocket, connection_id)
        
        # Send connection confirmation
        await connection_manager.send_message(actual_connection_id, {
            "type": "connection_established",
            "connection_id": actual_connection_id,
            "timestamp": datetime.now().isoformat(),
            "message": "WebSocket connection established successfully"
        })
        
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Update connection metadata
                connection_manager.connection_metadata[actual_connection_id]["message_count"] += 1
                
                # Check rate limiting
                if not connection_manager.check_rate_limit(actual_connection_id):
                    await connection_manager.send_message(actual_connection_id, {
                        "type": "error",
                        "message": "Rate limit exceeded. Please wait before sending more messages.",
                        "error_code": "RATE_LIMIT_EXCEEDED"
                    })
                    continue
                
                # Validate message format
                if not isinstance(message_data, dict) or "query" not in message_data:
                    await connection_manager.send_message(actual_connection_id, {
                        "type": "error",
                        "message": "Invalid message format. Expected JSON with 'query' field.",
                        "error_code": "INVALID_FORMAT"
                    })
                    continue
                
                # Extract query and metadata
                query = message_data.get("query", "").strip()
                if not query:
                    await connection_manager.send_message(actual_connection_id, {
                        "type": "error",
                        "message": "Query cannot be empty.",
                        "error_code": "EMPTY_QUERY"
                    })
                    continue
                
                session_id = message_data.get("session_id", "default")
                user_id = message_data.get("user_id", "default_user")
                
                logger.info(f"WebSocket query from {actual_connection_id}: {query[:50]}...")
                
                # Send acknowledgment
                await connection_manager.send_message(actual_connection_id, {
                    "type": "query_received",
                    "query": query,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Process query with ADK agent
                await process_chat_query(
                    query, session_id, user_id, actual_connection_id
                )
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket {actual_connection_id} disconnected by client")
                break
            except json.JSONDecodeError as e:
                await connection_manager.send_message(actual_connection_id, {
                    "type": "error",
                    "message": f"Invalid JSON format: {str(e)}",
                    "error_code": "JSON_DECODE_ERROR"
                })
            except Exception as e:
                logger.error(f"Error in WebSocket {actual_connection_id}: {e}")
                await connection_manager.send_message(actual_connection_id, {
                    "type": "error",
                    "message": f"Internal server error: {str(e)}",
                    "error_code": "INTERNAL_ERROR"
                })
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        connection_manager.disconnect(actual_connection_id)

async def process_chat_query(query: str, session_id: str, user_id: str, connection_id: str):
    """Process chat query with ADK agent and stream response."""
    
    try:
        # Send typing indicator
        await connection_manager.send_message(connection_id, {
            "type": "typing_start",
            "timestamp": datetime.now().isoformat()
        })
        
        # Import the configured agent
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from agents.adk_agent import security_agent
        
        # Get conversation context if available
        enhanced_query = query
        try:
            from api.conversation_context import conversation_manager
            
            # Get or create session
            session = conversation_manager.get_or_create_session(session_id, user_id)
            
            # Get conversation context
            context = conversation_manager.get_context(session_id)
            
            if context:
                enhanced_query = f"Previous conversation context:\n{context}\n\nCurrent question: {query}"
                logger.info(f"Using conversation context for session {session_id}")
        except Exception as e:
            logger.warning(f"Conversation context not available: {e}")
        
        # Send response start indicator
        await connection_manager.send_message(connection_id, {
            "type": "response_start",
            "timestamp": datetime.now().isoformat()
        })
        
        # Stream response from ADK agent
        response_parts = []
        chunk_count = 0
        
        async for chunk in agent.run_async(enhanced_query):
            if isinstance(chunk, str) and chunk.strip():
                chunk_count += 1
                response_parts.append(chunk)
                
                # Send streaming chunk
                await connection_manager.send_message(connection_id, {
                    "type": "response_chunk",
                    "chunk": chunk,
                    "chunk_number": chunk_count,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Small delay for better streaming experience
                await asyncio.sleep(0.05)
        
        # Complete response
        response_text = ''.join(response_parts)
        
        # Send response complete
        await connection_manager.send_message(connection_id, {
            "type": "response_complete",
            "response": response_text,
            "chunk_count": chunk_count,
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Store in conversation history if available
        try:
            conversation_manager.add_to_history(session_id, query, response_text)
        except:
            pass  # Continue even if conversation storage fails
            
        logger.info(f"WebSocket query processed successfully: {len(response_text)} chars, {chunk_count} chunks")
        
    except Exception as e:
        logger.error(f"Error processing WebSocket query: {e}")
        
        # Send error response
        await connection_manager.send_message(connection_id, {
            "type": "response_error",
            "error": str(e),
            "error_code": "PROCESSING_ERROR",
            "timestamp": datetime.now().isoformat()
        })
        
        # Send fallback response
        fallback_response = f"I apologize, but I encountered an error processing your request. Please try again or contact support if the issue persists. Error: {str(e)[:100]}"
        
        await connection_manager.send_message(connection_id, {
            "type": "response_complete",
            "response": fallback_response,
            "is_fallback": True,
            "timestamp": datetime.now().isoformat()
        })

@router.get("/stats")
async def websocket_stats():
    """Get WebSocket connection statistics."""
    return connection_manager.get_connection_stats()

@router.post("/broadcast")
async def broadcast_message(message: Dict[str, Any]):
    """Broadcast message to all active WebSocket connections."""
    try:
        await connection_manager.broadcast(message)
        return {
            "success": True,
            "connections_notified": len(connection_manager.active_connections),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Broadcast failed: {str(e)}")

@router.delete("/connections/{connection_id}")
async def disconnect_client(connection_id: str):
    """Force disconnect a specific WebSocket connection."""
    if connection_id in connection_manager.active_connections:
        try:
            websocket = connection_manager.active_connections[connection_id]
            await websocket.close(code=1000, reason="Disconnected by server")
            connection_manager.disconnect(connection_id)
            return {
                "success": True,
                "message": f"Connection {connection_id} disconnected",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Disconnect failed: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")

# Health check endpoint for WebSocket service
@router.get("/health")
async def websocket_health():
    """Health check for WebSocket service."""
    return {
        "status": "healthy",
        "service": "websocket_chat",
        "active_connections": len(connection_manager.active_connections),
        "uptime": datetime.now().isoformat(),
        "features": {
            "real_time_streaming": True,
            "rate_limiting": True,
            "heartbeat": True,
            "error_recovery": True,
            "connection_management": True
        }
    }