"""WebSocket manager for real-time chat communication and event broadcasting."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Callable
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger(__name__)

@dataclass
class WebSocketConnection:
    """Represents a WebSocket connection with metadata."""
    websocket: WebSocket
    connection_id: str
    user_id: str
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    subscriptions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

class EventType:
    """Constants for different event types."""
    CHAT_MESSAGE = "chat_message"
    AGENT_STATUS = "agent_status"
    DELEGATION_DECISION = "delegation_decision"
    PERFORMANCE_METRICS = "performance_metrics"
    TYPING_INDICATOR = "typing_indicator"
    CONNECTION_STATUS = "connection_status"
    ERROR = "error"
    SYSTEM_NOTIFICATION = "system_notification"
    CONVERSATION_UPDATE = "conversation_update"
    USER_PRESENCE = "user_presence"

class WebSocketManager:
    """Enhanced WebSocket manager for real-time communication."""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}  # connection_id -> connection
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
        self.room_subscriptions: Dict[str, Set[str]] = {}  # room_name -> set of connection_ids
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Statistics
        self.total_connections = 0
        self.total_messages_sent = 0
        
    async def connect(
        self, 
        websocket: WebSocket, 
        user_id: str, 
        metadata: Dict[str, Any] = None
    ) -> str:
        """Connect a new WebSocket client."""
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        connection = WebSocketConnection(
            websocket=websocket,
            connection_id=connection_id,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        # Store connection
        self.connections[connection_id] = connection
        
        # Update user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        self.total_connections += 1
        
        # Send welcome message
        await self._send_to_connection(connection_id, {
            "type": EventType.CONNECTION_STATUS,
            "status": "connected",
            "connection_id": connection_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Broadcast user presence
        await self.broadcast_user_presence(user_id, "online")
        
        logger.info(f"WebSocket connected: {connection_id} for user {user_id}")
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Disconnect a WebSocket client."""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        user_id = connection.user_id
        
        # Remove from connections
        del self.connections[connection_id]
        
        # Update user connections
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
                # User is now offline
                await self.broadcast_user_presence(user_id, "offline")
        
        # Remove from room subscriptions
        for room_connections in self.room_subscriptions.values():
            room_connections.discard(connection_id)
        
        logger.info(f"WebSocket disconnected: {connection_id} for user {user_id}")
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send message to all connections for a specific user."""
        if user_id not in self.user_connections:
            return
        
        message["timestamp"] = datetime.now().isoformat()
        
        failed_connections = []
        for connection_id in self.user_connections[user_id]:
            success = await self._send_to_connection(connection_id, message)
            if not success:
                failed_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)
    
    async def send_to_room(self, room: str, message: Dict[str, Any]):
        """Send message to all connections subscribed to a room."""
        if room not in self.room_subscriptions:
            return
        
        message["timestamp"] = datetime.now().isoformat()
        
        failed_connections = []
        for connection_id in self.room_subscriptions[room].copy():
            success = await self._send_to_connection(connection_id, message)
            if not success:
                failed_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)
    
    async def broadcast(self, message: Dict[str, Any], exclude_user: str = None):
        """Broadcast message to all connected clients."""
        message["timestamp"] = datetime.now().isoformat()
        
        failed_connections = []
        for connection_id, connection in self.connections.items():
            if exclude_user and connection.user_id == exclude_user:
                continue
                
            success = await self._send_to_connection(connection_id, message)
            if not success:
                failed_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)
    
    async def _send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send message to a specific connection."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        try:
            await connection.websocket.send_text(json.dumps(message))
            connection.last_activity = datetime.now()
            self.total_messages_sent += 1
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            return False
    
    async def subscribe_to_room(self, connection_id: str, room: str):
        """Subscribe a connection to a room."""
        if connection_id not in self.connections:
            return
        
        if room not in self.room_subscriptions:
            self.room_subscriptions[room] = set()
        
        self.room_subscriptions[room].add(connection_id)
        self.connections[connection_id].subscriptions.add(room)
        
        await self._send_to_connection(connection_id, {
            "type": EventType.SYSTEM_NOTIFICATION,
            "message": f"Subscribed to room: {room}",
            "room": room
        })
    
    async def unsubscribe_from_room(self, connection_id: str, room: str):
        """Unsubscribe a connection from a room."""
        if connection_id not in self.connections:
            return
        
        if room in self.room_subscriptions:
            self.room_subscriptions[room].discard(connection_id)
        
        self.connections[connection_id].subscriptions.discard(room)
        
        await self._send_to_connection(connection_id, {
            "type": EventType.SYSTEM_NOTIFICATION,
            "message": f"Unsubscribed from room: {room}",
            "room": room
        })
    
    async def handle_message(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        connection.last_activity = datetime.now()
        
        message_type = message_data.get("type")
        
        if message_type == "ping":
            await self._send_to_connection(connection_id, {
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            })
        
        elif message_type == "subscribe":
            room = message_data.get("room")
            if room:
                await self.subscribe_to_room(connection_id, room)
        
        elif message_type == "unsubscribe":
            room = message_data.get("room")
            if room:
                await self.unsubscribe_from_room(connection_id, room)
        
        elif message_type == "typing_start":
            await self.broadcast_typing_indicator(connection.user_id, True)
        
        elif message_type == "typing_stop":
            await self.broadcast_typing_indicator(connection.user_id, False)
        
        elif message_type == "get_status":
            await self._send_to_connection(connection_id, {
                "type": EventType.CONNECTION_STATUS,
                "status": "active",
                "user_id": connection.user_id,
                "connected_at": connection.connected_at.isoformat(),
                "subscriptions": list(connection.subscriptions)
            })
        
        # Trigger custom event handlers
        if message_type in self.event_handlers:
            for handler in self.event_handlers[message_type]:
                try:
                    await handler(connection, message_data)
                except Exception as e:
                    logger.error(f"Error in event handler for {message_type}: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add custom event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def broadcast_agent_status(self, agent_data: Dict[str, Any]):
        """Broadcast agent status update."""
        await self.broadcast({
            "type": EventType.AGENT_STATUS,
            "data": agent_data
        })
    
    async def broadcast_delegation_decision(self, decision_data: Dict[str, Any]):
        """Broadcast delegation decision."""
        await self.broadcast({
            "type": EventType.DELEGATION_DECISION,
            "data": decision_data
        })
    
    async def broadcast_performance_metrics(self, metrics: Dict[str, Any]):
        """Broadcast performance metrics."""
        await self.broadcast({
            "type": EventType.PERFORMANCE_METRICS,
            "data": metrics
        })
    
    async def broadcast_typing_indicator(self, user_id: str, is_typing: bool):
        """Broadcast typing indicator."""
        await self.broadcast({
            "type": EventType.TYPING_INDICATOR,
            "user_id": user_id,
            "is_typing": is_typing
        }, exclude_user=user_id)
    
    async def broadcast_user_presence(self, user_id: str, status: str):
        """Broadcast user presence update."""
        await self.broadcast({
            "type": EventType.USER_PRESENCE,
            "user_id": user_id,
            "status": status  # online, offline, away
        }, exclude_user=user_id)
    
    async def send_chat_response(
        self, 
        user_id: str, 
        response: str, 
        agent_used: str = None,
        suggestions: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Send a chat response to a user."""
        await self.send_to_user(user_id, {
            "type": EventType.CHAT_MESSAGE,
            "response": response,
            "agent_used": agent_used,
            "suggestions": suggestions or [],
            "metadata": metadata or {}
        })
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        active_users = len(self.user_connections)
        total_connections = len(self.connections)
        rooms = len(self.room_subscriptions)
        
        return {
            "active_users": active_users,
            "total_connections": total_connections,
            "rooms": rooms,
            "total_connections_served": self.total_connections,
            "total_messages_sent": self.total_messages_sent,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_user_connections(self, user_id: str) -> List[Dict[str, Any]]:
        """Get connection info for a user."""
        if user_id not in self.user_connections:
            return []
        
        connections = []
        for connection_id in self.user_connections[user_id]:
            if connection_id in self.connections:
                conn = self.connections[connection_id]
                connections.append({
                    "connection_id": connection_id,
                    "connected_at": conn.connected_at.isoformat(),
                    "last_activity": conn.last_activity.isoformat(),
                    "subscriptions": list(conn.subscriptions),
                    "metadata": conn.metadata
                })
        
        return connections

# Global WebSocket manager instance
websocket_manager = WebSocketManager()

async def websocket_endpoint(websocket: WebSocket, user_id: str = "default"):
    """Main WebSocket endpoint handler."""
    connection_id = await websocket_manager.connect(websocket, user_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                await websocket_manager.handle_message(connection_id, message_data)
            except json.JSONDecodeError:
                await websocket_manager._send_to_connection(connection_id, {
                    "type": EventType.ERROR,
                    "error": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await websocket_manager._send_to_connection(connection_id, {
                    "type": EventType.ERROR,
                    "error": f"Error processing message: {str(e)}"
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error for connection {connection_id}: {e}")
    finally:
        await websocket_manager.disconnect(connection_id)