"""
RADAR WebSocket Client for Real-time Updates

This module provides WebSocket connectivity for real-time RADAR phase
updates, progress streaming, and live coordination between frontend and backend.

Key Features:
- Real-time phase progress updates
- Live streaming of agent responses
- Cross-phase coordination events
- Connection management and recovery
- Event filtering and routing
"""

import asyncio
import json
import logging
import streamlit as st
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import websockets

from frontend.components.radar.radar_state_manager import radar_state, RADARPhase, PhaseStatus, PhaseResult

logger = logging.getLogger(__name__)


class RADARWebSocketClient:
    """
    WebSocket client for real-time RADAR coordination.
    
    Manages WebSocket connections to the backend RADAR coordinator
    and handles real-time events and updates.
    """
    
    def __init__(self):
        """Initialize RADAR WebSocket client."""
        self.websocket = None
        self.connected = False
        self.backend_ws_url = self._get_websocket_url()
        self.event_handlers = {}
        self._setup_default_handlers()
    
    def _get_websocket_url(self) -> str:
        """Get WebSocket URL from configuration."""
        backend_host = st.secrets.get("BACKEND_HOST", "localhost")
        backend_port = st.secrets.get("BACKEND_PORT", "8000")
        return f"ws://{backend_host}:{backend_port}/api/v1/radar/ws"
    
    def _setup_default_handlers(self):
        """Setup default event handlers."""
        self.event_handlers = {
            "phase_started": self._handle_phase_started,
            "phase_progress": self._handle_phase_progress,
            "phase_completed": self._handle_phase_completed,
            "phase_error": self._handle_phase_error,
            "streaming_response": self._handle_streaming_response,
            "coordination_update": self._handle_coordination_update
        }
    
    async def connect(self, user_id: str = "default") -> bool:
        """
        Connect to RADAR WebSocket endpoint.
        
        Args:
            user_id: User identifier for the connection
            
        Returns:
            True if connection successful
        """
        try:
            self.websocket = await websockets.connect(
                f"{self.backend_ws_url}?user_id={user_id}"
            )
            self.connected = True
            logger.info(f"🔌 Connected to RADAR WebSocket: {self.backend_ws_url}")
            
            # Update session state
            st.session_state.radar_websocket_connected = True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to RADAR WebSocket: {e}")
            self.connected = False
            st.session_state.radar_websocket_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.connected = False
        st.session_state.radar_websocket_connected = False
        logger.info("🔌 Disconnected from RADAR WebSocket")
    
    async def send_message(self, message: Dict[str, Any]):
        """
        Send message to backend via WebSocket.
        
        Args:
            message: Message to send
        """
        if not self.connected or not self.websocket:
            logger.warning("WebSocket not connected, cannot send message")
            return
        
        try:
            await self.websocket.send(json.dumps(message))
            logger.debug(f"📤 Sent WebSocket message: {message.get('type', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            await self._handle_connection_error()
    
    async def listen_for_events(self):
        """
        Listen for incoming WebSocket events.
        
        This should be run in a background task to continuously
        receive and process events from the backend.
        """
        if not self.connected or not self.websocket:
            logger.warning("WebSocket not connected, cannot listen for events")
            return
        
        try:
            async for message in self.websocket:
                try:
                    event = json.loads(message)
                    await self._handle_event(event)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket event: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            await self._handle_connection_error()
        except Exception as e:
            logger.error(f"Error in WebSocket event loop: {e}")
            await self._handle_connection_error()
    
    async def start_radar_session(self, query: str, project_id: str) -> bool:
        """
        Start a new RADAR session via WebSocket.
        
        Args:
            query: User query to process
            project_id: GCP project ID
            
        Returns:
            True if session started successfully
        """
        message = {
            "type": "start_radar_session",
            "data": {
                "query": query,
                "project_id": project_id,
                "session_id": radar_state.get_current_context().session_id if radar_state.get_current_context() else None,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        await self.send_message(message)
        return True
    
    async def execute_phase(self, phase: RADARPhase, context: Dict[str, Any]) -> bool:
        """
        Execute a specific RADAR phase via WebSocket.
        
        Args:
            phase: Phase to execute
            context: Phase execution context
            
        Returns:
            True if execution started successfully
        """
        message = {
            "type": "execute_phase",
            "data": {
                "phase": phase.value,
                "context": context,
                "session_id": radar_state.get_current_context().session_id if radar_state.get_current_context() else None,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        await self.send_message(message)
        return True
    
    async def _handle_event(self, event: Dict[str, Any]):
        """
        Handle incoming WebSocket event.
        
        Args:
            event: Event data from backend
        """
        event_type = event.get("type")
        
        if event_type in self.event_handlers:
            handler = self.event_handlers[event_type]
            await handler(event.get("data", {}))
        else:
            logger.warning(f"Unknown WebSocket event type: {event_type}")
            
        # Store event in session state for UI updates
        if 'radar_streaming_events' not in st.session_state:
            st.session_state.radar_streaming_events = []
        
        st.session_state.radar_streaming_events.append({
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": event.get("data", {})
        })
        
        # Keep only last 100 events
        if len(st.session_state.radar_streaming_events) > 100:
            st.session_state.radar_streaming_events = st.session_state.radar_streaming_events[-100:]
    
    async def _handle_phase_started(self, data: Dict[str, Any]):
        """Handle phase started event."""
        phase_name = data.get("phase")
        logger.info(f"🚀 Phase started: {phase_name}")
        
        # Update radar state
        try:
            phase = RADARPhase(phase_name)
            # Create in-progress result
            result = PhaseResult(
                phase=phase,
                status=PhaseStatus.IN_PROGRESS,
                data={"started_at": datetime.now().isoformat()},
                timestamp=datetime.now().isoformat()
            )
            radar_state.store_phase_result(result)
        except ValueError:
            logger.warning(f"Unknown phase in phase_started event: {phase_name}")
    
    async def _handle_phase_progress(self, data: Dict[str, Any]):
        """Handle phase progress update."""
        phase_name = data.get("phase")
        progress = data.get("progress", 0)
        message = data.get("message", "")
        
        logger.debug(f"📊 Phase progress {phase_name}: {progress}% - {message}")
        
        # Could update UI progress indicators here
        # For now, just store in session state
        if 'radar_phase_progress' not in st.session_state:
            st.session_state.radar_phase_progress = {}
        
        st.session_state.radar_phase_progress[phase_name] = {
            "progress": progress,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_phase_completed(self, data: Dict[str, Any]):
        """Handle phase completed event."""
        phase_name = data.get("phase")
        results = data.get("results", {})
        
        logger.info(f"✅ Phase completed: {phase_name}")
        
        # Update radar state
        try:
            phase = RADARPhase(phase_name)
            result = PhaseResult(
                phase=phase,
                status=PhaseStatus.COMPLETED,
                data=results,
                timestamp=datetime.now().isoformat(),
                duration_seconds=data.get("duration_seconds")
            )
            radar_state.store_phase_result(result)
            
            # Auto-proceed to next phase if enabled
            if st.session_state.get('radar_auto_proceed', False):
                next_phase = radar_state.get_next_phase()
                if next_phase:
                    radar_state.update_phase(next_phase)
                    
        except ValueError:
            logger.warning(f"Unknown phase in phase_completed event: {phase_name}")
    
    async def _handle_phase_error(self, data: Dict[str, Any]):
        """Handle phase error event."""
        phase_name = data.get("phase")
        error_message = data.get("error", "Unknown error")
        
        logger.error(f"❌ Phase error {phase_name}: {error_message}")
        
        # Update radar state
        try:
            phase = RADARPhase(phase_name)
            result = PhaseResult(
                phase=phase,
                status=PhaseStatus.ERROR,
                data={"error": error_message},
                timestamp=datetime.now().isoformat(),
                error_message=error_message
            )
            radar_state.store_phase_result(result)
        except ValueError:
            logger.warning(f"Unknown phase in phase_error event: {phase_name}")
    
    async def _handle_streaming_response(self, data: Dict[str, Any]):
        """Handle streaming response chunk."""
        phase_name = data.get("phase")
        chunk = data.get("chunk", "")
        is_final = data.get("is_final", False)
        
        # Store streaming chunks for real-time display
        if 'radar_streaming_chunks' not in st.session_state:
            st.session_state.radar_streaming_chunks = {}
        
        if phase_name not in st.session_state.radar_streaming_chunks:
            st.session_state.radar_streaming_chunks[phase_name] = []
        
        st.session_state.radar_streaming_chunks[phase_name].append({
            "chunk": chunk,
            "timestamp": datetime.now().isoformat(),
            "is_final": is_final
        })
    
    async def _handle_coordination_update(self, data: Dict[str, Any]):
        """Handle coordination update between phases."""
        update_type = data.get("update_type")
        message = data.get("message", "")
        
        logger.info(f"🔄 Coordination update: {update_type} - {message}")
        
        # Store coordination updates
        if 'radar_coordination_updates' not in st.session_state:
            st.session_state.radar_coordination_updates = []
        
        st.session_state.radar_coordination_updates.append({
            "type": update_type,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _handle_connection_error(self):
        """Handle WebSocket connection errors."""
        self.connected = False
        st.session_state.radar_websocket_connected = False
        
        # Attempt to reconnect
        logger.info("🔄 Attempting to reconnect...")
        await asyncio.sleep(5)  # Wait before reconnecting
        
        # Try to reconnect (would need user_id from somewhere)
        # await self.connect("default")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """
        Add custom event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Async function to handle the event
        """
        self.event_handlers[event_type] = handler
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status."""
        return {
            "connected": self.connected,
            "websocket_url": self.backend_ws_url,
            "last_event": st.session_state.get('radar_streaming_events', [])[-1] if st.session_state.get('radar_streaming_events') else None
        }


# Global WebSocket client instance
radar_ws_client = RADARWebSocketClient()