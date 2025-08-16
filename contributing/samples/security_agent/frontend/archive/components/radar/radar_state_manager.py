"""
RADAR State Manager - Centralized state management for RADAR phases.

This module manages the state and context flow between RADAR phases,
ensuring consistent data sharing and phase coordination.
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field, asdict
import json
from enum import Enum

logger = logging.getLogger(__name__)


class RADARPhase(Enum):
    """RADAR phase enumeration."""
    RECOGNITION = "recognition"
    ASSESSMENT = "assessment"
    DECISION = "decision"
    ACTION = "action"
    REVIEW = "review"


@dataclass
class PhaseResult:
    """Container for individual phase results."""
    phase: RADARPhase
    status: str  # pending, in_progress, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "phase": self.phase.value,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "results": self.results,
            "errors": self.errors,
            "metadata": self.metadata
        }


@dataclass
class RADARContext:
    """Complete RADAR session context."""
    session_id: str
    project_id: str
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    current_phase: Optional[RADARPhase] = None
    phases: Dict[RADARPhase, PhaseResult] = field(default_factory=dict)
    global_context: Dict[str, Any] = field(default_factory=dict)
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_phase_result(self, phase: RADARPhase) -> Optional[PhaseResult]:
        """Get results for a specific phase."""
        return self.phases.get(phase)
    
    def set_phase_result(self, phase: RADARPhase, result: PhaseResult):
        """Set results for a specific phase."""
        self.phases[phase] = result
        self.updated_at = datetime.now()
    
    def get_context_for_phase(self, phase: RADARPhase) -> Dict[str, Any]:
        """Get accumulated context for a specific phase."""
        context = {"global": self.global_context}
        
        # Add results from all previous phases
        phase_order = list(RADARPhase)
        current_index = phase_order.index(phase)
        
        for i in range(current_index):
            prev_phase = phase_order[i]
            if prev_phase in self.phases:
                context[prev_phase.value] = self.phases[prev_phase].results
        
        return context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "current_phase": self.current_phase.value if self.current_phase else None,
            "phases": {phase.value: result.to_dict() for phase, result in self.phases.items()},
            "global_context": self.global_context,
            "chat_history": self.chat_history
        }


class RADARStateManager:
    """
    Manages RADAR state across phases and sessions.
    
    This class provides centralized state management for the RADAR workflow,
    ensuring consistent data flow between phases and maintaining session context.
    """
    
    def __init__(self):
        """Initialize the RADAR State Manager."""
        self._ensure_session_state()
        logger.info("RADAR State Manager initialized")
    
    def _ensure_session_state(self):
        """Ensure RADAR state exists in Streamlit session."""
        if 'radar_context' not in st.session_state:
            st.session_state.radar_context = None
        
        if 'radar_phase_stack' not in st.session_state:
            st.session_state.radar_phase_stack = []
        
        if 'radar_websocket_connected' not in st.session_state:
            st.session_state.radar_websocket_connected = False
    
    def initialize_context(self, project_id: str, user_id: str = "default") -> RADARContext:
        """
        Initialize a new RADAR context for a session.
        
        Args:
            project_id: GCP project ID
            user_id: User identifier
            
        Returns:
            Initialized RADARContext
        """
        import uuid
        session_id = str(uuid.uuid4())
        
        context = RADARContext(
            session_id=session_id,
            project_id=project_id,
            user_id=user_id
        )
        
        # Initialize all phases as pending
        for phase in RADARPhase:
            context.phases[phase] = PhaseResult(
                phase=phase,
                status="pending"
            )
        
        st.session_state.radar_context = context
        logger.info(f"Initialized RADAR context for session {session_id}")
        
        return context
    
    def get_context(self) -> Optional[RADARContext]:
        """Get current RADAR context."""
        return st.session_state.get('radar_context')
    
    def set_context(self, context: RADARContext):
        """Set RADAR context."""
        st.session_state.radar_context = context
    
    def start_phase(self, phase: RADARPhase) -> bool:
        """
        Start a RADAR phase.
        
        Args:
            phase: Phase to start
            
        Returns:
            True if phase started successfully
        """
        context = self.get_context()
        if not context:
            logger.error("No RADAR context initialized")
            return False
        
        # Check dependencies
        if not self._check_phase_dependencies(phase):
            logger.warning(f"Cannot start {phase.value}: dependencies not met")
            return False
        
        # Update phase status
        phase_result = context.phases[phase]
        phase_result.status = "in_progress"
        phase_result.start_time = datetime.now()
        
        # Update current phase
        context.current_phase = phase
        context.updated_at = datetime.now()
        
        # Add to phase stack
        st.session_state.radar_phase_stack.append(phase)
        
        logger.info(f"Started RADAR phase: {phase.value}")
        return True
    
    def complete_phase(self, phase: RADARPhase, results: Dict[str, Any]) -> bool:
        """
        Complete a RADAR phase with results.
        
        Args:
            phase: Phase to complete
            results: Phase execution results
            
        Returns:
            True if phase completed successfully
        """
        context = self.get_context()
        if not context:
            logger.error("No RADAR context initialized")
            return False
        
        phase_result = context.phases[phase]
        
        # Validate phase is in progress
        if phase_result.status != "in_progress":
            logger.warning(f"Cannot complete {phase.value}: not in progress")
            return False
        
        # Update phase result
        phase_result.status = "completed"
        phase_result.end_time = datetime.now()
        phase_result.results = results
        
        # Update context
        context.updated_at = datetime.now()
        
        # Remove from phase stack if it's the current phase
        if st.session_state.radar_phase_stack and st.session_state.radar_phase_stack[-1] == phase:
            st.session_state.radar_phase_stack.pop()
        
        logger.info(f"Completed RADAR phase: {phase.value}")
        return True
    
    def fail_phase(self, phase: RADARPhase, error: str) -> bool:
        """
        Mark a phase as failed.
        
        Args:
            phase: Phase that failed
            error: Error message
            
        Returns:
            True if phase marked as failed
        """
        context = self.get_context()
        if not context:
            return False
        
        phase_result = context.phases[phase]
        phase_result.status = "failed"
        phase_result.end_time = datetime.now()
        phase_result.errors.append(error)
        
        context.updated_at = datetime.now()
        
        logger.error(f"RADAR phase {phase.value} failed: {error}")
        return True
    
    def _check_phase_dependencies(self, phase: RADARPhase) -> bool:
        """
        Check if phase dependencies are met.
        
        Args:
            phase: Phase to check
            
        Returns:
            True if dependencies are met
        """
        context = self.get_context()
        if not context:
            return False
        
        # Define phase dependencies
        dependencies = {
            RADARPhase.RECOGNITION: [],  # No dependencies
            RADARPhase.ASSESSMENT: [RADARPhase.RECOGNITION],
            RADARPhase.DECISION: [RADARPhase.RECOGNITION, RADARPhase.ASSESSMENT],
            RADARPhase.ACTION: [RADARPhase.RECOGNITION, RADARPhase.ASSESSMENT, RADARPhase.DECISION],
            RADARPhase.REVIEW: []  # Can run independently to review any phase
        }
        
        required_phases = dependencies.get(phase, [])
        
        for req_phase in required_phases:
            phase_result = context.phases.get(req_phase)
            if not phase_result or phase_result.status != "completed":
                return False
        
        return True
    
    def get_phase_dependencies(self, phase: RADARPhase) -> List[RADARPhase]:
        """Get list of required phases for a given phase."""
        dependencies = {
            RADARPhase.RECOGNITION: [],
            RADARPhase.ASSESSMENT: [RADARPhase.RECOGNITION],
            RADARPhase.DECISION: [RADARPhase.RECOGNITION, RADARPhase.ASSESSMENT],
            RADARPhase.ACTION: [RADARPhase.RECOGNITION, RADARPhase.ASSESSMENT, RADARPhase.DECISION],
            RADARPhase.REVIEW: []
        }
        return dependencies.get(phase, [])
    
    def can_execute_phase(self, phase: RADARPhase) -> bool:
        """Check if a phase can be executed."""
        return self._check_phase_dependencies(phase)
    
    def get_completed_phases(self) -> List[RADARPhase]:
        """Get list of completed phases."""
        context = self.get_context()
        if not context:
            return []
        
        completed = []
        for phase, result in context.phases.items():
            if result.status == "completed":
                completed.append(phase)
        
        return completed
    
    def get_pending_phases(self) -> List[RADARPhase]:
        """Get list of pending phases."""
        context = self.get_context()
        if not context:
            return []
        
        pending = []
        for phase, result in context.phases.items():
            if result.status == "pending":
                pending.append(phase)
        
        return pending
    
    def add_chat_message(self, phase: RADARPhase, role: str, content: str):
        """
        Add a chat message to the context.
        
        Args:
            phase: Phase where message was sent
            role: Message role (user/assistant)
            content: Message content
        """
        context = self.get_context()
        if not context:
            return
        
        message = {
            "phase": phase.value,
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        context.chat_history.append(message)
        context.updated_at = datetime.now()
    
    def get_phase_chat_history(self, phase: RADARPhase) -> List[Dict[str, Any]]:
        """Get chat history for a specific phase."""
        context = self.get_context()
        if not context:
            return []
        
        return [msg for msg in context.chat_history if msg.get("phase") == phase.value]
    
    def export_context(self) -> Optional[str]:
        """Export context as JSON string."""
        context = self.get_context()
        if not context:
            return None
        
        return json.dumps(context.to_dict(), indent=2)
    
    def import_context(self, json_str: str) -> bool:
        """
        Import context from JSON string.
        
        Args:
            json_str: JSON representation of context
            
        Returns:
            True if import successful
        """
        try:
            data = json.loads(json_str)
            
            # Reconstruct RADARContext
            context = RADARContext(
                session_id=data["session_id"],
                project_id=data["project_id"],
                user_id=data["user_id"],
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                global_context=data["global_context"],
                chat_history=data["chat_history"]
            )
            
            # Reconstruct phase results
            for phase_str, phase_data in data["phases"].items():
                phase = RADARPhase(phase_str)
                phase_result = PhaseResult(
                    phase=phase,
                    status=phase_data["status"],
                    results=phase_data["results"],
                    errors=phase_data["errors"],
                    metadata=phase_data["metadata"]
                )
                
                if phase_data["start_time"]:
                    phase_result.start_time = datetime.fromisoformat(phase_data["start_time"])
                if phase_data["end_time"]:
                    phase_result.end_time = datetime.fromisoformat(phase_data["end_time"])
                
                context.phases[phase] = phase_result
            
            # Set current phase
            if data["current_phase"]:
                context.current_phase = RADARPhase(data["current_phase"])
            
            self.set_context(context)
            logger.info(f"Imported RADAR context for session {context.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import context: {e}")
            return False
    
    def reset_context(self):
        """Reset the RADAR context."""
        st.session_state.radar_context = None
        st.session_state.radar_phase_stack = []
        st.session_state.radar_websocket_connected = False
        logger.info("RADAR context reset")


# Global state manager instance
radar_state_manager = RADARStateManager()