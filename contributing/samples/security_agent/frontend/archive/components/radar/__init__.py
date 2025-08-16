"""
RADAR Component Package - Frontend components for RADAR methodology.

This package contains all the frontend components for implementing
the RADAR (Recognize, Assess, Decide, Act, Review) security analysis workflow.
"""

from .radar_state_manager import (
    RADARStateManager,
    RADARPhase,
    RADARContext,
    PhaseResult,
    radar_state_manager
)

from .radar_coordinator_view import render_radar_coordinator_view
from .recognition_chat_view import render_recognition_chat_view

__all__ = [
    # State management
    'RADARStateManager',
    'RADARPhase',
    'RADARContext',
    'PhaseResult',
    'radar_state_manager',
    
    # View functions
    'render_radar_coordinator_view',
    'render_recognition_chat_view',
]