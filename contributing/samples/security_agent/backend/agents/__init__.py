"""
RADAR Agent System for Cloud Operations

Clean, simple architecture:
- One LLM Coordinator (radar_coordinator.py) 
- Five specialized worker agents (not LLMs)
- Direct API calls, no unnecessary abstraction
"""

# Import the coordinator (includes FastAPI router)
from backend.agents.radar_coordinator import (
    RADARCoordinator,
    create_radar_pipeline,
    router  # FastAPI router for endpoints
)

__all__ = [
    # Convenience functions
    'create_radar_pipeline',
    
    # Coordinator
    'RADARCoordinator',
    
    # FastAPI router
    'router'
]
