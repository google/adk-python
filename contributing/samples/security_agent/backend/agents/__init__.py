"""
ADK-compliant agent implementations following Google ADK patterns.

This module provides specialized agents with tools for GCP security analysis.
Each agent follows the ADK architecture:
- Agent: AI brain with specific instructions
- Tools: Python functions that grant capabilities
- Runner: Orchestrates agent execution
- SessionService: Manages conversation state
"""

from typing import List, Dict, Any

# Import all agents for easy access
__all__ = [
    'SecurityCoordinatorAgent',
    'StorageSecurityAgent', 
    'IAMSecurityAgent',
    'NetworkSecurityAgent',
    'ComplianceAgent',
    'CostOptimizationAgent',
    'create_agent_with_tools'
]