# ADK eval wrapper - expects module.agent.root_agent structure
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the actual agent
from agents.agent import root_agent

# Create the nested structure ADK eval expects
class agent:
    """Wrapper class to provide expected structure"""
    root_agent = root_agent

# Export for ADK
__all__ = ['agent']