"""GCP Security Agent Module - ADK Compatible"""

# Import the agent and expose it as root_agent for ADK
import sys
import os
from pathlib import Path

# Add parent directories to path to allow imports
agent_dir = Path(__file__).parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

# Set up Google Application Credentials before importing agent
if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
    creds_path = agent_dir / "mgm-digitalconcierge-8ba3b2f28e5f.json"
    if creds_path.exists():
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(creds_path)
        print(f"✅ Set GOOGLE_APPLICATION_CREDENTIALS in __init__.py")

# Import the appropriate agent based on configuration
import os

# FORCE SQLite agent - the only one with proper multi-tool support
# The vertex_agent only has google_search and can't access project data
from .vertex_sqlite_agent import root_agent
print("✅ Using SQLite-based agent with query_security_data tool")

# Create agent submodule for ADK compatibility
# ADK expects agent_module.agent.root_agent structure
class AgentModule:
    """Agent submodule to satisfy ADK's expected structure"""
    def __init__(self):
        self.root_agent = root_agent

# Create the agent submodule that ADK expects
agent = AgentModule()

# Note: Set AGENT_MODE in .env to choose:
# - 'sqlite': Full data access via SQLite (recommended)
# - 'simple': No tools, just conversation
# - 'search': Google search only