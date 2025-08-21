"""ADK Agent module for vertex sqlite security agent"""

import sys
from pathlib import Path

# Add parent directories to path
gcp_security_dir = Path(__file__).parent.parent
security_agent_dir = gcp_security_dir.parent.parent

if str(gcp_security_dir) not in sys.path:
    sys.path.insert(0, str(gcp_security_dir))
if str(security_agent_dir) not in sys.path:
    sys.path.insert(0, str(security_agent_dir))

# Import the root agent
from vertex_sqlite_agent import root_agent