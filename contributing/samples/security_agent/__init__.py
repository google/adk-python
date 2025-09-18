"""ADK Security Agent Sample package.

This package exposes a top-level `agent` object that ADK eval harnesses expect.
We lazily import `backend.adk_agent` to avoid import-time failures in test
environments that don't have ADK/GenAI packages installed.
"""

try:
    from .backend.adk_agent import security_agent as root_agent
except Exception:
    root_agent = None


class agent:
    root_agent = root_agent
# ADK Security Agent Sample
# Import the agent module required by ADK eval

from .backend.adk_agent import security_agent as root_agent

# Create agent module for ADK eval
class agent:
    root_agent = root_agent