# ADK Security Agent Sample
# Import the agent module required by ADK eval

from .backend.adk_agent import security_agent as root_agent

# Create agent module for ADK eval
class agent:
    root_agent = root_agent