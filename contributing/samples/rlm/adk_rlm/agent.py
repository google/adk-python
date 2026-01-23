"""
ADK agent entry point for the built-in web interface.

This module exposes the root_agent for ADK's web UI and CLI tools.
Run with: adk web adk_rlm
"""

from adk_rlm.agents.rlm_agent import RLMAgent

root_agent = RLMAgent()
