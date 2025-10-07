#!/usr/bin/env python3
"""Chainlit web application for the BigQuery Security Agent.

This is a standalone Chainlit app that uses the modular SecurityAgentProfile class.
It can also serve as a reference for integrating into existing Chainlit applications.

Standalone usage:
    chainlit run chainlit_app.py

Integration usage:
    from chainlit_agent import SecurityAgentProfile
    # See docs/CHAINLIT_PLUGIN_INTEGRATION.md for details
"""

from __future__ import annotations

import chainlit as cl

# Import the modular security agent
from chainlit_agent import SecurityAgentProfile


@cl.set_chat_profiles
async def chat_profile():
    """Define security agent profiles using the modular class."""
    return SecurityAgentProfile.get_profiles()


@cl.on_chat_start
async def start():
    """Initialize a new chat session for security agent."""
    await SecurityAgentProfile.on_chat_start()


@cl.on_message
async def main(message: cl.Message):
    """Process incoming chat messages for security agent."""
    await SecurityAgentProfile.on_message(message)


@cl.on_chat_end
async def end():
    """Handle chat session cleanup for security agent."""
    await SecurityAgentProfile.on_chat_end()


if __name__ == "__main__":
    # Note: Chainlit apps are run with `chainlit run chainlit_app.py`
    # This block is here for documentation purposes
    pass
