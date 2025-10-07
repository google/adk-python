#!/usr/bin/env python3
"""
Example: Integrating GCP Security Agent into an existing Chainlit application

This example shows how to add the security agent profiles to your existing Chainlit app
without modifying your existing code.
"""

import chainlit as cl
from chainlit_agent import SecurityAgentProfile, register_security_agent


# Your existing chat profiles
def get_my_existing_profiles() -> list:
    """Your existing chat profile definitions."""
    return [
        cl.ChatProfile(
            name="General Assistant",
            markdown_description="💬 **General AI Assistant** - Help with various tasks",
            icon="https://api.iconify.design/mdi/robot.svg?color=%23000000",
        ),
        cl.ChatProfile(
            name="Code Helper",
            markdown_description="👨‍💻 **Code Assistant** - Help with programming",
            icon="https://api.iconify.design/mdi/code-braces.svg?color=%23000000",
        ),
    ]


# OPTION 1: Simple integration using convenience function
@cl.set_chat_profiles
async def chat_profile():
    """Define chat profiles including security agent."""
    # Get your existing profiles
    my_profiles = get_my_existing_profiles()

    # Add security agent profiles (one line!)
    return register_security_agent(my_profiles)


# OPTION 2: Manual integration for more control
@cl.set_chat_profiles
async def chat_profile_manual():
    """Define chat profiles with manual control."""
    profiles = []

    # Add your existing profiles
    profiles.extend(get_my_existing_profiles())

    # Add security agent profiles
    profiles.extend(SecurityAgentProfile.get_profiles())

    # Could add more profiles here...

    return profiles


@cl.on_chat_start
async def start():
    """Initialize chat session - routes to appropriate handler."""
    profile = cl.user_session.get("chat_profile")

    # Route to security agent
    if SecurityAgentProfile.is_security_profile(profile):
        await SecurityAgentProfile.on_chat_start()

    # Route to your existing handlers
    elif profile == "General Assistant":
        await cl.Message(content="👋 I'm your general assistant!").send()

    elif profile == "Code Helper":
        await cl.Message(content="👨‍💻 Ready to help with coding!").send()

    else:
        # Default handler
        await cl.Message(content="Hello! How can I help you?").send()


@cl.on_message
async def main(message: cl.Message):
    """Process messages - routes to appropriate handler."""
    profile = cl.user_session.get("chat_profile")

    # Route to security agent
    if SecurityAgentProfile.is_security_profile(profile):
        await SecurityAgentProfile.on_message(message)

    # Route to your existing handlers
    elif profile == "General Assistant":
        # Your general assistant logic
        await cl.Message(content=f"General assistant received: {message.content}").send()

    elif profile == "Code Helper":
        # Your code helper logic
        await cl.Message(content=f"Code helper received: {message.content}").send()

    else:
        # Default handler
        await cl.Message(content=f"Echo: {message.content}").send()


@cl.on_chat_end
async def end():
    """Handle session cleanup - routes to appropriate handler."""
    profile = cl.user_session.get("chat_profile")

    # Route to security agent cleanup
    if SecurityAgentProfile.is_security_profile(profile):
        await SecurityAgentProfile.on_chat_end()
    else:
        # Your cleanup logic
        await cl.Message(content="👋 Goodbye!").send()


if __name__ == "__main__":
    # Run with: chainlit run examples/chainlit_integration_example.py
    pass
