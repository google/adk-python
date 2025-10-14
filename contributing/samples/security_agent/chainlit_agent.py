"""
Modular Chainlit Agent for GCP Security Intelligence Platform

This module provides a plug-and-play security agent that can be integrated
into any existing Chainlit application using the chat profiles decorator.

Usage in existing Chainlit app:
    from chainlit_agent import SecurityAgentProfile

    @cl.set_chat_profiles
    async def chat_profile():
        profiles = []

        # Add your existing profiles
        profiles.extend(your_existing_profiles())

        # Add security agent profiles
        profiles.extend(SecurityAgentProfile.get_profiles())

        return profiles

    @cl.on_chat_start
    async def start():
        profile = cl.user_session.get("chat_profile")

        # Handle security agent profiles
        if SecurityAgentProfile.is_security_profile(profile):
            await SecurityAgentProfile.on_chat_start()
        else:
            # Handle your existing profiles
            await your_existing_handler()

    @cl.on_message
    async def main(message: cl.Message):
        profile = cl.user_session.get("chat_profile")

        # Route to security agent
        if SecurityAgentProfile.is_security_profile(profile):
            await SecurityAgentProfile.on_message(message)
        else:
            # Handle your existing profiles
            await your_existing_handler(message)
"""

from __future__ import annotations

import os
import uuid
from typing import Dict, List, Optional

import chainlit as cl
import requests
from dotenv import load_dotenv

from backend.api.agent_observability import log_interaction

# Load environment variables
load_dotenv()


class SecurityAgentProfile:
    """Modular Security Agent for Chainlit integration."""

    # Configuration
    ADK_BASE_URL = os.getenv("ADK_BASE_URL", "http://localhost:8000")
    ADK_SESSION_URL = f"{ADK_BASE_URL}/apps/agents/users/web-user/sessions"
    ADK_RUN_URL = f"{ADK_BASE_URL}/run"

    # Profile identifiers (customize these to avoid conflicts)
    PROFILE_NAMES = [
        "GCP Security Agent",
        "GCP Compliance Expert",
        "GCP Service Discovery",
        "GCP Documentation Search"
    ]

    @classmethod
    def get_profiles(cls) -> List[cl.ChatProfile]:
        """Return list of security agent chat profiles."""
        return [
            cl.ChatProfile(
                name="GCP Security Agent",
                markdown_description="🔒 **Unified Security Agent** - One agent with ALL 32 tools. General security analysis, compliance, discovery, and documentation.",
                icon="https://api.iconify.design/mdi/shield-check.svg?color=%234285f4",
            ),
            cl.ChatProfile(
                name="GCP Compliance Expert",
                markdown_description="✅ **Start with Compliance** - Same agent, compliance-focused examples. Has ALL tools (BigQuery, discovery, docs, etc.).",
                icon="https://api.iconify.design/mdi/certificate.svg?color=%2334a853",
            ),
            cl.ChatProfile(
                name="GCP Service Discovery",
                markdown_description="☁️ **Start with Service Discovery** - Same agent, discovery-focused examples. Has ALL tools (compliance, BigQuery, docs, etc.).",
                icon="https://api.iconify.design/mdi/cloud-search.svg?color=%23fbbc04",
            ),
            cl.ChatProfile(
                name="GCP Documentation Search",
                markdown_description="📚 **Start with Documentation** - Same agent, docs-focused examples. Has ALL tools (compliance, discovery, BigQuery, etc.).",
                icon="https://api.iconify.design/mdi/book-search.svg?color=%23ea4335",
            ),
        ]

    @classmethod
    def is_security_profile(cls, profile_name: Optional[str]) -> bool:
        """Check if the given profile is a security agent profile."""
        return profile_name in cls.PROFILE_NAMES

    @classmethod
    def create_adk_session(cls) -> str:
        """Create a new ADK session and return its identifier."""
        try:
            response = requests.post(cls.ADK_SESSION_URL, timeout=10)
            if response.ok:
                payload = response.json()
                return payload.get("id", str(uuid.uuid4()))
        except requests.RequestException:
            print("Unable to create ADK session; using random fallback")
        return str(uuid.uuid4())

    @classmethod
    def extract_text_from_adk_response(cls, result: object) -> str:
        """Extract the textual content from the ADK response payload."""
        response_text = ""
        if isinstance(result, list):
            for event in result:
                if isinstance(event, dict):
                    content = event.get("content")
                    if isinstance(content, dict):
                        parts = content.get("parts", [])
                        for part in parts:
                            if isinstance(part, dict) and "text" in part:
                                response_text += part["text"]
        return response_text or "No response from agent. Please try again."

    @classmethod
    async def run_agent_interaction(cls, message: str, session_id: str) -> str:
        """Send a message to the ADK backend and return the response text."""
        if not message:
            raise ValueError("No message provided")

        payload = {
            "appName": "agents",
            "userId": "web-user",
            "sessionId": session_id,
            "newMessage": {
                "parts": [{"text": message}],
                "role": "user",
            },
        }

        try:
            response = requests.post(cls.ADK_RUN_URL, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return cls.extract_text_from_adk_response(result)
        except requests.Timeout:
            return "⏱️ Request timed out. The agent may still be processing. Please try again."
        except requests.RequestException as e:
            return f"❌ Error communicating with agent: {str(e)}"

    @classmethod
    def get_welcome_message(cls, profile_name: str, session_id: str) -> str:
        """Get the welcome message for a specific profile."""
        agent_welcomes = {
            "GCP Security Agent": """# 🔒 GCP Security Intelligence Platform

I'm your unified AI security agent with **32 specialized tools** across all security domains:

**🔍 BigQuery Analysis** • **🎯 Service Evaluation** • **☁️ Service Discovery**
**📚 Confluence Docs** • **📰 Feed Analysis** • **🔐 Compliance** • **🛡️ Threat Intel**

## 💬 Try These Questions
- "Show me critical security findings from the last 24 hours"
- "Evaluate BigQuery for PCI-DSS compliance"
- "Search Confluence for data encryption policies"
- "What are the latest GCP security updates?"
- "Onboard Cloud Run service from documentation"
- "Find IAM accounts with admin privileges"

I can help with **security analysis, compliance checking, service discovery, documentation search, and more** - just ask!

**Session ID:** `{}`""",

            "GCP Compliance Expert": """# ✅ Compliance & Security Controls

Start here for **compliance and audit** questions. I have full access to:

**All 32 Tools:** BigQuery • Service Evaluation • Service Discovery • Confluence • Feeds • Compliance • Threat Intel

## 💬 Suggested Compliance Questions
- "Evaluate BigQuery for PCI-DSS compliance"
- "Check Cloud Storage HIPAA compliance status"
- "List all security controls for Cloud Run"
- "Generate SOC2 compliance report for GKE"

## 💬 But I Can Also Help With
- Security data analysis ("Show critical findings from last week")
- Service discovery ("Onboard new GCP service")
- Documentation search ("Find incident response policy")
- Threat intelligence ("Latest security updates")

**Session ID:** `{}`""",

            "GCP Service Discovery": """# ☁️ GCP Service Discovery & Analysis

Start here for **service onboarding and analysis**. I have full access to:

**All 32 Tools:** BigQuery • Service Evaluation • Service Discovery • Confluence • Feeds • Compliance • Threat Intel

## 💬 Suggested Service Discovery Questions
- "Onboard Cloud Run service from documentation"
- "List all resources in project for Cloud Storage"
- "What APIs are available for BigQuery?"
- "Analyze the architecture of GKE"

## 💬 But I Can Also Help With
- Security analysis ("Show security posture of Cloud Run")
- Compliance checking ("Evaluate service for PCI-DSS")
- Documentation search ("Find service security policies")
- Threat monitoring ("Check for service vulnerabilities")

**Session ID:** `{}`""",

            "GCP Documentation Search": """# 📚 Documentation & Knowledge Search

Start here for **documentation and policy** questions. I have full access to:

**All 32 Tools:** BigQuery • Service Evaluation • Service Discovery • Confluence • Feeds • Compliance • Threat Intel

## 💬 Suggested Documentation Questions
- "Search Confluence for data encryption policies"
- "Find documentation gaps in security policies"
- "Get the incident response runbook"
- "Show documentation coverage statistics"

## 💬 But I Can Also Help With
- Security analysis ("Query security insights")
- Compliance checking ("Evaluate for HIPAA compliance")
- Service discovery ("Onboard new GCP service")
- Threat intelligence ("Latest security advisories")

**Session ID:** `{}`""",
        }

        welcome_msg = agent_welcomes.get(
            profile_name, agent_welcomes["GCP Security Agent"]
        ).format(session_id)

        return welcome_msg

    @classmethod
    async def on_chat_start(cls):
        """Initialize a new chat session for security agent."""
        # Get selected agent profile
        chat_profile = cl.user_session.get("chat_profile")

        # Check if session already exists (to prevent duplicate sessions)
        existing_session = cl.user_session.get("security_session_id")

        if existing_session:
            # Reuse existing session
            session_id = existing_session
        else:
            # Create new ADK session
            session_id = cls.create_adk_session()
            cl.user_session.set("security_session_id", session_id)

        # Send welcome message
        welcome_msg = cls.get_welcome_message(chat_profile, session_id)
        await cl.Message(content=welcome_msg).send()

    @classmethod
    async def on_message(cls, message: cl.Message):
        """Process incoming chat messages for security agent."""
        # Get session ID
        session_id = cl.user_session.get("security_session_id")

        # Show thinking indicator
        msg = cl.Message(content="")
        await msg.send()

        # Call ADK backend
        response_text = await cls.run_agent_interaction(message.content, session_id)

        # Update message with response
        msg.content = response_text
        await msg.update()

        # Persist the exchange for observability and evaluation
        interaction_index = cl.user_session.get("interaction_index", 0) + 1
        cl.user_session.set("interaction_index", interaction_index)
        try:
            log_interaction(
                session_id=session_id,
                interaction_index=interaction_index,
                user_prompt=message.content,
                agent_response=response_text,
            )
        except Exception as exc:
            # Logging failures should not break the chat experience, but surface
            # a warning in the Chainlit console for debugging.
            print(f"[security_agent] Failed to log interaction: {exc}")

    @classmethod
    async def on_chat_end(cls):
        """Handle chat session cleanup for security agent."""
        session_id = cl.user_session.get("security_session_id")
        await cl.Message(
            content=f"👋 Chat session ended. Session ID: `{session_id[:8]}...`"
        ).send()


# Convenience function for simple integration
def register_security_agent(
    existing_profiles: List[cl.ChatProfile]
) -> List[cl.ChatProfile]:
    """
    Convenience function to add security agent profiles to existing profiles.

    Args:
        existing_profiles: List of your existing ChatProfile objects

    Returns:
        Combined list of existing + security agent profiles
    """
    return existing_profiles + SecurityAgentProfile.get_profiles()
