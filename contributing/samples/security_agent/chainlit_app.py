#!/usr/bin/env python3
"""Chainlit web application for the BigQuery Security Agent.

This provides a modern chat interface using Chainlit that connects to the
ADK backend (port 8000) with the security agent and 32 tools.
"""

from __future__ import annotations

import os
import uuid
from typing import Dict, Optional

import chainlit as cl
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ADK_BASE_URL = os.getenv("ADK_BASE_URL", "http://localhost:8000")
ADK_SESSION_URL = f"{ADK_BASE_URL}/apps/agents/users/web-user/sessions"
ADK_RUN_URL = f"{ADK_BASE_URL}/run"


def create_adk_session() -> str:
    """Create a new ADK session and return its identifier."""
    try:
        response = requests.post(ADK_SESSION_URL, timeout=10)
        if response.ok:
            payload = response.json()
            return payload.get("id", str(uuid.uuid4()))
    except requests.RequestException:
        print("Unable to create ADK session; using random fallback")
    return str(uuid.uuid4())


def extract_text_from_adk_response(result: object) -> str:
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


async def run_agent_interaction(message: str, session_id: str) -> str:
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
        response = requests.post(ADK_RUN_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return extract_text_from_adk_response(result)
    except requests.Timeout:
        return "⏱️ Request timed out. The agent may still be processing. Please try again."
    except requests.RequestException as e:
        return f"❌ Error communicating with agent: {str(e)}"


@cl.set_chat_profiles
async def chat_profile():
    """Define multiple agent profiles for the dropdown selector."""
    return [
        cl.ChatProfile(
            name="Security Agent",
            markdown_description="🔒 **Unified Security Agent** - One agent with ALL 32 tools. General security analysis, compliance, discovery, and documentation.",
            icon="https://api.iconify.design/mdi/shield-check.svg?color=%234285f4",
        ),
        cl.ChatProfile(
            name="Compliance Expert",
            markdown_description="✅ **Start with Compliance** - Same agent, compliance-focused examples. Has ALL tools (BigQuery, discovery, docs, etc.).",
            icon="https://api.iconify.design/mdi/certificate.svg?color=%2334a853",
        ),
        cl.ChatProfile(
            name="Service Discovery",
            markdown_description="☁️ **Start with Service Discovery** - Same agent, discovery-focused examples. Has ALL tools (compliance, BigQuery, docs, etc.).",
            icon="https://api.iconify.design/mdi/cloud-search.svg?color=%23fbbc04",
        ),
        cl.ChatProfile(
            name="Documentation Search",
            markdown_description="📚 **Start with Documentation** - Same agent, docs-focused examples. Has ALL tools (compliance, discovery, BigQuery, etc.).",
            icon="https://api.iconify.design/mdi/book-search.svg?color=%23ea4335",
        ),
    ]


@cl.on_chat_start
async def start():
    """Initialize a new chat session."""
    # Get selected agent profile
    chat_profile = cl.user_session.get("chat_profile")

    # Create ADK session
    session_id = create_adk_session()
    cl.user_session.set("session_id", session_id)

    # Customize welcome message based on selected agent profile
    agent_welcomes = {
        "Security Agent": """# 🔒 GCP Security Intelligence Platform

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

        "Compliance Expert": """# ✅ Compliance & Security Controls

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

        "Service Discovery": """# ☁️ GCP Service Discovery & Analysis

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

        "Documentation Search": """# 📚 Documentation & Knowledge Search

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

    # Get appropriate welcome message
    welcome_msg = agent_welcomes.get(
        chat_profile, agent_welcomes["Security Agent"]
    ).format(session_id[:8] + "...")

    await cl.Message(content=welcome_msg).send()


@cl.on_message
async def main(message: cl.Message):
    """Process incoming chat messages."""
    # Get session ID
    session_id = cl.user_session.get("session_id")

    # Show thinking indicator
    msg = cl.Message(content="")
    await msg.send()

    # Call ADK backend
    response_text = await run_agent_interaction(message.content, session_id)

    # Update message with response
    msg.content = response_text
    await msg.update()


@cl.on_chat_end
async def end():
    """Handle chat session cleanup."""
    session_id = cl.user_session.get("session_id")
    await cl.Message(
        content=f"👋 Chat session ended. Session ID: `{session_id[:8]}...`"
    ).send()


if __name__ == "__main__":
    # Note: Chainlit apps are run with `chainlit run chainlit_app.py`
    # This block is here for documentation purposes
    pass
