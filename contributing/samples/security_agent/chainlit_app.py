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
            markdown_description="🔒 **GCP Security Intelligence** - Access to 32 security tools across BigQuery, compliance, service discovery, and documentation.",
            icon="https://api.iconify.design/mdi/shield-check.svg?color=%234285f4",
        ),
        cl.ChatProfile(
            name="Compliance Expert",
            markdown_description="✅ **Compliance & Audit** - Specialized in PCI-DSS, HIPAA, SOC2 compliance checking and security controls.",
            icon="https://api.iconify.design/mdi/certificate.svg?color=%2334a853",
        ),
        cl.ChatProfile(
            name="Service Discovery",
            markdown_description="☁️ **GCP Service Discovery** - Onboard new services, analyze resources, and explore API specifications.",
            icon="https://api.iconify.design/mdi/cloud-search.svg?color=%23fbbc04",
        ),
        cl.ChatProfile(
            name="Documentation Search",
            markdown_description="📚 **Confluence & Docs** - Search security documentation, gap analysis, and knowledge retrieval.",
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

Welcome! I'm your AI security agent with access to **32 specialized tools** across 7 categories:

**🔍 BigQuery Analysis** • **🎯 Service Evaluation** • **☁️ Service Discovery**
**📚 Confluence Docs** • **📰 Feed Analysis** • **🔐 Compliance** • **🛡️ Threat Intel**

## 💬 Example Questions
- "Show me all critical security findings from the last 24 hours"
- "Analyze the security posture of Cloud Run"
- "Find IAM accounts with admin privileges"
- "What are the latest GCP security updates?"

**Session ID:** `{}`""",

        "Compliance Expert": """# ✅ Compliance & Audit Expert

I specialize in **compliance frameworks** and **security controls** with these capabilities:

**✓ PCI-DSS Compliance** - Payment card security standards
**✓ HIPAA Compliance** - Healthcare data protection
**✓ SOC2 Compliance** - Service organization controls
**✓ Security Controls** - Inventory and validation

## 💬 Example Questions
- "Evaluate BigQuery for PCI-DSS compliance"
- "Check Cloud Storage HIPAA compliance status"
- "List all security controls for Cloud Run"
- "Generate SOC2 compliance report for GKE"

**Session ID:** `{}`""",

        "Service Discovery": """# ☁️ GCP Service Discovery Agent

I help you **onboard and analyze** GCP services with these capabilities:

**🔎 Service Onboarding** - Learn from GCP documentation
**📋 Resource Enumeration** - Discover and catalog resources
**🔌 API Exploration** - Analyze service specifications
**🏗️ Architecture Analysis** - Map service dependencies

## 💬 Example Questions
- "Onboard Cloud Run service from documentation"
- "List all resources in project for Cloud Storage"
- "What APIs are available for BigQuery?"
- "Analyze the architecture of GKE"

**Session ID:** `{}`""",

        "Documentation Search": """# 📚 Confluence & Documentation Search

I search and analyze **security documentation** with these capabilities:

**🔍 Documentation Search** - Find policies and procedures
**📊 Gap Analysis** - Identify missing documentation
**📈 Coverage Reports** - Track documentation completeness
**📄 Document Retrieval** - Access specific resources

## 💬 Example Questions
- "Search Confluence for data encryption policies"
- "Find documentation gaps in security policies"
- "Get the incident response runbook"
- "Show documentation coverage statistics"

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
