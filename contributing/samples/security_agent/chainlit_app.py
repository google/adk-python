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


@cl.on_chat_start
async def start():
    """Initialize a new chat session."""
    # Create ADK session
    session_id = create_adk_session()
    cl.user_session.set("session_id", session_id)

    # Send welcome message
    welcome_msg = """# 🔒 GCP Security Intelligence Platform

Welcome! I'm your AI security agent powered by **ADK** with access to **32 specialized tools** across 7 categories:

## 🛠️ Available Capabilities

**🔍 BigQuery Analysis (12 tools)**
- Security insights summary & statistics
- Custom SQL queries with cost analysis
- Table exploration & data sampling

**🎯 Service Evaluation (3 tools)**
- New service security assessment
- Compliance checking (PCI, HIPAA, SOC2)
- Security controls inventory

**☁️ Service Discovery (8 tools)**
- GCP service onboarding from documentation
- Resource enumeration & analysis
- API specification learning

**📚 Confluence Documentation (5 tools)**
- Search across security documentation
- Gap analysis & coverage reports
- Document retrieval & statistics

**📰 Feed & Release Analysis (4 tools)**
- GCP release notes monitoring
- Security RSS feed aggregation
- Threat intelligence tracking

## 💬 Example Questions

- "Show me all critical security findings from the last 24 hours"
- "Analyze the security posture of Cloud Run"
- "Find IAM accounts with admin privileges"
- "Search Confluence for data encryption policies"
- "What are the latest GCP security updates?"
- "Evaluate BigQuery for PCI-DSS compliance"

---

**Connected to:** ADK Backend (localhost:8000)
**Session ID:** `{}`

What would you like to explore?
""".format(session_id[:8] + "...")

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
