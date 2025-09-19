#!/usr/bin/env python3
"""
Direct test of the ADK agent to verify it works
"""

import os
import sys
from pathlib import Path

# Add the agents directory to the path
sys.path.insert(0, str(Path(__file__).parent / "agents"))

# Set environment variables
os.environ["GOOGLE_CLOUD_PROJECT"] = "mgm-digitalconcierge"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path(__file__).parent / "config" / "mgm-digitalconcierge-8e6bb83a7e22.json")
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
os.environ["DATABASE_PATH"] = str(Path(__file__).parent / "backend" / "cache" / "gcp_data.db")
os.environ["ADK_AGENT_MODEL"] = "gemini-2.0-flash-exp"

# Import and test the agent
from adk_agent import agent

print("Agent loaded successfully!")
print(f"Agent name: {agent.name}")
print(f"Agent description: {agent.description}")
print(f"Agent tools: {[tool.name for tool in agent.tools]}")

# Test a simple query
print("\n" + "="*50)
print("Testing agent with query: 'How many security findings are there?'")
print("="*50)

try:
    response = agent.say("How many security findings are there?")
    print(f"\nAgent response: {response}")
except Exception as e:
    print(f"\nError: {e}")