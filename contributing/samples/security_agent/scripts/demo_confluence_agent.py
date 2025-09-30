#!/usr/bin/env python3
"""
Demo script for Confluence integration with ADK Security Agent

This script demonstrates all the Confluence features available through the agent.
"""

import json
import requests
import time
from typing import Dict, Any

# Configuration
AGENT_URL = "http://127.0.0.1:8000"
USER_ID = "demo_user"

# Color codes for output
COLORS = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
    "BOLD": "\033[1m"
}

def print_colored(text: str, color: str = "BLUE"):
    """Print colored text."""
    print(f"{COLORS[color]}{text}{COLORS['ENDC']}")

def create_session() -> str:
    """Create a new session with the agent."""
    response = requests.post(
        f"{AGENT_URL}/apps/agents/users/{USER_ID}/sessions",
        json={"app_name": "agents"}
    )
    session_data = response.json()
    return session_data["id"]

def send_query(session_id: str, query: str) -> Dict[str, Any]:
    """Send a query to the agent and get response."""
    response = requests.post(
        f"{AGENT_URL}/run",
        json={
            "appName": "agents",
            "userId": USER_ID,
            "sessionId": session_id,
            "newMessage": {
                "parts": [{"text": query}],
                "role": "user"
            },
            "streaming": False
        }
    )
    return response.json()

def extract_agent_response(response_data: list) -> str:
    """Extract the agent's text response from the API response."""
    for item in response_data:
        if item.get("content", {}).get("parts"):
            for part in item["content"]["parts"]:
                if "text" in part and len(part["text"]) > 50:
                    # Skip short system messages
                    if "I can" not in part["text"][:50]:
                        return part["text"]
    return "No response found"

def run_demo():
    """Run the Confluence integration demo."""
    print_colored("=" * 60, "HEADER")
    print_colored("🚀 CONFLUENCE INTEGRATION DEMO", "HEADER")
    print_colored("=" * 60, "HEADER")
    print()

    # Create session
    print_colored("Creating agent session...", "BLUE")
    session_id = create_session()
    print_colored(f"✅ Session created: {session_id[:8]}...\n", "GREEN")

    # Demo queries
    queries = [
        {
            "title": "1️⃣ Search for Security Documentation",
            "query": "Search Confluence for GCP security best practices",
            "description": "Demonstrates document search capability"
        },
        {
            "title": "2️⃣ Get Confluence Statistics",
            "query": "Show me statistics about our Confluence documentation cache",
            "description": "Shows cache statistics and document counts"
        },
        {
            "title": "3️⃣ Analyze Documentation Coverage",
            "query": "Analyze our documentation coverage for these topics: IAM security, network security, data encryption, compliance, incident response",
            "description": "Analyzes gaps in documentation"
        },
        {
            "title": "4️⃣ Retrieve Specific Document",
            "query": "Get the GCP Security Best Practices Guide document from Confluence",
            "description": "Retrieves a specific document by title"
        },
        {
            "title": "5️⃣ Search by Space",
            "query": "What security policies are documented in the POLICY space?",
            "description": "Space-specific search"
        }
    ]

    for i, demo in enumerate(queries, 1):
        print_colored(f"\n{demo['title']}", "BOLD")
        print_colored(f"Description: {demo['description']}", "YELLOW")
        print_colored(f"Query: {demo['query']}", "BLUE")
        print()

        # Send query
        print("Sending query to agent...")
        start_time = time.time()
        response = send_query(session_id, demo["query"])
        elapsed = time.time() - start_time

        # Extract and display response
        agent_response = extract_agent_response(response)
        print_colored("Agent Response:", "GREEN")
        print(agent_response[:500] + "..." if len(agent_response) > 500 else agent_response)
        print(f"\n⏱️ Response time: {elapsed:.2f} seconds")

        if i < len(queries):
            print_colored("\n" + "-" * 60, "HEADER")
            input("Press Enter to continue to the next demo...")

    # Summary
    print_colored("\n" + "=" * 60, "HEADER")
    print_colored("✅ DEMO COMPLETED", "HEADER")
    print_colored("=" * 60, "HEADER")
    print()
    print_colored("Key Features Demonstrated:", "BOLD")
    print("  ✓ Document search with caching")
    print("  ✓ Statistics and metrics retrieval")
    print("  ✓ Coverage analysis and gap detection")
    print("  ✓ Specific document retrieval")
    print("  ✓ Space-based filtering")
    print()
    print_colored("Next Steps:", "YELLOW")
    print("  1. Configure real Confluence credentials in .env")
    print("  2. Deploy Cloud Function for BigQuery sync")
    print("  3. Run: ./cloud_functions/confluence_sync/deploy.sh")
    print("  4. Query data in BigQuery for advanced analytics")

if __name__ == "__main__":
    # Check if agent is running
    try:
        response = requests.get(f"{AGENT_URL}/docs", timeout=2)
        if response.status_code == 200:
            run_demo()
        else:
            print_colored("❌ Agent not responding properly", "RED")
    except requests.exceptions.RequestException:
        print_colored("❌ ADK agent is not running!", "RED")
        print_colored("Start it with: adk web", "YELLOW")