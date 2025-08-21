#!/usr/bin/env python3
"""Test Vertex AI authentication"""

import os
from pathlib import Path

# Set credentials
creds_path = Path(__file__).parent / "mgm-digitalconcierge-8ba3b2f28e5f.json"
if creds_path.exists():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(creds_path)
    print(f"✅ Credentials file found: {creds_path}")
else:
    print(f"❌ Credentials file not found: {creds_path}")

# Test Vertex AI initialization
try:
    import vertexai
    vertexai.init(project="mgm-digitalconcierge", location="us-central1")
    print("✅ Vertex AI initialized successfully")
except Exception as e:
    print(f"❌ Vertex AI initialization failed: {e}")

# Test with Google GenAI client
try:
    from google import genai
    client = genai.Client(
        vertexai=True,
        project="mgm-digitalconcierge", 
        location="us-central1"
    )
    print("✅ Google GenAI client created successfully")
except Exception as e:
    print(f"❌ Google GenAI client failed: {e}")

# Test with ADK Agent
try:
    from google.adk import Agent
    test_agent = Agent(
        name="test",
        model="gemini-2.0-flash-exp",
        vertexai=True,
        project="mgm-digitalconcierge",
        location="us-central1",
        instruction="You are a test agent"
    )
    print("✅ ADK Agent created successfully")
except Exception as e:
    print(f"❌ ADK Agent creation failed: {e}")